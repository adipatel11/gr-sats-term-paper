# fits the calibrators on clean + corrupted pool images and compares them on corruption
# families they never saw (5-fold cross-fitting x leave one family out). cpu is fine
# python calibrate.py --dataset ham --models frozen
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import EPS, FAMILIES, KNN_K, N_BOOT, N_FOLDS, OUT, SEEDS
from data import GROUPS, LOADERS
from metrics import brier, ece

CORRUPTIONS = [c for fam in FAMILIES.values() for c in fam]

# temperature form and loss for each calibrator. global = one inverse temperature,
# shift = a0 + a1 * s, bias = shift plus b0 + b1 * s added to the logit
METHODS = {
    "ts_clean": ("global", "clean"),
    "ts_corr": ("global", "avg"),
    "ts_wg": ("global", "wg"),
    "sats_avg": ("shift", "avg"),
    "grsats": ("shift", "wg"),
    "grsats_bias": ("bias", "wg"),
}


def knn_distance(bank, queries):
    # distance from each query to its k-th nearest clean training image, l2-normalized features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = torch.nn.functional.normalize(torch.tensor(bank, dtype=torch.float32, device=device), dim=1)
    out = []
    for i in range(0, len(queries), 4096):
        q = torch.tensor(queries[i:i + 4096], dtype=torch.float32, device=device)
        cos = torch.nn.functional.normalize(q, dim=1) @ bank.T
        kth = cos.topk(KNN_K, dim=1).values[:, -1]
        out.append(torch.sqrt((2 - 2 * kth).clamp(min=1e-12)).cpu())
    return torch.cat(out).numpy()


def load_outputs(ds, df, pool, bb, kind, seed, distances=True):
    # clean logits z0 and distances d0 are (n,), corrupted zc and dc are (15, 5, n)
    train = np.flatnonzero(df["split"] == "train")
    if kind == "frozen":
        X = np.load(OUT / f"{ds}_{bb}_features.npz")["X"]
        z0 = np.load(OUT / f"{ds}_{bb}_lp_seed{seed}.npz")["logits"][pool]
        probe = joblib.load(OUT / f"{ds}_{bb}_lp_seed{seed}.joblib")
        key = f"{bb}_frozen"
    else:
        f = np.load(OUT / f"{ds}_{bb}_lora_seed{seed}.npz")
        X, z0 = f["X"], f["logits"][pool]
        key = f"{bb}_lora{seed}"

    n = len(pool)
    zc, dc = np.zeros((15, 5, n)), np.zeros((15, 5, n))
    for j, c in enumerate(CORRUPTIONS):
        f = np.load(OUT / f"{ds}_c_{c}_{kind}.npz")
        assert (f["ids"] == df["id"].to_numpy(dtype=str)[pool]).all()
        Xc = f[key + "_X"].astype(np.float32)
        flat = Xc.reshape(-1, Xc.shape[-1])
        zc[j] = probe.decision_function(flat).reshape(5, n) if kind == "frozen" else f[key + "_logits"]
        if distances:
            dc[j] = knn_distance(X[train], flat).reshape(5, n)
    d0 = knn_distance(X[train], X[pool]) if distances else None
    return z0.astype(float), d0, zc, (dc if distances else None)


def stack_rows(z0, d0, zc, dc, imgs, corrs):
    # fitting rows: the clean images, then their copies under each corruption and severity
    z, ld, sev, img = [z0[imgs]], [np.log(d0[imgs])], [np.zeros(len(imgs), int)], [imgs]
    for j in corrs:
        for l in range(5):
            z.append(zc[j, l, imgs])
            ld.append(np.log(dc[j, l, imgs]))
            sev.append(np.full(len(imgs), l + 1))
            img.append(imgs)
    return np.concatenate(z), np.concatenate(ld), np.concatenate(sev), np.concatenate(img)


def design(z, s, form):
    if form == "global":
        return z[:, None]
    if form == "shift":
        return np.stack([z, s * z], 1)
    return np.stack([z, s * z, np.ones_like(z), s], 1)


def positivity(form, smin, smax):
    # the inverse temperature has to stay above EPS at both ends of the score range
    if form == "global":
        return np.array([[1.0]])
    C = np.array([[1, smin], [1, smax]])
    return C if form == "shift" else np.hstack([C, np.zeros((2, 2))])


def weights(loss, sev, masks):
    # one row per group. each severity counts 1/6 of the group's loss, split evenly over its images
    if loss == "clean":
        return ((sev == 0) / (sev == 0).sum())[None]
    if loss == "avg":
        masks = [np.ones(len(sev), bool)]
    W = np.zeros((len(masks), len(sev)))
    for g, m in enumerate(masks):
        for l in range(6):
            rows = m & (sev == l)
            W[g, rows] = 1 / (6 * rows.sum())
    return W


def fit(A, y, W, C):
    # minimize t subject to L_g(theta) <= t for every group and C @ theta >= EPS (epigraph form).
    # each L_g is a weighted logistic loss of A @ theta, so the problem is convex
    p = A.shape[1]

    def losses(theta):
        u = A @ theta
        return W @ (np.logaddexp(0, u) - y * u)

    def grads(theta):
        return (W * (expit(A @ theta) - y)) @ A

    theta0 = np.zeros(p)
    theta0[0] = 1  # plain logits to start
    x0 = np.append(theta0, losses(theta0).max())
    cons = [{"type": "ineq", "fun": lambda x: x[-1] - losses(x[:-1]),
             "jac": lambda x: np.hstack([-grads(x[:-1]), np.ones((len(W), 1))])},
            {"type": "ineq", "fun": lambda x: C @ x[:-1] - EPS,
             "jac": lambda x: np.hstack([C, np.zeros((len(C), 1))])}]
    res = minimize(lambda x: x[-1], x0, jac=lambda x: np.eye(p + 1)[-1], constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})
    assert res.success, res.message
    return res.x[:-1]


def fit_method(form, loss, z0, d0, zc, dc, y, masks, imgs, corrs):
    # fits one calibrator and returns a function that predicts from logits and distances
    z, ld, sev, img = stack_rows(z0, d0, zc, dc, imgs, corrs)
    m, r = ld[sev == 0].mean(), ld[sev == 0].std()
    s = (ld - m) / r
    smin, smax = s.min(), s.max()
    theta = fit(design(z, s, form), y[img], weights(loss, sev, [g[img] for g in masks]),
                positivity(form, smin, smax))

    def predict(zz, dd):
        ss = np.clip((np.log(dd) - m) / r, smin, smax)
        return expit(design(zz.ravel(), ss.ravel(), form) @ theta).reshape(zz.shape)
    return predict, theta


def make_folds(dp, ds):
    # stratified by group and label, all images of a lesion in one fold
    strata = dp[GROUPS[ds]].astype(str).agg("_".join, axis=1) + "_" + dp["y"].astype(str)
    folds = np.zeros(len(dp), int)
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0)
    for f, (_, test) in enumerate(cv.split(dp, strata, dp["lesion"])):
        folds[test] = f
    return folds


def crossfit(z0, d0, zc, dc, y, masks, folds):
    # every pool image gets one clean prediction per held-out family, and one prediction per
    # corruption from the calibrator that held out that corruption's family
    fams = list(FAMILIES)
    q_clean = {"none": np.tile(expit(z0), (len(fams), 1))}
    q_corr = {"none": expit(zc)}
    for name in METHODS:
        q_clean[name], q_corr[name] = np.zeros((len(fams), len(y))), np.zeros(zc.shape)

    for f in range(N_FOLDS):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        for h, fam in enumerate(fams):
            held = [j for j, c in enumerate(CORRUPTIONS) if c in FAMILIES[fam]]
            rest = [j for j in range(15) if j not in held]
            for name, (form, loss) in METHODS.items():
                predict, _ = fit_method(form, loss, z0, d0, zc, dc, y, masks, tr, rest)
                q_clean[name][h, te] = predict(z0[te], d0[te])
                for j in held:
                    q_corr[name][j][:, te] = predict(zc[j][:, te], dc[j][:, te])
    return q_clean, q_corr


def nll(y, q):
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def evaluate(q_clean, q_corr, y, masks, names, eval_c):
    # main test pool: held-out corruptions at severities 3 to 5
    qc = q_corr[eval_c][:, 2:5]
    loss, sq = nll(y, qc), (qc - y) ** 2
    row = {"nll": loss.mean()}
    for g, m in zip(names, masks):
        row[f"nll {g}"] = loss[..., m].mean()
        row[f"brier {g}"] = sq[..., m].mean()
    row["wg_nll"] = max(row[f"nll {g}"] for g in names)
    row["wg_brier"] = max(row[f"brier {g}"] for g in names)
    row["clean_ece"] = np.mean([ece(y, q) for q in q_clean])
    row["clean_auc"] = np.mean([roc_auc_score(y, q) for q in q_clean])
    row["corr_auc"] = roc_auc_score(np.tile(y, qc.shape[0] * 3), qc.ravel())
    # worst-group nll at each severity, 0 is clean
    for l in range(6):
        q = q_clean if l == 0 else q_corr[eval_c][:, l - 1]
        row[f"wg_nll sev{l}"] = max(nll(y, q)[..., m].mean() for m in masks)
    return row


def bootstrap_wg(y, masks, lesions, qa, qb, rng):
    # paired bootstrap over lesions of worst-group nll(a) - worst-group nll(b) on the main pool
    # every image has the same number of rows, so averaging per image first gives the same group means
    sa, sb = nll(y, qa).reshape(-1, len(y)).mean(0), nll(y, qb).reshape(-1, len(y)).mean(0)
    _, lid = np.unique(lesions, return_inverse=True)
    draws = rng.integers(lid.max() + 1, size=(N_BOOT, lid.max() + 1))
    counts = np.stack([np.bincount(d, minlength=lid.max() + 1) for d in draws])[:, lid]
    wg = lambda s: np.max([(counts[:, m] @ s[m]) / counts[:, m].sum(1) for m in masks], axis=0)
    diff = wg(sa) - wg(sb)
    return diff.mean(), diff.std(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def pilot_check(z0, d0, zc, dc, y, masks, names, eval_c, lesions, rng):
    # go/no-go check from the paper: fit avg and worst-group directly on all pool images and see
    # if the worst-group fit lowers worst-group nll by more than one bootstrap standard error
    everything = np.arange(len(y))
    rows = {}
    for name in ["ts_corr", "ts_wg", "sats_avg", "grsats"]:
        predict, theta = fit_method(*METHODS[name], z0, d0, zc, dc, y, masks, everything, list(range(15)))
        rows[name] = (predict(zc[eval_c][:, 2:5], dc[eval_c][:, 2:5]), theta)
    out = {}
    for a, b in [("ts_wg", "ts_corr"), ("grsats", "sats_avg")]:
        mean, se, _, _ = bootstrap_wg(y, masks, lesions, rows[a][0], rows[b][0], rng)
        out[f"{a} - {b}"] = (mean, se)
    out["grsats theta"] = rows["grsats"][1]
    # mean shift score per group, clean vs severities 3 to 5
    ld0, ldc = np.log(d0), np.log(dc[eval_c][:, 2:5])
    m, r = ld0.mean(), ld0.std()
    for g, mask in zip(names, masks):
        out[f"score {g}"] = (((ld0 - m) / r)[mask].mean(), ((ldc - m) / r)[..., mask].mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
    parser.add_argument("--models", choices=["frozen", "lora"], required=True)
    args = parser.parse_args()
    ds, kind = args.dataset, args.models

    df = LOADERS[ds]()
    pool = np.flatnonzero(df["split"].isin(["val", "test"]))
    dp = df.iloc[pool].reset_index(drop=True)
    y = dp["y"].to_numpy().astype(float)
    names = [v for col in GROUPS[ds] for v in sorted(dp[col].unique())]
    masks = [(dp[col] == v).to_numpy() for col in GROUPS[ds] for v in sorted(dp[col].unique())]
    folds = make_folds(dp, ds)
    # task-specific corruptions imitate dermoscope artifacts, so fitz leaves them out of the main pool
    eval_c = [j for j, c in enumerate(CORRUPTIONS) if ds == "ham" or c not in FAMILIES["task-specific"]]
    rng = np.random.default_rng(0)

    results, boots, pilots = [], [], []
    for bb in ["dinov2", "biomedclip"]:
        for seed in SEEDS:
            # frozen features are the same for every probe seed, so the distances are too
            new_d = kind == "lora" or seed == SEEDS[0]
            z0, d0_new, zc, dc_new = load_outputs(ds, df, pool, bb, kind, seed, distances=new_d)
            if new_d:
                d0, dc = d0_new, dc_new
            model = f"{bb} {'lp' if kind == 'frozen' else 'lora'}"
            print(f"\n{model} seed {seed}")

            q_clean, q_corr = crossfit(z0, d0, zc, dc, y, masks, folds)
            for name in q_clean:
                results.append({"model": model, "seed": seed, "method": name,
                                **evaluate(q_clean[name], q_corr[name], y, masks, names, eval_c)})
            for other in ["ts_corr", "ts_clean", "sats_avg", "ts_wg"]:
                mean, se, lo, hi = bootstrap_wg(y, masks, dp["lesion"].to_numpy(), q_corr["grsats"][eval_c][:, 2:5],
                                                q_corr[other][eval_c][:, 2:5], rng)
                boots.append({"model": model, "seed": seed, "comparison": f"grsats - {other}",
                              "wg_nll diff": mean, "ci low": lo, "ci high": hi})

            p = pilot_check(z0, d0, zc, dc, y, masks, names, eval_c, dp["lesion"].to_numpy(), rng)
            pilots.append({"model": model, "seed": seed, **{k: v for k, v in p.items()}})
            for k, v in p.items():
                print(f"  {k}: {np.round(v, 4)}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / f"{ds}_{kind}_calibration.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / f"{ds}_{kind}_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    main_cols = ["wg_nll", "wg_brier", "nll", "clean_ece", "clean_auc", "corr_auc"]
    sev_cols = [f"wg_nll sev{l}" for l in range(6)]
    group_cols = [f"nll {g}" for g in names]
    for cols, title in [(main_cols, "held-out families, severities 3 to 5"),
                        (group_cols, "nll per group, same pool"),
                        (sev_cols, "worst-group nll by severity (0 = clean)")]:
        print(f"\n{ds} {kind}: {title}, mean over seeds")
        print(res.groupby(["model", "method"], sort=False)[cols].mean().round(4).to_string())

    print(f"\n{ds} {kind}: paired bootstrap of worst-group nll differences (negative = grsats better)")
    print(pd.DataFrame(boots).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
