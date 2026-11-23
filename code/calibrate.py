# fits the calibrators on clean + corrupted pool images and compares them on corruption
# families they never saw (5-fold cross-fitting x leave one family out). cpu is fine
# python calibrate.py --dataset ham --models frozen
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import EPS, FAMILIES, KNN_K, N_BOOT, N_FOLDS, OUT, SEEDS
from data import GROUPS, LOADERS
from metrics import ece

CORRUPTIONS = [c for fam in FAMILIES.values() for c in fam]

# temperature form and loss for each calibrator. global = one inverse temperature,
# shift = a0 + a1 * s, bias = shift plus b0 + b1 * s added to the logit,
# dac and aar are the two shift-aware baselines, both fit for the average
METHODS = {
    "ts_clean": ("global", "clean"),
    "ts_corr": ("global", "avg"),
    "ts_wg": ("global", "wg"),
    "dac": ("dac", "avg"),
    "aar": ("aar", "avg"),
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


def atypicality(bank, bank_y, queries):
    # aar's input atypicality: minus the largest class log density under gaussians with one shared
    # covariance fit to the training features. constants are dropped because aar standardizes it
    means = np.stack([bank[bank_y == c].mean(0) for c in (0, 1)])
    centered = bank - means[bank_y]
    vals, vecs = np.linalg.eigh(centered.T @ centered / len(bank))
    whiten = vecs / np.sqrt(np.maximum(vals, 1e-6 * vals.max()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.tensor(whiten, dtype=torch.float32, device=device)
    M = torch.tensor(means, dtype=torch.float32, device=device) @ W
    out = []
    for i in range(0, len(queries), 4096):
        q = torch.tensor(queries[i:i + 4096], dtype=torch.float32, device=device) @ W
        maha = torch.stack([((q - m) ** 2).sum(1) for m in M], 1)
        out.append(0.5 * maha.min(1).values.cpu())
    return torch.cat(out).numpy().astype(float)


def load_outputs(ds, df, pool, bb, kind, seed, scores=True):
    # logits z, k-nn distances d and atypicality a, clean ones are (n,), corrupted (15, 5, n).
    # d and a only depend on the features, so they are skipped when scores=False
    train = np.flatnonzero(df["split"] == "train")
    train_y = df["y"].to_numpy()[train]
    if kind == "frozen":
        X = np.load(OUT / f"{ds}_{bb}_features.npz")["X"]
        z0 = np.load(OUT / f"{ds}_{bb}_lp_seed{seed}.npz")["logits"][pool]
        probe = joblib.load(OUT / f"{ds}_{bb}_lp_seed{seed}.joblib")
        key = f"{bb}_frozen"
    else:
        f = np.load(OUT / f"{ds}_{bb}_lora_seed{seed}.npz")
        X, z0 = f["X"], f["logits"][pool]
        key = f"{bb}_lora{seed}"
    X = X.astype(np.float32)

    n = len(pool)
    zc, dc, ac = np.zeros((15, 5, n)), np.zeros((15, 5, n)), np.zeros((15, 5, n))
    for j, c in enumerate(CORRUPTIONS):
        f = np.load(OUT / f"{ds}_c_{c}_{kind}.npz")
        assert (f["ids"] == df["id"].to_numpy(dtype=str)[pool]).all()
        Xc = f[key + "_X"].astype(np.float32)
        flat = Xc.reshape(-1, Xc.shape[-1])
        zc[j] = probe.decision_function(flat).reshape(5, n) if kind == "frozen" else f[key + "_logits"]
        if scores:
            dc[j] = knn_distance(X[train], flat).reshape(5, n)
            ac[j] = atypicality(X[train], train_y, flat).reshape(5, n)
    if not scores:
        return z0.astype(float), zc, None
    clean = (knn_distance(X[train], X[pool]), atypicality(X[train], train_y, X[pool]))
    return z0.astype(float), zc, (clean[0], dc, clean[1], ac)


def stack_rows(z0, zc, sc, imgs, corrs):
    # fitting rows: the clean images, then their copies under each corruption and severity
    d0, dc, a0, ac = sc
    z, d, a, sev, img = [z0[imgs]], [d0[imgs]], [a0[imgs]], [np.zeros(len(imgs), int)], [imgs]
    for j in corrs:
        for l in range(5):
            z.append(zc[j, l, imgs])
            d.append(dc[j, l, imgs])
            a.append(ac[j, l, imgs])
            sev.append(np.full(len(imgs), l + 1))
            img.append(imgs)
    return [np.concatenate(v) for v in (z, d, a, sev, img)]


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


def fit(A, y, W, C, offset=0):
    # minimize t subject to L_g(theta) - offset_g <= t for every group and C @ theta >= EPS
    # (epigraph form). each L_g is a weighted logistic loss of A @ theta, so the problem is convex
    p = A.shape[1]

    def losses(theta):
        u = A @ theta
        return W @ (np.logaddexp(0, u) - y * u) - offset

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
    if not res.success and p == 1:
        # slsqp sometimes gives up on very confident lora logits ("positive directional derivative").
        # with one parameter (plain ts) just search over log(beta) instead. the worst-group loss is a
        # max of convex functions of beta, so it has one minimum and the bounded search finds it
        print(f"  slsqp failed ({res.message}), using the 1-d search")
        return fit_1d(losses, C)
    assert res.success, res.message
    return res.x[:-1]


def fit_1d(losses, C):
    # beta has to stay above EPS / C (the positivity constraint), and 1000 is far past any real fit
    lo = np.log(EPS / C[0, 0])
    res = minimize_scalar(lambda b: losses(np.array([np.exp(b)])).max(), bounds=(lo, np.log(1000)),
                          method="bounded", options={"xatol": 1e-8})
    return np.array([np.exp(res.x)])


def fit_dac(z, d, y, w):
    # dac (tomani et al. 2023, eq. 6 to 9) with one layer: p = sigmoid(z / T), T = w1 * s + w0, where s
    # is the squared distance to the k-th neighbor (faiss returns squared distances). fit by squared
    # error with the bounds and start of the official code. the official code clips T at 1e-20, but
    # then the loss is flat there and the optimizer can get stuck at T = 0, so this keeps T >= 1e-3
    # at the smallest distance with a constraint instead (T only grows with s since w1 >= 0)
    s = d ** 2
    smin = s.min()

    def loss(p):
        return w @ (expit(z / (p[0] * s + p[1])) - y) ** 2

    def grad(p):
        T = p[0] * s + p[1]
        q = expit(z / T)
        dT = w * 2 * (q - y) * q * (1 - q) * (-z / T ** 2)
        return np.array([dT @ s, dT.sum()])

    res = minimize(loss, [1.0, 1.0], jac=grad, method="SLSQP", bounds=[(0, 1e4), (-100, 100)],
                   constraints=[{"type": "ineq", "fun": lambda p: p[0] * smin + p[1] - 1e-3,
                                 "jac": lambda p: np.array([smin, 1.0])}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    assert res.success, res.message

    def predict(zz, dd, aa):
        T = np.maximum(res.x[0] * dd ** 2 + res.x[1], 1e-3)  # a test image can be closer than any fit image
        return expit(zz / T)
    return predict, res.x


def fit_aar(z, a, y, w):
    # aar (yuksekgonul et al. 2023, eq. 3 and app. d.2): log p(y|x) ~ phi(a) log p_model(y|x) + S_y with
    # phi(a) = c0 + c1 a + c2 a^2 and a standardized on the calibration rows. for two classes this is
    # sigmoid(phi(a) z + S_1 - S_0), a logistic regression on [z, az, a^2 z, 1], fit by cross-entropy
    m, r = a.mean(), a.std()

    def design_aar(zz, aa):
        aa = (aa - m) / r
        return np.stack([zz, aa * zz, aa ** 2 * zz, np.ones_like(zz)], 1)

    A = design_aar(z, a)
    res = minimize(lambda t: w @ (np.logaddexp(0, A @ t) - y * (A @ t)), np.zeros(4),
                   jac=lambda t: (w * (expit(A @ t) - y)) @ A, method="L-BFGS-B")

    def predict(zz, dd, aa):
        return expit(design_aar(zz.ravel(), aa.ravel()) @ res.x).reshape(zz.shape)
    return predict, res.x


def fit_method(name, z0, zc, sc, y, masks, imgs, corrs, methods=METHODS):
    # fits one calibrator and returns a function that predicts from logits, distances and atypicality
    form, loss = methods[name]
    z, d, a, sev, img = stack_rows(z0, zc, sc, imgs, corrs)
    W = weights(loss, sev, [g[img] for g in masks])
    if form == "dac":
        return fit_dac(z, d, y[img], W[0])
    if form == "aar":
        return fit_aar(z, a, y[img], W[0])

    ld = np.log(d)
    m, r = ld[sev == 0].mean(), ld[sev == 0].std()
    s = (ld - m) / r
    smin, smax = s.min(), s.max()
    A, C = design(z, s, form), positivity(form, smin, smax)
    offset = 0
    if loss == "excess":
        # excess loss: each group's loss minus the lowest loss that group could get on its own
        offset = []
        for g in range(len(W)):
            u = A @ fit(A, y[img], W[g:g + 1], C)
            offset.append(W[g] @ (np.logaddexp(0, u) - y[img] * u))
        offset = np.array(offset)
    theta = fit(A, y[img], W, C, offset)

    def predict(zz, dd, aa):
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


def crossfit(z0, zc, sc, y, masks, folds):
    # every pool image gets one clean prediction per held-out family, and one prediction per
    # corruption from the calibrator that held out that corruption's family
    d0, dc, a0, ac = sc
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
            for name in METHODS:
                predict, _ = fit_method(name, z0, zc, sc, y, masks, tr, rest)
                q_clean[name][h, te] = predict(z0[te], d0[te], a0[te])
                for j in held:
                    q_corr[name][j][:, te] = predict(zc[j][:, te], dc[j][:, te], ac[j][:, te])
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
    # aar can move the 0.5 threshold, so accuracy is reported too
    row["corr_acc"] = ((qc > 0.5) == y).mean()
    # worst-group nll at each severity, 0 is clean
    for l in range(6):
        q = q_clean if l == 0 else q_corr[eval_c][:, l - 1]
        row[f"wg_nll sev{l}"] = max(nll(y, q)[..., m].mean() for m in masks)
    return row


def bootstrap_wg(y, masks, lesions, qa, qb, rng):
    # paired bootstrap over lesions of worst-group nll(a) - worst-group nll(b) on the main pool.
    # every image has the same number of rows, so averaging per image first gives the same group means
    sa, sb = nll(y, qa).reshape(-1, len(y)).mean(0), nll(y, qb).reshape(-1, len(y)).mean(0)
    _, lid = np.unique(lesions, return_inverse=True)
    draws = rng.integers(lid.max() + 1, size=(N_BOOT, lid.max() + 1))
    counts = np.stack([np.bincount(d, minlength=lid.max() + 1) for d in draws])[:, lid]
    wg = lambda s: np.max([(counts[:, m] @ s[m]) / counts[:, m].sum(1) for m in masks], axis=0)
    diff = wg(sa) - wg(sb)
    return diff.mean(), diff.std(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def pilot_check(z0, zc, sc, y, masks, names, eval_c, lesions, rng):
    # go/no-go check from the paper: fit avg and worst-group directly on all pool images and see
    # if the worst-group fit lowers worst-group nll by more than one bootstrap standard error
    d0, dc, a0, ac = sc
    everything = np.arange(len(y))
    rows = {}
    for name in ["ts_corr", "ts_wg", "sats_avg", "grsats"]:
        predict, theta = fit_method(name, z0, zc, sc, y, masks, everything, list(range(15)))
        rows[name] = (predict(zc[eval_c][:, 2:5], dc[eval_c][:, 2:5], ac[eval_c][:, 2:5]), theta)
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
    lesions = dp["lesion"].to_numpy()
    rng = np.random.default_rng(0)

    results, boots = [], []
    for bb in ["dinov2", "biomedclip"]:
        for seed in SEEDS:
            # frozen features are the same for every probe seed, so the scores are too
            new_scores = kind == "lora" or seed == SEEDS[0]
            z0, zc, sc_new = load_outputs(ds, df, pool, bb, kind, seed, scores=new_scores)
            if new_scores:
                sc = sc_new
            model = f"{bb} {'lp' if kind == 'frozen' else 'lora'}"
            print(f"\n{model} seed {seed}")

            q_clean, q_corr = crossfit(z0, zc, sc, y, masks, folds)
            for name in q_clean:
                results.append({"model": model, "seed": seed, "method": name,
                                **evaluate(q_clean[name], q_corr[name], y, masks, names, eval_c)})
            for other in ["ts_corr", "dac", "aar", "ts_clean", "sats_avg", "ts_wg"]:
                mean, se, lo, hi = bootstrap_wg(y, masks, lesions, q_corr["grsats"][eval_c][:, 2:5],
                                                q_corr[other][eval_c][:, 2:5], rng)
                boots.append({"model": model, "seed": seed, "comparison": f"grsats - {other}",
                              "wg_nll diff": mean, "ci low": lo, "ci high": hi})

            for k, v in pilot_check(z0, zc, sc, y, masks, names, eval_c, lesions, rng).items():
                print(f"  {k}: {np.round(v, 4)}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / f"{ds}_{kind}_calibration.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / f"{ds}_{kind}_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    main_cols = ["wg_nll", "wg_brier", "nll", "clean_ece", "clean_auc", "corr_auc", "corr_acc"]
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
