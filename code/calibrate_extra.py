# the extra calibrators and checks from the paper that calibrate.py doesn't run: ts with gaussian
# noise, gr-sats with excess nll, k = 1 and 50, a group-balanced reference bank, dac fit for nll and
# for the worst group (dac-wg), cluster-wise ts, multi-domain ts, and the two oracles that use the
# test image's group. it refits calibrate.py's methods too (same folds, fitting rows and held-out
# families) and checks them against results/, so every table comes from one run
# python calibrate_extra.py --dataset ham --models frozen
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans

from calibrate import (CORRUPTIONS, METHODS, atypicality, evaluate, fit, fit_method, make_folds, nll,
                       positivity, stack_rows, weights)
from config import FAMILIES, N_FOLDS, OUT, SEEDS
from data import GROUPS, LOADERS
from metrics import debiased_ce, ece

KS = {"k1": 1, "k10": 10, "k50": 50}  # k = 10 is the main one
N_CLUSTERS = 8
GAUSS = CORRUPTIONS.index("gaussian_noise")
NOISE = [CORRUPTIONS.index(c) for c in FAMILIES["noise"]]
FAMILY = {c: fam for fam, cs in FAMILIES.items() for c in cs}
EXTRA = ["ts_gauss", "grsats_excess", "grsats_k1", "grsats_k50", "grsats_bal", "dac_nll", "dac_wg",
         "cluster_ts", "md_ts", "group_ts", "group_grsats"]
RELIABILITY = ["none", "ts_clean", "ts_corr", "dac", "aar", "grsats"]


def knn_distances(bank, queries, ks):
    # same distance as calibrate.knn_distance, for several k at once. returns (n, len(ks))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = torch.nn.functional.normalize(torch.tensor(bank, dtype=torch.float32, device=device), dim=1)
    cols = [k - 1 for k in ks]
    out = []
    for i in range(0, len(queries), 4096):
        q = torch.tensor(queries[i:i + 4096], dtype=torch.float32, device=device)
        cos = torch.nn.functional.normalize(q, dim=1) @ bank.T
        kth = cos.topk(max(ks), dim=1).values[:, cols]
        out.append(torch.sqrt((2 - 2 * kth).clamp(min=1e-12)).cpu())
    return torch.cat(out).numpy()


def balanced_bank(df, train, ds):
    # the same number of training images from every group cell (sex x age on ham, skin type on fitz)
    cell = df[GROUPS[ds]].astype(str).agg("_".join, axis=1).to_numpy()[train]
    n = min((cell == c).sum() for c in set(cell))
    rng = np.random.default_rng(0)
    return np.sort(np.concatenate([rng.choice(train[cell == c], n, replace=False) for c in sorted(set(cell))]))


def load_all(ds, df, pool, bb, kind, seed, scores=True):
    # like calibrate.load_outputs, but it also keeps the features (cluster ts and md-ts use them) and
    # gets distances for every k and for the balanced bank. d0[v] is (n,) and dc[v] is (15, 5, n)
    train = np.flatnonzero(df["split"] == "train")
    train_y = df["y"].to_numpy()[train]
    bal = balanced_bank(df, train, ds)
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

    n, dim = len(pool), X.shape[1]
    D = {"z0": z0.astype(float), "zc": np.zeros((15, 5, n)), "X0": X[pool],
         "Xc": np.zeros((15, 5, n, dim), np.float16)}
    if scores:
        D["dc"] = {v: np.zeros((15, 5, n)) for v in [*KS, "bal"]}
        D["ac"] = np.zeros((15, 5, n))
    for j, c in enumerate(CORRUPTIONS):
        f = np.load(OUT / f"{ds}_c_{c}_{kind}.npz")
        assert (f["ids"] == df["id"].to_numpy(dtype=str)[pool]).all()
        Xc = f[key + "_X"].astype(np.float32)
        flat = Xc.reshape(-1, dim)
        D["Xc"][j] = Xc
        D["zc"][j] = probe.decision_function(flat).reshape(5, n) if kind == "frozen" else f[key + "_logits"]
        if scores:
            dk = knn_distances(X[train], flat, list(KS.values()))
            for i, v in enumerate(KS):
                D["dc"][v][j] = dk[:, i].reshape(5, n)
            D["dc"]["bal"][j] = knn_distances(X[bal], flat, [10])[:, 0].reshape(5, n)
            D["ac"][j] = atypicality(X[train], train_y, flat).reshape(5, n)
    if scores:
        dk = knn_distances(X[train], X[pool], list(KS.values()))
        D["d0"] = {v: dk[:, i] for i, v in enumerate(KS)}
        D["d0"]["bal"] = knn_distances(X[bal], X[pool], [10])[:, 0]
        D["a0"] = atypicality(X[train], train_y, X[pool])
    return D


def scores(D, v="k10"):
    return D["d0"][v], D["dc"][v], D["a0"], D["ac"]


def wrap(predict, v="k10"):
    # calibrate.py's predict functions take (logits, distances, atypicality)
    return lambda T: predict(T["z"], T["d"][v], T["a"])


def dac_losses(p, z, s, y, W):
    # group nll of dac's sigmoid(z / T) with T = p0 * s + p1, and its gradient in p
    T = p[0] * s + p[1]
    u = z / T
    dT = W * ((expit(u) - y) * (-z / T ** 2))
    return W @ (np.logaddexp(0, u) - y * u), np.stack([dT @ s, dT.sum(1)], 1)


def fit_dac_nll(z, d, y, W, starts):
    # dac's temperature (same s, bounds and T >= 1e-3 as calibrate.fit_dac) fit for nll, minimizing the
    # largest group loss in epigraph form like gr-sats. with one weight row it's dac fit for average
    # nll. it isn't convex in p, so it tries every start and keeps the lowest worst-group loss
    s = d ** 2
    smin = s.min()
    best, best_val = None, np.inf
    for p0 in starts:
        p0 = np.asarray(p0, float)
        x0 = np.append(p0, dac_losses(p0, z, s, y, W)[0].max())
        cons = [{"type": "ineq", "fun": lambda x: x[2] - dac_losses(x[:2], z, s, y, W)[0],
                 "jac": lambda x: np.hstack([-dac_losses(x[:2], z, s, y, W)[1], np.ones((len(W), 1))])},
                {"type": "ineq", "fun": lambda x: np.array([x[0] * smin + x[1] - 1e-3]),
                 "jac": lambda x: np.array([[smin, 1.0, 0.0]])}]
        res = minimize(lambda x: x[2], x0, jac=lambda x: np.array([0.0, 0.0, 1.0]), method="SLSQP",
                       bounds=[(0, 1e4), (-100, 100), (None, None)], constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-10})
        for p in [p0, res.x[:2]]:
            ok = p[0] >= -1e-9 and p[0] * smin + p[1] >= 1e-3 - 1e-9
            val = dac_losses(p, z, s, y, W)[0].max()
            if ok and val < best_val:
                best, best_val = p, val

    def predict(T):
        temp = np.maximum(best[0] * T["d"]["k10"] ** 2 + best[1], 1e-3)
        return expit(T["z"] / temp)
    return predict, best


def unit(X):
    X = X.astype(np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def nearest(centers, X):
    # closest center for each row, in chunks to save memory. rows are unit length, so the closest
    # center is the one with the smallest |c|^2 - 2 x.c
    X = X.reshape(-1, X.shape[-1])
    out = [((centers ** 2).sum(1) - 2 * unit(X[i:i + 50000]) @ centers.T).argmin(1)
           for i in range(0, len(X), 50000)]
    return np.concatenate(out)


def fit_cluster_ts(z, X, y, w):
    # cluster-wise ts in the style of faircal (salvador et al. 2022): k-means on the l2-normalized
    # features of the fitting rows, then one temperature per cluster fit for the average like ts_corr.
    # k-means sees 20,000 random fitting rows to keep it fast. a cluster with under 50 rows, one
    # class, or a fit that doesn't converge keeps the global temperature
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(20000, len(X)), replace=False)
    km = KMeans(N_CLUSTERS, n_init=3, random_state=0).fit(unit(X[sub]))
    centers = km.cluster_centers_.astype(np.float32)
    lab = nearest(centers, X)
    C = positivity("global", 0, 0)
    betas = np.full(N_CLUSTERS, fit(z[:, None], y, w[None], C)[0])
    for c in range(N_CLUSTERS):
        m = lab == c
        if m.sum() >= 50 and 0 < y[m].sum() < m.sum():
            try:
                betas[c] = fit(z[m, None], y[m], (w[m] / w[m].sum())[None], C)[0]
            except AssertionError:
                pass

    def predict(T):
        return expit(betas[nearest(centers, T["X"]).reshape(T["z"].shape)] * T["z"])
    return predict, betas


def fit_md_ts(z, X, y, sev, corr):
    # multi-domain ts (yu et al. 2022, alg. 1): ts on each domain (clean, and each training corruption
    # at each severity), then least squares from the features to the domain's temperature. the
    # predicted temperature is clipped to the range of the domain temperatures so a new corruption
    # can't make it negative
    dom = (corr + 1) * 6 + sev
    C = positivity("global", 0, 0)
    temp = np.zeros(len(z))
    for k in np.unique(dom):
        m = dom == k
        temp[m] = 1 / fit(z[m, None], y[m], np.full((1, m.sum()), 1 / m.sum()), C)[0]
    # normal equations in chunks, all the fitting rows as float64 would need too much memory
    G, b = 0, 0
    for i in range(0, len(X), 20000):
        A = np.hstack([X[i:i + 20000].astype(float), np.ones((len(X[i:i + 20000]), 1))])
        G = G + A.T @ A
        b = b + A.T @ temp[i:i + 20000]
    theta = np.linalg.lstsq(G, b, rcond=None)[0]
    lo, hi = temp.min(), temp.max()

    def predict(T):
        Xt = T["X"].reshape(-1, T["X"].shape[-1]).astype(float)
        t = np.clip(Xt @ theta[:-1] + theta[-1], lo, hi)
        return expit(T["z"] / t.reshape(T["z"].shape))
    return predict, theta


def fit_oracle(base, D, y, masks, cells, imgs, corrs):
    # oracle: knows the test image's group cell (sex x age on ham, skin type on fitz) and fits one
    # calibrator per cell for the average on that cell's fitting images. ts_corr per cell is
    # group-wise ts, sats_avg per cell is per-group gr-sats (one group, so worst-group = average)
    preds = {c: fit_method(base, D["z0"], D["zc"], scores(D), y, masks, imgs[cells[imgs] == c], corrs)[0]
             for c in np.unique(cells)}

    def predict(T):
        out = np.zeros(T["z"].shape)
        for c, p in preds.items():
            m = T["cell"] == c
            out[m] = p(T["z"][m], T["d"]["k10"][m], T["a"][m])
        return out
    return predict, None


def fit_rows_features(D, imgs, corrs):
    # features and corruption index (-1 = clean) of the fitting rows, same order as stack_rows
    X = np.concatenate([D["X0"][imgs].astype(np.float16)] + [D["Xc"][j, l, imgs] for j in corrs for l in range(5)])
    corr = np.concatenate([np.full(len(imgs), -1)] + [np.full(len(imgs), j) for j in corrs for l in range(5)])
    return X, corr


def fit_all(D, y, masks, cells, imgs, corrs):
    # every calibrator, fit on the same rows. each one comes back as a function of a test input dict
    sc = scores(D)
    out = {name: wrap(fit_method(name, D["z0"], D["zc"], sc, y, masks, imgs, corrs)[0]) for name in METHODS}
    # tomani et al. 2021 fit ts on images with gaussian noise. this uses medmnist-c's gaussian noise at
    # all 5 severities, so it has seen the noise family when that's the held-out one (see wg_nll no noise)
    out["ts_gauss"] = wrap(fit_method("ts_corr", D["z0"], D["zc"], sc, y, masks, imgs, [GAUSS])[0])
    out["grsats_excess"] = wrap(fit_method("grsats_excess", D["z0"], D["zc"], sc, y, masks, imgs, corrs,
                                           {"grsats_excess": ("shift", "excess")})[0])
    for v in ["k1", "k50", "bal"]:
        out[f"grsats_{v}"] = wrap(fit_method("grsats", D["z0"], D["zc"], scores(D, v), y, masks, imgs, corrs)[0], v)

    z, d, a, sev, img = stack_rows(D["z0"], D["zc"], sc, imgs, corrs)
    W_avg, W_wg = weights("avg", sev, []), weights("wg", sev, [g[img] for g in masks])
    out["dac_nll"], p_avg = fit_dac_nll(z, d, y[img], W_avg, [[1.0, 1.0]])
    out["dac_wg"], _ = fit_dac_nll(z, d, y[img], W_wg, [[1.0, 1.0], p_avg])
    X, corr = fit_rows_features(D, imgs, corrs)
    out["cluster_ts"], _ = fit_cluster_ts(z, X, y[img], W_avg[0])
    out["md_ts"], _ = fit_md_ts(z, X, y[img], sev, corr)
    out["group_ts"], _ = fit_oracle("ts_corr", D, y, masks, cells, imgs, corrs)
    out["group_grsats"], _ = fit_oracle("sats_avg", D, y, masks, cells, imgs, corrs)
    return out


def test_inputs(D, cells, te):
    # what the calibrators see for the test fold: the clean images, then each corruption (5 severities)
    clean = {"z": D["z0"][te], "d": {v: d[te] for v, d in D["d0"].items()}, "a": D["a0"][te],
             "X": D["X0"][te], "cell": cells[te]}
    corr = [{"z": D["zc"][j][:, te], "d": {v: d[j][:, te] for v, d in D["dc"].items()},
             "a": D["ac"][j][:, te], "X": D["Xc"][j][:, te], "cell": np.tile(cells[te], (5, 1))}
            for j in range(15)]
    return clean, corr


def crossfit_all(D, y, masks, cells, folds):
    # same as calibrate.crossfit, for every method
    fams = list(FAMILIES)
    names = ["none", *METHODS, *EXTRA]
    q_clean = {k: np.zeros((len(fams), len(y))) for k in names}
    q_corr = {k: np.zeros(D["zc"].shape) for k in names}
    q_clean["none"][:] = expit(D["z0"])
    q_corr["none"] = expit(D["zc"])
    for f in range(N_FOLDS):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        clean, corr = test_inputs(D, cells, te)
        for h, fam in enumerate(fams):
            held = [j for j, c in enumerate(CORRUPTIONS) if c in FAMILIES[fam]]
            rest = [j for j in range(15) if j not in held]
            for name, predict in fit_all(D, y, masks, cells, tr, rest).items():
                q_clean[name][h, te] = predict(clean)
                for j in held:
                    q_corr[name][j][:, te] = predict(corr[j])
        print(f"  fold {f} done")
    return q_clean, q_corr


def evaluate_more(q_clean, q_corr, y, masks, names, attrs, eval_c):
    # columns calibrate.evaluate doesn't have. corr = held-out corruptions at severities 3 to 5
    row = {}
    keep = [j for j in eval_c if j not in NOISE]
    row["wg_nll no noise"] = max(nll(y, q_corr[keep][:, 2:5])[..., m].mean() for m in masks)
    flat = q_corr[eval_c][:, 2:5].reshape(-1, len(y))
    row["clean_dce"] = np.mean([debiased_ce(y, q) for q in q_clean])
    row["corr_ece"] = ece(np.tile(y, len(flat)), flat.ravel())
    row["corr_dce"] = debiased_ce(np.tile(y, len(flat)), flat.ravel())
    for g, m in zip(names, masks):
        row[f"clean ece {g}"] = np.mean([ece(y[m], q[m]) for q in q_clean])
        row[f"corr ece {g}"] = ece(np.tile(y[m], len(flat)), flat[:, m].ravel())
    for col, gs in attrs.items():
        for tag in ["clean", "corr"]:
            e = [row[f"{tag} ece {g}"] for g in gs]
            row[f"{tag} ece gap {col}"] = max(e) - min(e)
    return row


def detail_rows(q_clean, q_corr, y, masks, names):
    # nll and brier per group on clean images and on every corruption at every severity
    rows = []
    L0, S0 = nll(y, q_clean), (q_clean - y) ** 2
    L, S = nll(y, q_corr), (q_corr - y) ** 2
    for g, m in zip(names + ["all"], masks + [np.ones(len(y), bool)]):
        rows.append({"corruption": "clean", "family": "clean", "severity": 0, "group": g,
                     "nll": L0[:, m].mean(), "brier": S0[:, m].mean()})
        for j, c in enumerate(CORRUPTIONS):
            for l in range(5):
                rows.append({"corruption": c, "family": FAMILY[c], "severity": l + 1, "group": g,
                             "nll": L[j, l, m].mean(), "brier": S[j, l, m].mean()})
    return rows


def reliability_rows(q_clean, q_corr, y, masks, names, eval_c, bins=15):
    # reliability diagram points per group: 15 equal-mass bins of the predicted risk. clean pools the
    # 5 clean predictions of each image (one per held-out family), so n counts each image 5 times
    rows = []
    for tag, q in [("clean", q_clean), ("sev3-5", q_corr[eval_c][:, 2:5])]:
        flat = q.reshape(-1, len(y))
        for g, m in zip(names + ["all"], masks + [np.ones(len(y), bool)]):
            p, yy = flat[:, m].ravel(), np.tile(y[m], len(flat))
            for b, idx in enumerate(np.array_split(np.argsort(p), bins)):
                rows.append({"condition": tag, "group": g, "bin": b, "mean risk": p[idx].mean(),
                             "frac malignant": yy[idx].mean(), "n": len(idx)})
    return rows


def score_rows(D, masks, names, eval_c):
    # mean standardized shift score per group, clean and at severities 3 to 5, for each k and bank.
    # a group that scores high on clean too is high because it's rare in training, not from corruption
    rows = []
    for v in D["d0"]:
        ld0, ldc = np.log(D["d0"][v]), np.log(D["dc"][v][eval_c][:, 2:5])
        m, r = ld0.mean(), ld0.std()
        for g, mask in zip(names, masks):
            rows.append({"bank": v, "group": g, "clean": ((ld0 - m) / r)[mask].mean(),
                         "sev3-5": ((ldc - m) / r)[..., mask].mean()})
    return rows


def check_old(res, ds, kind):
    # calibrate.py's methods should come out the same as the committed results
    old = Path(__file__).resolve().parent / "results" / f"{ds}_{kind}_calibration.csv"
    if not old.exists():
        print("\nno committed results to check against")
        return
    both = pd.read_csv(old).merge(res, on=["model", "seed", "method"], suffixes=("_old", ""))
    diff = (both["wg_nll_old"] - both["wg_nll"]).abs().max()
    print(f"\nlargest wg_nll difference from results/{old.name}: {diff:.1e} ({len(both)} rows matched)")


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
    attrs = {col: sorted(dp[col].unique()) for col in GROUPS[ds]}
    cells = np.unique(dp[GROUPS[ds]].astype(str).agg("_".join, axis=1), return_inverse=True)[1]
    folds = make_folds(dp, ds)
    eval_c = [j for j, c in enumerate(CORRUPTIONS) if ds == "ham" or c not in FAMILIES["task-specific"]]

    results, details, rel, score_tab = [], [], [], []
    per_image = {"y": y, "lesion": dp["lesion"].to_numpy(dtype=str),
                 **{f"group {g}": m for g, m in zip(names, masks)}}
    for bb in ["dinov2", "biomedclip"]:
        for seed in SEEDS:
            start = time.time()
            # frozen features are the same for every probe seed, so the scores are too
            new_scores = kind == "lora" or seed == SEEDS[0]
            D = load_all(ds, df, pool, bb, kind, seed, scores=new_scores)
            if new_scores:
                saved = {k: D[k] for k in ["d0", "dc", "a0", "ac"]}
            D.update(saved)
            model = f"{bb} {'lp' if kind == 'frozen' else 'lora'}"
            print(f"\n{model} seed {seed}")

            q_clean, q_corr = crossfit_all(D, y, masks, cells, folds)
            info = {"model": model, "seed": seed}
            for name in q_clean:
                results.append({**info, "method": name,
                                **evaluate(q_clean[name], q_corr[name], y, masks, names, eval_c),
                                **evaluate_more(q_clean[name], q_corr[name], y, masks, names, attrs, eval_c)})
                details += [{**info, "method": name, **r} for r in detail_rows(q_clean[name], q_corr[name], y, masks, names)]
                # per-image losses on the main pool, for a bootstrap that pools all 8 settings later
                qc = q_corr[name][eval_c][:, 2:5]
                per_image[f"{model}|{seed}|{name}|nll"] = nll(y, qc).reshape(-1, len(y)).mean(0).astype(np.float32)
                per_image[f"{model}|{seed}|{name}|brier"] = ((qc - y) ** 2).reshape(-1, len(y)).mean(0).astype(np.float32)
            for name in RELIABILITY:
                rel += [{**info, "method": name, **r}
                        for r in reliability_rows(q_clean[name], q_corr[name], y, masks, names, eval_c)]
            score_tab += [{**info, **r} for r in score_rows(D, masks, names, eval_c)]
            print(f"  took {(time.time() - start) / 60:.1f} min")

    res = pd.DataFrame(results)
    res.to_csv(OUT / f"{ds}_{kind}_extra.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / f"{ds}_{kind}_detail.csv", index=False)
    pd.DataFrame(rel).to_csv(OUT / f"{ds}_{kind}_reliability.csv", index=False)
    pd.DataFrame(score_tab).to_csv(OUT / f"{ds}_{kind}_scores.csv", index=False)
    np.savez_compressed(OUT / f"{ds}_{kind}_perimage.npz", **per_image)
    check_old(res, ds, kind)

    pd.set_option("display.width", 200)
    cols = ["wg_nll", "wg_nll no noise", "wg_brier", "nll", "clean_ece", "clean_dce", "clean_auc", "corr_acc"]
    print(f"\n{ds} {kind}: held-out families, severities 3 to 5, mean over seeds")
    print(res.groupby(["model", "method"], sort=False)[cols].mean().round(4).to_string())
    print(f"\n{ds} {kind}: mean shift score per group (clean, sev 3-5), mean over seeds")
    print(pd.DataFrame(score_tab).groupby(["model", "bank", "group"], sort=False)[["clean", "sev3-5"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
