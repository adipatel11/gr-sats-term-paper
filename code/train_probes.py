# trains a logistic regression probe on the saved features, one per seed
# python train_probes.py --dataset ham --backbone dinov2
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import L2_GRID, OUT, SEEDS
from data import GROUPS, LOADERS, add_dev
from metrics import group_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
parser.add_argument("--backbone", choices=["dinov2", "biomedclip"], required=True)
args = parser.parse_args()
name = f"{args.dataset}_{args.backbone}"

df = LOADERS[args.dataset]()
f = np.load(OUT / f"{name}_features.npz")
assert (f["ids"] == df["id"].to_numpy(dtype=str)).all()
X, y = f["X"], df["y"].to_numpy()

results = []
for seed in SEEDS:
    d = add_dev(df, seed)
    tr = (d["split"] == "train").to_numpy()
    dev = (d["split"] == "dev").to_numpy()

    # pick the l2 strength with the lowest dev nll
    best_nll = float("inf")
    for l2 in L2_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1 / l2, max_iter=5000))
        model.fit(X[tr], y[tr])
        nll = log_loss(y[dev], model.predict_proba(X[dev])[:, 1])
        if nll < best_nll:
            best_nll, best_l2, probe = nll, l2, model

    # logits for every image, the calibrators use these later
    logits = probe.decision_function(X)
    np.savez(OUT / f"{name}_lp_seed{seed}.npz", ids=f["ids"], split=d["split"].to_numpy(dtype=str), logits=logits)
    # the probe itself too, the corrupted features need it
    joblib.dump(probe, OUT / f"{name}_lp_seed{seed}.joblib")

    # clean results on the pool (val + test)
    d["p"] = probe.predict_proba(X)[:, 1]
    m = group_metrics(d[d["split"].isin(["val", "test"])], GROUPS[args.dataset])
    print(f"\nseed {seed}: l2 = {best_l2}, dev nll = {best_nll:.3f}")
    print(m.round(3).to_string(index=False))
    results.append(m.assign(seed=seed, l2=best_l2))

res = pd.concat(results)
res.to_csv(OUT / f"{name}_lp_results.csv", index=False)
print(f"\n{name} mean over seeds:")
print(res.groupby("group", sort=False)[["auc", "nll"]].agg(["mean", "std"]).round(3))
