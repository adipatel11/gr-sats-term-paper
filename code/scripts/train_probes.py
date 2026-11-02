# trains the linear probes on saved features, one per seed (each seed gets its own dev slice).
# run from code/: python -m scripts.train_probes --dataset ham --backbone dinov2
import argparse

import numpy as np
import pandas as pd

from grsats.config import OUT, SEEDS
from grsats.data import GROUPS, LOADERS
from grsats.data.splits import add_dev
from grsats.metrics import group_metrics
from grsats.probes import train_probe

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
parser.add_argument("--backbone", choices=["dinov2", "biomedclip"], required=True)
args = parser.parse_args()
name = f"{args.dataset}_{args.backbone}"

df = LOADERS[args.dataset]()
f = np.load(OUT / f"{name}_features.npz")
assert (f["ids"] == df["id"].to_numpy(dtype=str)).all(), "features and csv rows don't line up"
X, y = f["X"], df["y"].to_numpy()

results = []
for seed in SEEDS:
    d = add_dev(df, seed)
    tr = (d["split"] == "train").to_numpy()
    dev = (d["split"] == "dev").to_numpy()
    lam, dev_nll, probe = train_probe(X[tr], y[tr], X[dev], y[dev])

    # save logits for every image, the calibrators use these later
    logits = probe.decision_function(X)
    np.savez(OUT / f"{name}_lp_seed{seed}.npz", ids=f["ids"], split=d["split"].to_numpy(dtype=str), logits=logits)

    # clean results on the pool (val + test)
    d["p"] = probe.predict_proba(X)[:, 1]
    m = group_metrics(d[d["split"].isin(["val", "test"])], GROUPS[args.dataset])
    print(f"\nseed {seed}: l2 = {lam}, dev nll = {dev_nll:.3f}")
    print(m.round(3).to_string(index=False))
    results.append(m.assign(seed=seed, l2=lam))

res = pd.concat(results)
res.to_csv(OUT / f"{name}_lp_results.csv", index=False)
print(f"\n{name} mean over seeds:")
print(res.groupby("group", sort=False)[["auc", "nll"]].agg(["mean", "std"]).round(3))
