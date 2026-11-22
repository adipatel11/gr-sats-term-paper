import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def brier(y, p):
    return np.mean((p - y) ** 2)


def ece(y, p, bins=15):
    # equal-mass bins: sort by p and cut into bins with the same number of images
    y, p = np.asarray(y), np.asarray(p)
    gaps = [abs(y[b].mean() - p[b].mean()) * len(b) for b in np.array_split(np.argsort(p), bins)]
    return sum(gaps) / len(p)


def debiased_ce(y, p, bins=15):
    # kumar et al. 2019 (def. 5.2): squared calibration error over the same equal-mass bins, minus
    # the label variance in each bin, which is what makes the plain estimate too high with few images.
    # square root at the end so it reads on the same scale as ece
    y, p = np.asarray(y), np.asarray(p)
    total = 0
    for b in np.array_split(np.argsort(p), bins):
        yb = y[b].mean()
        total += len(b) / len(p) * ((p[b].mean() - yb) ** 2 - yb * (1 - yb) / (len(b) - 1))
    return np.sqrt(max(total, 0))


def group_metrics(df, group_cols):
    # df needs columns y and p (prob of malignant)
    groups = [("all", df)]
    for col in group_cols:
        groups += list(df.groupby(col))

    rows = []
    for name, d in groups:
        rows.append({
            "group": name,
            "n": len(d),
            "auc": roc_auc_score(d["y"], d["p"]),
            "nll": log_loss(d["y"], d["p"], labels=[0, 1]),
            "brier": brier(d["y"], d["p"]),
            "ece": ece(d["y"], d["p"]),
        })
    return pd.DataFrame(rows)
