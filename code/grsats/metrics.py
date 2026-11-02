import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

# TODO: ECE and brier once the calibrators are in


def group_metrics(df, group_cols):
    """AUC and NLL for all images and for each group. df needs columns y and p (prob of malignant)."""
    parts = [("all", df)] + [(g, d) for col in group_cols for g, d in df.groupby(col)]
    rows = []
    for name, d in parts:
        rows.append({
            "group": name,
            "n": len(d),
            "auc": roc_auc_score(d["y"], d["p"]),
            "nll": log_loss(d["y"], d["p"], labels=[0, 1]),
        })
    return pd.DataFrame(rows)
