import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

# TODO: ECE and brier


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
        })
    return pd.DataFrame(rows)
