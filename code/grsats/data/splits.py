import numpy as np

from grsats.config import DEV_FRAC


def add_dev(df, seed):
    """Relabel ~10% of the train lesions as 'dev'. All images of a lesion move together."""
    df = df.copy()
    lesions = df.loc[df["split"] == "train", "lesion"].unique()
    rng = np.random.default_rng(seed)
    dev = rng.choice(lesions, size=int(DEV_FRAC * len(lesions)), replace=False)
    df.loc[(df["split"] == "train") & df["lesion"].isin(dev), "split"] = "dev"
    return df
