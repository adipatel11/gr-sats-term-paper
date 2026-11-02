import pandas as pd

from grsats.config import DATA

SKIN = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}


def load_fitz():
    df = pd.read_csv(DATA / "Fitzpatrick17k-C.csv")
    # drop the 296 images with no skin type (fitzpatrick = -1), leaves 11,098
    df = df[df["fitzpatrick"] > 0].reset_index(drop=True)

    return pd.DataFrame({
        "id": df["md5hash"],
        "lesion": df["md5hash"],  # no lesion ids here, so each image is its own lesion
        "path": [str(DATA / "finalfitz17k" / f"{h}.jpg") for h in df["md5hash"]],
        "y": (df["three_partition_label"] == "malignant").astype(int),
        "split": df["partition"],
        "skin": df["fitzpatrick"].map(SKIN),
    })
