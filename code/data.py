import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from config import BATCH_SIZE, DATA, DEV_FRAC

MALIGNANT = ["mel", "bcc", "akiec"]
SKIN = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}
GROUPS = {"ham": ["sex", "age"], "fitz": ["skin"]}


def load_ham():
    df = pd.read_csv(DATA / "DermaMNIST-C.csv")
    # drop the 67 images with no sex or age like MEDFAIR, leaves 9,948
    df = df[(df["sex"] != "unknown") & df["age"].notna()].reset_index(drop=True)

    folder = DATA / "HAM10000" / "dataverse_files"
    paths = {p.stem: str(p) for p in folder.glob("HAM10000_images_part_*/**/*.jpg")}

    return pd.DataFrame({
        "id": df["image_id"],
        "lesion": df["lesion_id"],
        "path": [paths.get(i, "") for i in df["image_id"]],
        "y": df["dx"].isin(MALIGNANT).astype(int),
        "split": df["split"],
        "sex": df["sex"],
        "age": ["60+" if a >= 60 else "<60" for a in df["age"]],
    })


def load_fitz():
    df = pd.read_csv(DATA / "Fitzpatrick17k-C.csv")
    # drop the 296 images with no skin type, leaves 11,098
    df = df[df["fitzpatrick"] > 0].reset_index(drop=True)

    return pd.DataFrame({
        "id": df["md5hash"],
        "lesion": df["md5hash"],  # no lesion ids in this dataset
        "path": [str(DATA / "finalfitz17k" / f"{h}.jpg") for h in df["md5hash"]],
        "y": (df["three_partition_label"] == "malignant").astype(int),
        "split": df["partition"],
        "skin": df["fitzpatrick"].map(SKIN),
    })


LOADERS = {"ham": load_ham, "fitz": load_fitz}


def add_dev(df, seed):
    # moves 10% of the train lesions to a dev split, all images of a lesion stay together
    df = df.copy()
    lesions = df.loc[df["split"] == "train", "lesion"].unique()
    rng = np.random.default_rng(seed)
    dev = rng.choice(lesions, size=int(DEV_FRAC * len(lesions)), replace=False)
    df.loc[(df["split"] == "train") & df["lesion"].isin(dev), "split"] = "dev"
    return df


def missing_images(df):
    return sum(not os.path.exists(p) for p in df["path"])


class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def load_images(paths):
    # decode everything once as uint8, colab's 2 cpus are too slow to do it every epoch.
    # same resize and crop as get_transform in models.py
    crop = T.Compose([T.Resize(224), T.CenterCrop(224), T.PILToTensor()])
    loader = DataLoader(ImageDataset(paths, crop), batch_size=BATCH_SIZE, num_workers=2)
    return torch.cat(list(tqdm(loader, desc="loading images")))
