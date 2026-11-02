# runs a frozen backbone over every image of one dataset and saves the features.
# run from code/: python -m scripts.extract_features --dataset ham --backbone dinov2
import argparse

import numpy as np
import torch

from grsats.config import BATCH_SIZE, OUT
from grsats.data import LOADERS
from grsats.features import extract
from grsats.models.backbones import load_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
    parser.add_argument("--backbone", choices=["dinov2", "biomedclip"], required=True)
    args = parser.parse_args()

    out = OUT / f"{args.dataset}_{args.backbone}_features.npz"
    if out.exists():  # so rerunning after a colab disconnect skips finished ones
        print(f"{out.name} already done, skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = LOADERS[args.dataset]()
    encode, transform = load_backbone(args.backbone, device)
    X = extract(encode, transform, df["path"], device, BATCH_SIZE)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, ids=df["id"].to_numpy(dtype=str))
    print(f"saved {X.shape} to {out}")


# the guard matters: dataloader workers re-import this file on mac (and newer python on linux)
if __name__ == "__main__":
    main()
