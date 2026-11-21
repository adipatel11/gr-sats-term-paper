# saves frozen backbone features for every image in a dataset
# python extract_features.py --dataset ham --backbone dinov2
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, OUT
from data import LOADERS, ImageDataset, missing_images
from models import get_transform, load_backbone


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
    parser.add_argument("--backbone", choices=["dinov2", "biomedclip"], required=True)
    args = parser.parse_args()

    out = OUT / f"{args.dataset}_{args.backbone}_features.npz"
    if out.exists():
        print(f"{out.name} already done")
        return

    df = LOADERS[args.dataset]()
    assert missing_images(df) == 0, "images not found, check the unzip step"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(args.backbone, device)
    dataset = ImageDataset(df["path"], get_transform(args.backbone))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    feats = []
    for x in tqdm(loader):
        with torch.autocast(device, enabled=(device == "cuda")):
            feats.append(model(x.to(device)).float().cpu())
    X = torch.cat(feats).numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, ids=df["id"].to_numpy(dtype=str))
    print(f"saved {X.shape} to {out}")


if __name__ == "__main__":
    main()
