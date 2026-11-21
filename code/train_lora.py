# trains the lora models, one per seed. if colab disconnects just run it again
# python train_lora.py --dataset ham --backbone dinov2
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from config import BATCH_SIZE, LORA_EPOCHS, LORA_LR, LORA_PATIENCE, LORA_WD, OUT, SEEDS
from data import GROUPS, LOADERS, ImageDataset, add_dev, missing_images
from metrics import group_metrics
from models import MEAN_STD, LoRAModel, load_backbone


class CachedImages(Dataset):
    def __init__(self, images, y, idx, transform):
        self.images, self.y, self.idx, self.transform = images, y, idx, transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.transform(self.images[j]), self.y[j]


def rotate90(x):
    return torch.rot90(x, int(torch.randint(4, (1,))), (1, 2))


def load_images(paths):
    # decode everything once, colab's 2 cpus are too slow to do it every epoch
    crop = T.Compose([T.Resize(224), T.CenterCrop(224), T.PILToTensor()])
    loader = DataLoader(ImageDataset(paths, crop), batch_size=BATCH_SIZE, num_workers=2)
    return torch.cat(list(tqdm(loader, desc="loading images")))


@torch.no_grad()
def predict(net, images, idx, normalize, device):
    logits, feats = [], []
    for i in range(0, len(idx), BATCH_SIZE):
        x = normalize(images[idx[i:i + BATCH_SIZE]].to(device))
        with torch.autocast(device, enabled=(device == "cuda")):
            logit, f = net(x)
        logits.append(logit.cpu())
        feats.append(f.cpu())
    return torch.cat(logits), torch.cat(feats)


def train_model(backbone, seed, name, d, images, normalize, device):
    y = d["y"].to_numpy()
    tr = np.flatnonzero(d["split"] == "train")
    dev = np.flatnonzero(d["split"] == "dev")
    y_dev = torch.tensor(y[dev]).float()

    # geometric only, color and blur would overlap with the test corruptions
    augment = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
                         T.RandomVerticalFlip(), rotate90, normalize])
    loader = DataLoader(CachedImages(images, y, tr, augment), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True)

    torch.manual_seed(seed)
    net = LoRAModel(backbone, load_backbone(backbone, device)).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=LORA_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, LORA_EPOCHS * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    start, best_nll, best, bad = 0, float("inf"), None, 0
    ckpt = OUT / f"{name}_lora_seed{seed}_ckpt.pt"
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device)
        net.load_state_dict(c["net"], strict=False)
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        start, best_nll, best, bad = c["epoch"] + 1, c["best_nll"], c["best"], c["bad"]
        print(f"resuming seed {seed} at epoch {start}")

    net.eval()  # no dropout or batchnorm in these models
    for epoch in range(start, LORA_EPOCHS):
        if bad >= LORA_PATIENCE:
            break
        for x, target in tqdm(loader, desc=f"seed {seed} epoch {epoch}"):
            with torch.autocast(device, enabled=(device == "cuda")):
                logit, _ = net(x.to(device))
            loss = F.binary_cross_entropy_with_logits(logit, target.to(device).float())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        dev_logits, _ = predict(net, images, dev, normalize, device)
        dev_nll = F.binary_cross_entropy_with_logits(dev_logits, y_dev).item()
        bad += 1
        if dev_nll < best_nll:
            best_nll, best, bad = dev_nll, net.trainable(), 0
        print(f"seed {seed} epoch {epoch}: dev nll = {dev_nll:.4f}, best = {best_nll:.4f}")

        torch.save({"net": net.trainable(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_nll": best_nll,
                    "best": best, "bad": bad}, ckpt)

    # go back to the best epoch and save its outputs for every image
    net.load_state_dict(best, strict=False)
    logits, X = predict(net, images, np.arange(len(d)), normalize, device)
    torch.save(best, OUT / f"{name}_lora_seed{seed}.pt")
    np.savez(OUT / f"{name}_lora_seed{seed}.npz", ids=d["id"].to_numpy(dtype=str),
             split=d["split"].to_numpy(dtype=str), logits=logits.numpy(), X=X.numpy(), dev_nll=best_nll)
    ckpt.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
    parser.add_argument("--backbone", choices=["dinov2", "biomedclip"], required=True)
    args = parser.parse_args()
    name = f"{args.dataset}_{args.backbone}"

    df = LOADERS[args.dataset]()
    assert missing_images(df) == 0, "images not found, check the unzip step"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    mean, std = MEAN_STD[args.backbone]
    normalize = T.Compose([T.ConvertImageDtype(torch.float), T.Normalize(mean, std)])
    todo = [s for s in SEEDS if not (OUT / f"{name}_lora_seed{s}.npz").exists()]
    images = load_images(df["path"]) if todo else None

    results = []
    for seed in SEEDS:
        d = add_dev(df, seed)
        if seed in todo:
            train_model(args.backbone, seed, name, d, images, normalize, device)

        # clean results on the pool (val + test), same as the probes
        saved = np.load(OUT / f"{name}_lora_seed{seed}.npz")
        d["p"] = 1 / (1 + np.exp(-saved["logits"].astype(float)))
        m = group_metrics(d[d["split"].isin(["val", "test"])], GROUPS[args.dataset])
        print(f"\nseed {seed}: dev nll = {saved['dev_nll']:.3f}")
        print(m.round(3).to_string(index=False))
        results.append(m.assign(seed=seed))

    res = pd.concat(results)
    res.to_csv(OUT / f"{name}_lora_results.csv", index=False)
    print(f"\n{name} lora mean over seeds:")
    print(res.groupby("group", sort=False)[["auc", "nll"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
