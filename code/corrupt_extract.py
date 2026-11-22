# runs the 15 medmnist-c derm corruptions at 5 severities over the pool images (val + test)
# and saves what every model outputs. one file per corruption, so if colab disconnects
# just run it again and the finished corruptions get skipped
# python corrupt_extract.py --dataset ham --models frozen
# python corrupt_extract.py --dataset ham --models lora
# python corrupt_extract.py --dataset ham --models frozen --test   (64 images, saves nothing)
import argparse
import random
import time
from io import BytesIO

import cv2
import medmnistc.corruptions.filter as mc_filter
import numpy as np
import torch
from medmnistc.corruptions.registry import CORRUPTIONS_DS
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from config import BATCH_SIZE, OUT, SEEDS
from data import LOADERS, load_images, missing_images
from models import MEAN_STD, LoRAModel, load_backbone



class MotionBlur(mc_filter.MotionBlur):
    # same as medmnist-c's motion blur, but it calls np.fromstring, which newer numpy removed
    def apply(self, img, severity):
        radius, sigma = self.severity_params[severity]
        png = BytesIO()
        img.save(png, format="PNG")
        img = mc_filter.MotionImage(blob=png.getvalue())
        img.motion_blur(radius=radius, sigma=sigma, angle=np.random.uniform(-45, 45))
        img = cv2.imdecode(np.frombuffer(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        return img[..., [2, 1, 0]]  # bgr to rgb


# medmnist-c only has settings for dermamnist, so fitz uses the same ones
CORRUPTORS = dict(CORRUPTIONS_DS["dermamnist"])
CORRUPTORS["motion_blur"] = MotionBlur(CORRUPTORS["motion_blur"].severity_params)
CORRUPTIONS = list(CORRUPTORS)


class CorruptedImages(Dataset):
    def __init__(self, images, corruption, severity):
        self.images, self.severity = images, severity
        self.corruptor = CORRUPTORS[corruption]
        self.offset = (CORRUPTIONS.index(corruption) * 5 + severity) * 100_000

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # seed every image on its own so each run and each worker makes the exact same image
        seed = self.offset + i
        random.seed(seed)
        np.random.seed(seed)
        self.corruptor.rng = np.random.default_rng(seed)  # impulse noise reads this one
        img = Image.fromarray(self.images[i].permute(1, 2, 0).numpy())
        out = np.ascontiguousarray(self.corruptor.apply(img, self.severity))
        return torch.from_numpy(out).permute(2, 0, 1)


def load_models(dataset, kind, device):
    models = {}
    for bb in ["dinov2", "biomedclip"]:
        if kind == "frozen":
            models[f"{bb}_frozen"] = (bb, load_backbone(bb, device))
        else:
            for s in SEEDS:
                net = LoRAModel(bb, load_backbone(bb, device))
                weights = torch.load(OUT / f"{dataset}_{bb}_lora_seed{s}.pt", map_location=device)
                net.load_state_dict(weights, strict=False)
                models[f"{bb}_lora{s}"] = (bb, net.to(device).eval())
    return models


@torch.no_grad()
def run(models, x, device):
    out = {}
    x = x.to(device).float() / 255
    for key, (bb, net) in models.items():
        mean, std = MEAN_STD[bb]
        with torch.autocast(device, enabled=(device == "cuda")):
            res = net(T.Normalize(mean, std)(x))
        if "lora" in key:
            out[key + "_logits"] = res[0].float().cpu()
            res = res[1]
        out[key + "_X"] = res.float().cpu()
    return out


def check_clean(models, images, pool, dataset, device):
    # clean images through this same path should give back the saved clean outputs
    out = run(models, images[:BATCH_SIZE], device)
    rows = pool[:BATCH_SIZE]
    for key, got in out.items():
        bb, model, what = key.split("_")
        if model == "frozen":
            saved = np.load(OUT / f"{dataset}_{bb}_features.npz")["X"][rows]
        else:
            saved = np.load(OUT / f"{dataset}_{bb}_lora_seed{model[-1]}.npz")[what][rows]
        diff = np.abs(got.numpy() - saved).max() / np.abs(saved).max()
        print(f"{key}: max diff from saved clean outputs = {diff:.4f}")
        assert diff < 0.02, "clean images don't match the saved outputs, preprocessing changed?"


def test_run(images):
    # time each corruption at the worst severity and save a picture of them to look at
    tiles = [images[0].permute(1, 2, 0).numpy()]
    for corruption in CORRUPTIONS:
        data = CorruptedImages(images[:BATCH_SIZE], corruption, 4)
        start = time.time()
        batch = torch.stack([data[i] for i in range(len(data))])
        ms = 1000 * (time.time() - start) / len(data)
        change = (batch.float() - images[:BATCH_SIZE].float()).abs().mean()
        assert batch.shape == images[:BATCH_SIZE].shape and batch.dtype == torch.uint8
        assert torch.equal(batch[0], data[0]), "same seed should give the same image"
        print(f"{corruption:>16}: {ms:5.1f} ms per image, mean pixel change {change:5.1f}")
        tiles.append(batch[0].permute(1, 2, 0).numpy())
    grid = np.concatenate([np.concatenate(tiles[r * 4:r * 4 + 4], axis=1) for r in range(4)], axis=0)
    Image.fromarray(grid).save(OUT / "corruption_preview.png")
    print(f"saved {OUT / 'corruption_preview.png'} (clean first, then the 15 corruptions in order)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ham", "fitz"], required=True)
    parser.add_argument("--models", choices=["frozen", "lora"], required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    df = LOADERS[args.dataset]()
    assert missing_images(df) == 0, "images not found, check the unzip step"
    pool = np.flatnonzero(df["split"].isin(["val", "test"]))
    if args.test:
        pool = pool[:BATCH_SIZE]
    images = load_images(df["path"].iloc[pool])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(args.dataset, args.models, device)
    check_clean(models, images, pool, args.dataset, device)
    if args.test:
        test_run(images)
        return

    for corruption in CORRUPTIONS:
        path = OUT / f"{args.dataset}_c_{corruption}_{args.models}.npz"
        if path.exists():
            print(f"{path.name} already done")
            continue

        res = {}
        for severity in range(5):
            loader = DataLoader(CorruptedImages(images, corruption, severity), batch_size=BATCH_SIZE, num_workers=2)
            parts = [run(models, x, device) for x in tqdm(loader, desc=f"{corruption} severity {severity + 1}")]
            for key in parts[0]:
                res.setdefault(key, []).append(torch.cat([p[key] for p in parts]).numpy())

        # shape (5 severities, n images, ...). lora features are only for the knn score so
        # float16 is enough there, frozen features stay float32 because the probes read them
        arrays = {}
        for key, v in res.items():
            small = key.endswith("_X") and "lora" in key
            arrays[key] = np.stack(v).astype(np.float16 if small else np.float32)

        # save under a temp name first so a disconnect can't leave half a file behind
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=df["id"].to_numpy(dtype=str)[pool], **arrays)
        tmp.rename(path)
        print(f"saved {path.name}")


if __name__ == "__main__":
    main()
