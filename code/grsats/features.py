import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grsats.data.image_dataset import ImageDataset


@torch.no_grad()
def extract(encode, transform, paths, device, batch_size):
    # colab only has 2 cpu cores, so 2 workers
    loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size, num_workers=2)
    feats = []
    for x in tqdm(loader):
        with torch.autocast(device, enabled=(device == "cuda")):  # fp16 on the gpu
            feats.append(encode(x.to(device)).float().cpu())
    return torch.cat(feats).numpy()
