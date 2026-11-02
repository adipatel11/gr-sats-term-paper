from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Just the images. Labels and groups stay in the dataframe, same order."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))
