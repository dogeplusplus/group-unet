from einops import reduce
from typing import List
from pathlib import Path
from imageio import imread
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, images: List[Path], transform=None):
        self.images = images
        self.labels = [
            str(x).replace("images", "segmentations").replace(
                ".png", "_seg0.png")
            for x in self.images
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = imread(str(self.images[idx]))
        segmentation = imread(str(self.labels[idx])) / 255
        segmentation = reduce(segmentation, "h w c -> h w", "max")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=segmentation)
            image = transformed["image"]
            segmentation = transformed["mask"]

        return image, segmentation
