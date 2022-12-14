import cv2

from einops import reduce
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    class_labels = {
        1: "butterfly",
    }
    spatial_size = (256, 256)

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
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation = cv2.imread(str(self.labels[idx])) / 255
        segmentation = reduce(segmentation, "h w c -> h w", "max")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=segmentation)
            image = transformed["image"]
            segmentation = transformed["mask"]

        return image, segmentation


class BDD100K(Dataset):
    classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle", "void"
    ]
    class_labels = {i: c for i, c in enumerate(classes)}
    spatial_size = (720, 1280)

    def __init__(self, images: List[Path], transform=None):
        self.images = images
        self.labels = [
            str(x).replace("images", "labels").replace(".jpg", "_train_id.png")
            for x in self.images
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation = cv2.imread(str(self.labels[idx])) / 255
        segmentation = reduce(segmentation, "h w c -> h w", "max")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=segmentation)
            image = transformed["image"]
            segmentation = transformed["mask"]

        return image, segmentation


def resize_images(dataset_path: Path, destination: Path, size: Tuple[int, int] = (256, 256)):
    images = dataset_path / "images"
    segmentations = dataset_path / "segmentations"

    dest_images = destination / "images"
    dest_segments = destination / "segmentations"

    dest_images.mkdir(parents=True)
    dest_segments.mkdir(parents=True)

    for img_path in images.rglob("*.png"):
        image = cv2.imread(img_path)
        image = cv2.resize(image, size)
        cv2.imwrite(dest_images / img_path.name, image)

    for seg_path in segmentations.rglob("*.png"):
        segmentation = cv2.imread(seg_path)
        segmentation = cv2.resize(segmentation, size)
        cv2.imwrite(dest_segments / seg_path.name, segmentation)


if __name__ == "__main__":
    dataset_path = Path("data", "leedsbutterfly")
    dest_path = Path("data", "leedsbutterfly_resized")
    resize_images(dataset_path, dest_path)
