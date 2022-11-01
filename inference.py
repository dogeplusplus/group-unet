import cv2
import wandb
import torch
import random
import imageio
import numpy as np
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import Tuple, List
from pathlib import Path
from einops import repeat
from scipy.ndimage import rotate
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory
from albumentations.pytorch.transforms import ToTensorV2

from group_unet.unet import UNet
from group_unet.group_unet import GroupUNet
from group_unet.groups.cyclic import CyclicGroup


def load_model(model_uri: str) -> nn.Module:
    api = wandb.Api()
    artifact = api.artifact(model_uri)

    run_id = artifact.logged_by().id
    run = api.run(f"group-unet/{run_id}")
    config = run.config

    if config["model_type"] == "unet":
        model = UNet(
            config["in_channels"],
            config["out_channels"],
            config["filters"],
            config["kernel_size"],
            config["stride"],
            F.relu,
            config["res_block"]
        )
    elif config["model_type"] == "group_unet":
        model = GroupUNet(
            CyclicGroup(4),
            config["in_channels"],
            config["out_channels"],
            config["filters"],
            config["kernel_size"],
            F.relu,
            config["res_block"],
        )

    with TemporaryDirectory() as temp_dir:
        artifact.download(temp_dir)
        state_dict = torch.load(Path(temp_dir, "model.pth"))

    model.load_state_dict(state_dict)

    return model


def batched_prediction(
    model: nn.Module,
    image: np.ndarray,
    angles: List[float],
) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images = []
    for angle in angles:
        images.append(rotate(image, angle, reshape=False))

    preprocessing = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])
    images = np.stack([preprocessing(image=image)["image"] for image in images])
    images = torch.from_numpy(images)

    batch_size = 4
    predictions = []
    for i in range(0, images.shape[0], batch_size):
        pred = torch.sigmoid(model(images[i:i+batch_size])) > 0.5
        pred = torch.squeeze(pred)
        predictions.append(pred)

    predictions = torch.concat(predictions, dim=0)
    return predictions


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    color = repeat(np.array(color), "x -> 1 1 x")
    colored_mask = repeat(mask, "h w -> h w 3")
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined


def create_gif(
    image: torch.Tensor,
    predictions: torch.Tensor,
    angles: List[float],
    dest: Path
):
    predictions = [x.numpy()[0].astype(np.uint8) * 255 for x in predictions]
    predictions = [rotate(pred, -angle, reshape=False) for pred, angle in zip(predictions, angles)]
    predictions = [overlay(image, pred) for pred in predictions]

    frames = []
    for angle, pred in zip(angles, predictions):
        frame = Image.fromarray(pred)
        draw = ImageDraw.Draw(frame)
        draw.text((0, 0), f"Angle: {angle}", (255, 255, 255))
        frames.append(frame)

    imageio.mimsave(dest, frames, fps=8)


def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.sum(np.equal(ground_truth, prediction))
    union = np.sum(ground_truth) + np.sum(prediction)
    return 2 * intersection / union


def evaluate_rotation_equivariance(model: nn.Module, image: np.ndarray, ground_truth: np.ndarray):
    num_angles = 30
    angles = np.linspace(0, 360, num_angles)
    predictions = batched_prediction(model, image, angles).numpy()

    ground_truths = np.stack([
        rotate(ground_truth, angle, reshape=False) for angle in angles
    ])

    dices = [dice_score(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Rotation Equivariance Performance")
    ax[0, 0].set_title("Image")
    ax[0, 0].imshow(image)
    ax[0, 1].set_title("Ground Truth")
    ax[0, 1].matshow(ground_truth)

    ax[1, 0].plot(angles, dices)
    ax[1, 1].hist(dices)
    ax[1, 0].set_xlabel("Rotation Angle")
    ax[1, 0].set_ylabel("Dice Score")

    ax[1, 1].set_xlabel("Dice Score")
    ax[1, 1].set_ylabel("Density")

    plt.show()


def main():
    images = list(Path("data", "leedsbutterfly_resized", "images").rglob("*.png"))
    image_path = str(random.choice(images))
    gt_path = image_path.replace("images", "segmentations").replace(".png", "_seg0.png")

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255

    model_path = Path("artifacts", "model:v1", "model.pth")
    state_dict = torch.load(model_path)
    model = GroupUNet(
        group=CyclicGroup(4),
        in_channels=3,
        out_channels=1,
        filters=[32, 32, 64, 64],
        kernel_size=3,
        activation=F.relu,
        res_block=True,
    )
    model.load_state_dict(state_dict)
    evaluate_rotation_equivariance(model, image, gt)


if __name__ == "__main__":
    main()
