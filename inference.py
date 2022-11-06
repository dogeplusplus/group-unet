import cv2
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.animation as animation

from einops import repeat
from typing import Tuple, List
from pathlib import Path
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory
from torchvision.transforms.functional import rotate
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
            1,
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
    pad_size: int = 384,
) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    assert h == w, "Shape is not square"

    # Pad and rotate images
    preprocessing = A.Compose([
        A.PadIfNeeded(pad_size, pad_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(),
        ToTensorV2(),
    ])
    image = preprocessing(image=image)["image"]
    images = torch.stack([rotate(image, angle) for angle in angles])

    batch_size = 4
    predictions = []
    for i in range(0, images.shape[0], batch_size):
        pred = torch.sigmoid(model(images[i:i+batch_size])) > 0.5
        pred = torch.squeeze(pred)
        predictions.append(pred)

    # Invert padding and centre crop
    predictions = torch.concat(predictions, dim=0)
    predictions = torch.concat([
        rotate(repeat(pred, "... -> 1 ..."), -angle) for pred, angle in zip(predictions, angles)
    ], dim=0)
    crop = (pad_size - w) // 2
    predictions = predictions[:, crop:-crop, crop:-crop]
    return predictions


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    out = image.copy()
    image_overlay = image.copy()
    image_overlay[np.where(mask)] = color
    image_combined = cv2.addWeighted(image_overlay, 1 - alpha, out, alpha, 0, out)
    return image_combined


def create_gif(
    image: np.ndarray,
    predictions: np.ndarray,
    angles: List[float],
):
    predictions = [x.astype(np.uint8) * 255 for x in predictions]
    predictions = [overlay(image, pred) for pred in predictions]

    frames = []
    for angle, pred in zip(angles, predictions):
        frame = Image.fromarray(pred)
        draw = ImageDraw.Draw(frame)
        draw.text((0, 0), f"Angle: {angle}", (255, 255, 255))
        frames.append(frame)

    return frames


def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.sum(np.equal(ground_truth, prediction))
    union = np.sum(ground_truth) + np.sum(prediction)
    return 2 * intersection / union


def equivariance_scoring(model: nn.Module, image: np.ndarray, ground_truth: np.ndarray):
    num_angles = 30
    angles = np.linspace(0, 360, num_angles)
    predictions = batched_prediction(model, image, angles).numpy()

    dices = [dice_score(ground_truth, pred) for pred in predictions]
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


def visualise_equivariance(model: nn.Module, image: np.ndarray, ground_truth: np.ndarray):
    num_angles = 30
    angles = np.linspace(0, 360, num_angles)
    predictions = batched_prediction(model, image, angles).numpy()
    # Pad gt to preserve information during rotation
    frames_pred = create_gif(image, predictions, angles)

    fig, ax = plt.subplots(1, 2)
    ims = []

    for i, pred in enumerate(frames_pred):
        p = ax[0].imshow(pred, animated=True)
        g = ax[1].imshow(ground_truth, animated=True)
        if i == 0:
            ax[0].imshow(pred)
            ax[1].imshow(ground_truth)
        ims.append([p, g])

    # Need to assign ani variable, otherwise animation drops for some reason
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
    # Make linter happy
    assert ani is not None
    plt.show()


def main():
    images = list(Path("data", "leedsbutterfly_resized", "images").rglob("*.png"))
    image_path = str(random.choice(images))
    gt_path = image_path.replace("images", "segmentations").replace(".png", "_seg0.png")

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255

    model = load_model("dogeplusplus/group-unet/model:v9")
    visualise_equivariance(model, image, gt)


if __name__ == "__main__":
    main()
