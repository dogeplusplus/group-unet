import cv2
import wandb
import torch
import numpy as np
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.animation as animation

from einops import repeat
from typing import Tuple, List
from pathlib import Path
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.transforms.functional import rotate
from albumentations.pytorch.transforms import ToTensorV2

from group_unet.unet import UNet
from equivariance import compute_iou
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
    h, w, _ = image.shape

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
    crop_w = (pad_size - w) // 2
    crop_h = (pad_size - h) // 2
    predictions = predictions[:, crop_h:-crop_h, crop_w:-crop_w]
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
    color: Tuple[int, int, int] = (0, 0, 255),
):
    predictions = [x.astype(np.uint8) * 255 for x in predictions]
    predictions = [overlay(image, pred, color) for pred in predictions]

    frames = []
    for pred in predictions:
        frame = Image.fromarray(pred)
        frames.append(frame)

    return frames


def equivariance_scoring(model: nn.Module, image: np.ndarray, ground_truth: np.ndarray):
    num_angles = 30
    angles = np.linspace(0, 360, num_angles)
    predictions = batched_prediction(model, image, angles).numpy()

    ious = [compute_iou(ground_truth, pred) for pred in predictions]
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Rotation Equivariance Performance")
    ax[0, 0].set_title("Image")
    ax[0, 0].imshow(image)
    ax[0, 1].set_title("Ground Truth")
    ax[0, 1].matshow(ground_truth)

    ax[1, 0].plot(angles, ious)
    ax[1, 1].hist(ious)
    ax[1, 0].set_xlabel("Rotation Angle")
    ax[1, 0].set_ylabel("IoU Score")

    ax[1, 1].set_xlabel("IoU Score")
    ax[1, 1].set_ylabel("Density")

    plt.show()


def visualise_equivariance(
    model: nn.Module,
    image: np.ndarray,
    ground_truth: np.ndarray,
    color: Tuple[int, int, int],
    destination: str = "equivariance.gif",
):
    num_angles = 30
    angles = np.linspace(0, 360, num_angles)
    predictions = batched_prediction(model, image, angles).numpy()
    # Pad gt to preserve information during rotation
    frames_pred = create_gif(image, predictions, color)

    ious = [compute_iou(pred, ground_truth) for pred in predictions]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ims = []

    ax[0].axis("off")
    ax[1].set_aspect("auto")

    ax[1].plot(angles, ious)
    ax[1].set_xlabel("Angle")
    ax[1].set_ylabel("IoU Score")

    mpl_color = tuple(x / 255 for x in color)
    for i, (pred, angle) in enumerate(zip(frames_pred, angles)):
        pred_plot = ax[0].imshow(pred, animated=True)
        title = ax[0].text(0.5, 1.01, f"Angle: {angle:.0f}", horizontalalignment="center", transform=ax[0].transAxes)
        scores = ax[1].plot(angles, ious, animated=True, c=mpl_color)[0]
        if i == 0:
            ax[0].imshow(pred)
            ax[1].plot(angles, ious, c=mpl_color)
        ims.append([pred_plot, scores, title])

    # Need to assign ani variable, otherwise animation drops for some reason
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=0)
    ani.save(destination)
