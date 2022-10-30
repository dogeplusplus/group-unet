import os
import json
import wandb
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F


from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import load_dotenv, find_dotenv
from einops import rearrange, repeat, reduce
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.geometric.rotate import SafeRotate
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.transforms import RandomBrightnessContrast

from group_unet.dataset import ButterflyDataset, BDD100K
from group_unet.group_unet import GroupUNet
from group_unet.unet import UNet
from group_unet.groups.cyclic import CyclicGroup


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    intersection = (preds & targets).sum()
    union = (preds | targets).sum()
    iou = intersection / (union + eps)
    return iou


def create_image_grid(images: torch.Tensor, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    image_samples = rearrange(images, "b h w c -> b c h w")
    image_grid = rearrange(make_grid(image_samples, nrow=4), "c h w -> h w c").numpy()
    mask_grid = rearrange(make_grid(repeat(labels, "b h w -> b 1 h w"), nrow=4), "c h w -> h w c").numpy()
    mask_grid = reduce(mask_grid, "h w c -> h w", "max")

    return image_grid, mask_grid


def preprocess_sample_images(images: torch.Tensor, transform: A.Compose) -> torch.Tensor:
    images = [transform(image=image.numpy())["image"] for image in images]
    images = torch.stack(images)

    return images


def evaluate_metric(x: torchmetrics.Metric) -> float:
    return x.compute().detach().cpu().numpy()


def prepare_butterfly_dataset(train_transform, val_transform):
    dataset = list(Path("data", "leedsbutterfly_resized", "images").rglob("*.png"))
    np.random.shuffle(dataset)
    validation_ratio = 0.2
    validation_size = int(len(dataset) * validation_ratio)
    train_images, val_images = dataset[validation_size:], dataset[:validation_size]

    dataset_split = {
        "train": list(map(str, train_images)),
        "valid": list(map(str, val_images)),
    }

    raw_train_ds = ButterflyDataset(train_images, transform=None)
    raw_val_ds = ButterflyDataset(val_images, transform=None)
    train_ds = ButterflyDataset(train_images, transform=train_transform)
    val_ds = ButterflyDataset(val_images, transform=val_transform)

    return raw_train_ds, raw_val_ds, train_ds, val_ds, dataset_split


def prepare_bdd100k_dataset(train_transform, val_transform):
    train_images = list(Path("data", "bdd100k", "images", "train").rglob("*.jpg"))
    val_images = list(Path("data", "bdd100k", "images", "val").rglob("*.jpg"))

    raw_train_ds = BDD100K(train_images, transform=None)
    raw_val_ds = BDD100K(val_images, transform=None)
    train_ds = BDD100K(train_images, transform=train_transform)
    val_ds = BDD100K(val_images, transform=val_transform)

    return raw_train_ds, raw_val_ds, train_ds, val_ds


def train_model():
    wandb.init(project="group-unet")
    epochs = wandb.config.epochs
    in_channels = 3
    out_channels = 1
    seed = 42
    batch_size = wandb.config.batch_size

    np.random.seed(seed)
    model_type = wandb.config.model_type
    filters = wandb.config.filters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "unet":
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernel_size=3,
            stride=1,
            activation=F.relu,
            res_block=True,
        )
    else:
        model = GroupUNet(
            group=CyclicGroup(4),
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernel_size=3,
            activation=F.relu,
            res_block=True,
        )

    minimal_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    full_transform = A.Compose([
        ColorJitter(),
        RandomBrightnessContrast(),
        SafeRotate(limit=90),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(),
        A.RandomRotate90(),
        ToTensorV2(),
    ])

    if wandb.config.full_augmentation:
        train_transform = full_transform
    else:
        train_transform = minimal_transform

    (
        raw_train_ds,
        raw_val_ds,
        train_ds,
        val_ds,
        dataset_split,
    ) = prepare_butterfly_dataset(train_transform, val_transform)

    file = wandb.Artifact("dataset_split", type="dataset", description="Train/Validation Split")
    with TemporaryDirectory() as temp_dir, open(Path(temp_dir, "dataset_split.json"), "w") as f:
        json.dump(dataset_split, f, indent=4)
        file.add_file(Path(temp_dir, "dataset_split.json"))
    wandb.log_artifact(file)

    # Log sample images
    num_samples = 8

    raw_train_loader = DataLoader(raw_train_ds, batch_size=batch_size, shuffle=True)
    train_samples, train_samples_gt = next(iter(raw_train_loader))
    train_samples = train_samples[:num_samples]
    train_samples_gt = train_samples_gt[:num_samples]

    train_image_grid, train_gt_grid = create_image_grid(train_samples, train_samples_gt)
    train_samples = preprocess_sample_images(train_samples, train_transform).to(device)

    raw_val_loader = DataLoader(raw_val_ds, batch_size=batch_size, shuffle=True)
    val_samples, val_samples_gt = next(iter(raw_val_loader))
    val_samples = val_samples[:num_samples]
    val_samples_gt = val_samples_gt[:num_samples]

    val_image_grid, val_gt_grid = create_image_grid(val_samples, val_samples_gt)
    val_samples = preprocess_sample_images(val_samples, val_transform).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )
    lr = wandb.config.lr
    step = 0
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    display_every = 10
    log_every = 5

    accumulation_steps = max(1, 64 // batch_size)
    THRESHOLD = 0.5

    def eval_metric(x): return x.compute().detach().cpu().numpy()

    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        scaler = GradScaler()
        train_bar = tqdm(train_loader, ncols=0, desc=f"Train Epoch {e}")
        train_loss = torchmetrics.MeanMetric().to(device)
        train_acc = torchmetrics.MeanMetric().to(device)

        for idx, (x, y) in enumerate(train_bar):
            x = x.to(device)
            y = y.to(device)

            with autocast():
                predictions = model(x)
                predictions = rearrange(predictions, "b 1 h w -> b h w")
                loss = loss_fn(predictions, y)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            jaccard = compute_iou(torch.sigmoid(predictions) > THRESHOLD, y.to(torch.uint8))
            train_loss.update(loss)
            train_acc.update(jaccard)

            if idx % display_every == 0:
                train_bar.set_postfix({
                    "loss": evaluate_metric(train_loss),
                    "accuracy": evaluate_metric(train_acc),
                })

            step += 1

        val_loss = torchmetrics.MeanMetric().to(device)
        val_acc = torchmetrics.MeanMetric().to(device)
        val_bar = tqdm(val_loader, ncols=0, desc=f"Valid Epoch {e}")
        with torch.no_grad():
            for x, y in val_bar:
                x = x.to(device)
                y = y.to(device)
                predictions = model(x)
                predictions = rearrange(predictions, "b 1 h w -> b h w")
                loss = loss_fn(predictions, y)
                val_loss.update(loss)
                jaccard = compute_iou(torch.sigmoid(predictions) > THRESHOLD, y.to(torch.uint8))
                val_acc.update(jaccard)

                if step % display_every == 0:
                    val_bar.set_postfix({
                        "loss": evaluate_metric(val_loss),
                        "accuracy": evaluate_metric(val_acc),
                    })
                step += 1

        metrics = {
            "step": step,
            "train/loss": evaluate_metric(train_loss),
            "train/acc": evaluate_metric(train_acc),
            "val/loss": eval_metric(val_loss),
            "val/acc": eval_metric(val_acc),
        }

        # Add image predictions every so often
        if e % log_every == 0:
            train_sample_preds = torch.sigmoid(model(train_samples)) > 0.5
            train_pred_grid = rearrange(make_grid(train_sample_preds, nrow=4), "c h w -> h w c").detach().cpu().numpy()
            train_pred_grid = reduce(train_pred_grid, "h w c -> h w", "max")

            val_sample_preds = torch.sigmoid(model(val_samples)) > 0.5
            val_pred_grid = rearrange(make_grid(val_sample_preds, nrow=4), "c h w -> h w c").detach().cpu().numpy()
            val_pred_grid = reduce(val_pred_grid, "h w c -> h w", "max")

            # Manipulate the color to override chosen color of the foreground mask
            color = 3
            class_labels = {
                k * color: v for k, v in train_ds.class_labels.items()
            }

            # Log foreground as 2 to render as a different color in wandb than the default
            epoch_prediction = {
                "Train Images": wandb.Image(
                    train_image_grid, caption="Train Predictions", masks={
                        "ground_truth": {
                            "mask_data": train_gt_grid * color,
                            "class_labels": class_labels,
                        },
                        "predictions": {
                            "mask_data": train_pred_grid * color,
                            "class_labels": class_labels,
                        },
                    }),
                "Validation Images": wandb.Image(
                    val_image_grid, caption="Validation Predictions", masks={
                        "ground_truth": {
                            "mask_data": val_gt_grid * color,
                            "class_labels": class_labels
                        },
                        "predictions": {
                            "mask_data": val_pred_grid * color,
                            "class_labels": class_labels,
                        },
                    }),
            }
            metrics.update(epoch_prediction)

        wandb.log(metrics)

    with TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir, "model.pth")
        torch.save(model.state_dict(), save_path)

        model_artifact = wandb.Artifact("model", type="model")
        model_artifact.add_file(save_path)

        wandb.log_artifact(model_artifact)


def sweep_run():
    sweep_configuration = {
        "method": "random",
        "name": f"{os.environ['WANDB_USERNAME']}/group-unet/sweep",
        "metric": {"goal": "minimize", "name": "val/loss"},
        "parameters": {
            "model_type": {"values": ["unet", "group_unet"]},
            "lr": {"values": [1e-4]},
            "filters": {"values": [[16, 16, 32, 32], [32, 32, 64, 64]]},
            "epochs": {"values": [100]},
            "batch_size": {"values": [8]},
            "full_augmentation": {"values": [True, False]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="group-unet")
    wandb.agent(sweep_id, function=train_model, count=8)


def single_run():
    wandb.init(project="group-unet")
    wandb.config.model_type = "unet"
    wandb.config.filters = [32, 32, 64, 64]
    wandb.config.lr = 1e-4
    wandb.config.epochs = 1
    wandb.config.batch_size = 32
    wandb.config.full_augmentation = True
    train_model()


def main():
    load_dotenv(find_dotenv())
    single_run()


if __name__ == "__main__":
    main()
