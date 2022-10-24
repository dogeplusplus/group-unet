import wandb
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from einops import rearrange, repeat, reduce
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from albumentations.pytorch.transforms import ToTensorV2

from group_unet.dataset import ButterflyDataset
from group_unet.group_unet import GroupUNet
from group_unet.unet import UNet
from group_unet.groups.cyclic import CyclicGroup


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    intersection = (preds & targets).sum()
    union = (preds | targets).sum()
    iou = intersection / (union + eps)
    return iou


def main():
    wandb.init(project="group-unet")

    epochs = 50
    in_channels = 3
    out_channels = 1
    seed = 42
    batch_size = 64
    dataset = list(Path("data", "leedsbutterfly_resized", "images").rglob("*.png"))
    validation_ratio = 0.2
    validation_size = int(len(dataset) * validation_ratio)
    np.random.seed(seed)
    np.random.shuffle(dataset)
    train_images, val_images = dataset[validation_size:], dataset[:validation_size]
    model_type = "unet"
    filters = [32, 32, 64, 64]

    wandb.config.update(dict(
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        model_type=model_type,
        filters=filters,
        in_channels=in_channels,
        out_channels=out_channels,
    ))

    if model_type == "unet":
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernel_size=3,
            stride=1,
            activation=F.relu,
        )
    else:
        model = GroupUNet(
            group=CyclicGroup(4),
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernel_size=3,
            activation=F.relu,
        )

    train_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(),
        A.RandomRotate90(),
        ToTensorV2(),
    ])

    raw_ds = ButterflyDataset(train_images, transform=None)
    train_ds = ButterflyDataset(train_images, transform=train_transform)
    val_ds = ButterflyDataset(val_images, transform=val_transform)

    num_samples = 8
    raw_loader = DataLoader(raw_ds, batch_size=batch_size)
    sample_images, sample_labels = next(iter(raw_loader))
    sample_images = rearrange(sample_images, "b h w c -> b c h w")
    sample_images = sample_images[:num_samples]
    sample_labels = sample_labels[:num_samples]
    image_grid = rearrange(make_grid(sample_images, nrow=4), "c h w -> h w c").numpy()
    mask_grid = rearrange(make_grid(repeat(sample_labels, "b h w -> b 1 h w"), nrow=4), "c h w -> h w c").numpy()
    mask_grid = reduce(mask_grid, "h w c -> h w", "max")

    class_labels = {
        1: "butterfly",
    }

    wandb.log({
        "images": wandb.Image(
            image_grid, caption="Images", masks={
                "ground_truth": {
                    "mask_data": mask_grid,
                    "class_labels": class_labels,
                }
            })
    })

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

    print(sum(x.numel() for x in model.parameters() if x.requires_grad))

    step = 0
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    display_every = 1

    accumulation_steps = max(1, 64 // batch_size)
    THRESHOLD = 0.5

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

            if (idx+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            jaccard = compute_iou(torch.sigmoid(predictions) > THRESHOLD, y.to(torch.uint8))
            train_loss.update(loss)
            train_acc.update(jaccard)

            if idx % display_every == 0:
                train_bar.set_postfix({
                    "loss": train_loss.compute().detach().cpu().numpy(),
                    "accuracy": train_acc.compute().detach().cpu().numpy(),
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
                        "loss": val_loss.compute().detach().cpu().numpy(),
                        "accuracy": val_acc.compute().detach().cpu().numpy(),
                    })
                step += 1


if __name__ == "__main__":
    main()
