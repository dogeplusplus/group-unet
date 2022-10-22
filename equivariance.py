import torch
import torchmetrics
import torch.nn as nn
import numpy as np
import albumentations as A
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from group_unet.dataset import ButterflyDataset
from group_unet.group_unet import GroupUNet
from group_unet.unet import UNet
from group_unet.groups.cyclic import CyclicGroup


def main():
    seed = 42
    batch_size = 128
    dataset = list(Path("data", "leedsbutterfly", "images").rglob("*.png"))
    validation_ratio = 0.2
    validation_size = int(len(dataset) * validation_ratio)
    np.random.seed(seed)
    np.random.shuffle(dataset)
    train_images, val_images = dataset[validation_size:], dataset[:validation_size]

    train_transform = A.Compose([
        A.Normalize(),
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(),
        A.Resize(256, 256),
        A.RandomRotate90(),
        ToTensorV2(),
    ])

    train_ds = ButterflyDataset(train_images, transform=train_transform)
    val_ds = ButterflyDataset(val_images, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        filters=[32, 32, 64, 64],
        kernel_size=3,
        stride=1,
        activation=F.relu,
    )

    group_model = GroupUNet(
        group=CyclicGroup(4),
        in_channels=3,
        out_channels=1,
        filters=[16, 16, 32, 32],
        kernel_size=3,
        activation=F.relu,
    )

    print(sum(x.numel() for x in model.parameters() if x.requires_grad))
    print(sum(x.numel() for x in group_model.parameters() if x.requires_grad))

    epochs = 50
    display_every = 10
    step = 0
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    accumulation_steps = max(1, 64 // batch_size)

    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        scaler = GradScaler()
        train_bar = tqdm(train_loader, ncols=0, desc=f"Train Epoch {e}")
        train_loss = torchmetrics.MeanMetric().to(device)
        train_acc = torchmetrics.MeanMetric().to(device)
        iou = torchmetrics.JaccardIndex(num_classes=2).to(device)

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

            jaccard = iou(predictions, torch.sigmoid(y).to(torch.uint8))
            train_loss.update(loss)
            train_acc.update(jaccard)

            if idx % display_every == 0:
                train_bar.set_postfix({
                    "loss": train_loss.compute().detach().cpu().numpy(),
                    "accuracy": train_acc.compute().detach().cpu().numpy(),
                })

        val_loss = torchmetrics.MeanMetric().to(device)
        val_acc = torchmetrics.MeanMetric().to(device)
        val_bar = tqdm(val_loader, ncols=0, desc=f"Valid Epoch {e}")
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                predictions = model(x)
                predictions = rearrange(predictions, "b 1 h w -> b h w")
                loss = loss_fn(predictions, y)
                val_loss.update(loss)
                jaccard = iou(predictions, torch.sigmoid(y).to(torch.uint8))
                val_acc.update(jaccard)

            if step % display_every == 0:
                val_bar.set_postfix({
                    "loss": val_loss.compute().detach().cpu().numpy(),
                    "accuracy": val_acc.compute().detach().cpu().numpy(),
                })
            step += 1


if __name__ == "__main__":
    main()
