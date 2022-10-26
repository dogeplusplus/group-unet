import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from group_unet.groups.base import InterpolativeLiftingKernel, GroupKernelBase


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LiftingConvolution(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = InterpolativeLiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        conv_kernels = self.kernel.sample()
        conv_kernels = rearrange(conv_kernels, "o g i h w -> (o g) i h w")
        x = F.conv2d(
            input=x,
            weight=conv_kernels,
            padding="same",
        )

        x = rearrange(x, "b (c g) h w -> b c g h w", c=self.out_channels)
        return x


class InterpolativeGroupKernel(GroupKernelBase):
    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        self.weight = nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size,
        ), device=self.group.identity.device))

        nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

    def sample(self):
        group_size = self.group.elements().numel()
        weight = self.weight.view(
            1,
            self.out_channels * self.in_channels,
            group_size,
            self.kernel_size,
            self.kernel_size,
        )

        weight = repeat(
            self.weight,
            "o i g h w -> g2 (o i) g h w",
            g2=group_size,
        )
        transformed_weight = F.grid_sample(
            weight,
            self.transformed_grid_R2xH,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        transformed_weight = repeat(
            transformed_weight,
            "g2 (o i) g h w -> o g2 i g h w",
            g2=group_size,
            o=self.out_channels,
            i=self.in_channels,
        )

        return transformed_weight


class GroupConvolution(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = InterpolativeGroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = rearrange(x, "b c g h w -> b (c g) h w")
        conv_kernels = self.kernel.sample()
        conv_kernels = rearrange(
            conv_kernels, "o g i g2 h w -> (o g) (i g2) h w")
        x = F.conv2d(
            input=x,
            weight=conv_kernels,
            padding="same",
        )
        x = rearrange(x, "b (c g) h w -> b c g h w", c=self.out_channels)
        return x
