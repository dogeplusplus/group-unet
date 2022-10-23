import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class GroupBase(nn.Module):
    def __init__(self, dimension, identity):
        super().__init__()
        self.dimension = dimension
        self.register_buffer("identity", torch.tensor(identity))

    def elements(self):
        raise NotImplementedError()

    def product(self, h, h_prime):
        raise NotImplementedError()

    def inverse(self, h):
        raise NotImplementedError()

    def left_action_on_R2(self, h_batch, x_batch):
        raise NotImplementedError()

    def left_action_on_H(self, h_batch, h_prime_batch):
        raise NotImplementedError()

    def matrix_representation(self, h):
        raise NotImplementedError()

    def determinant(self, h):
        raise NotImplementedError()

    def normalize_group_elements(self, h):
        raise NotImplementedError()


class GroupKernelBase(nn.Module):
    def __init__(self, group: GroupBase, kernel_size, in_channels, out_channels):
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_R2xH",
                             self.create_transformed_grid_R2xH())

    def create_transformed_grid_R2xH(self):
        group_elements = self.group.elements()
        transformed_grid_R2 = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2,
        )
        transformed_grid_H = self.group.left_action_on_H(
            self.group.inverse(group_elements), self.grid_H,
        )
        transformed_grid_H = self.group.normalize_group_elements(
            transformed_grid_H)
        transformed_grid = torch.cat(
            (
                repeat(
                    transformed_grid_R2,
                    "g h w d -> g g2 h w d",
                    g=self.group.elements().numel(),
                    g2=self.group.elements().numel(),
                ),
                repeat(
                    transformed_grid_H,
                    "g1 g2 -> g1 g2 h w 1",
                    h=self.kernel_size,
                    w=self.kernel_size,
                ),
            ), dim=-1
        )

        return transformed_grid

    def sample(self, sampled_group_elements):
        raise NotImplementedError()


class LiftingKernelBase(torch.nn.Module):
    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        self.register_buffer("transformed_grid_R2",
                             self.create_transformed_grid_R2())

    def create_transformed_grid_R2(self):
        group_elements = self.group.elements()

        transformed_grid = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2,
        )
        return transformed_grid

    def sample(self, sampled_group_elements):
        raise NotImplementedError()


class InterpolativeLiftingKernel(LiftingKernelBase):
    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ), device=self.group.identity.device))

        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

    def sample(self):
        weight = rearrange(self.weight, " o i h w -> 1 (o i) h w")
        weight = repeat(weight, "1 n h w -> g n h w",
                        g=self.group.elements().numel())

        transformed_weight = F.grid_sample(
            weight,
            self.transformed_grid_R2,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        transformed_weight = rearrange(
            transformed_weight, "g (o i) h w -> g o i h w", o=self.out_channels, i=self.in_channels
        )
        transformed_weight = rearrange(
            transformed_weight, "g o ... -> o g ...")
        return transformed_weight
