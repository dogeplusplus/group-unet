import torch
import torch.nn as nn

from einops import reduce

from group_unet.layers import GroupConvolution, LiftingConvolution, Residual


class GroupBlock(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, activation):
        super().__init__()
        self.group_conv1 = GroupConvolution(
            group, in_channels, out_channels, kernel_size)
        self.group_conv2 = GroupConvolution(
            group, out_channels, out_channels, kernel_size)

        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.group_conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.group_conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x


class ResGroupBlock(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, activation):
        super().__init__()
        self.activation = activation
        self.residual = Residual(GroupBlock(group, in_channels, in_channels, kernel_size, activation))
        self.proj = GroupConvolution(group, in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x):
        x = self.residual(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class GroupUNet(nn.Module):
    def __init__(
        self,
        group,
        in_channels,
        out_channels,
        filters,
        kernel_size,
        activation,
        res_block=False,
    ):
        super().__init__()
        self.ord = group.elements().numel()
        pairs = list(zip(filters[:-1], filters[1:]))
        self.lifting_conv = LiftingConvolution(
            group, in_channels, filters[0], kernel_size=3)

        block = ResGroupBlock if res_block else GroupBlock

        self.down_convs = nn.ModuleList([
            block(
                group,
                in_channel,
                out_channel,
                kernel_size,
                activation,
            ) for (in_channel, out_channel) in pairs
        ])

        # Reverse filters and pairs
        # Factor of two to account for concatenation
        self.up_convs = nn.ModuleList([
            block(
                group,
                in_channel * 2,
                out_channel,
                kernel_size,
                activation,
            ) for (out_channel, in_channel) in pairs[::-1]
        ])

        self.final_conv = GroupConvolution(
            group, filters[0], out_channels, kernel_size)

    def forward(self, x):
        x = self.lifting_conv(x)

        skips = []
        for down in self.down_convs:
            x = down(x)
            skips.insert(0, x)
            x = nn.AvgPool3d((1, 2, 2))(x)

        for skip, up in zip(skips, self.up_convs):
            x = nn.Upsample(scale_factor=(1, 2, 2))(x)
            x = torch.cat([skip, x], dim=1)
            x = up(x)

        # Final convolution and group projection by averaging
        x = self.final_conv(x)
        x = reduce(x, "b c g h w -> b c h w", "mean")
        return x
