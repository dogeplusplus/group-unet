import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        activation,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size,
            stride,
            padding="same",
        )
        self.activation = activation
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=self.out_channel)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=self.out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filters,
        kernel_size,
        stride,
        activation,
    ):
        super().__init__()
        pairs = list(zip(filters[:-1], filters[1:]))
        self.init_conv = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding="same")
    
        self.down_convs = nn.ModuleList([
            Block(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                activation,
            ) for (in_channel, out_channel) in pairs
        ])

        # Reverse filters and pairs
        # Factor of two to account for concatenation
        self.up_convs = nn.ModuleList([
            Block(
                in_channel * 2,
                out_channel,
                kernel_size,
                stride,
                activation,
            ) for (out_channel, in_channel) in pairs[::-1]
        ])

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=3, padding="same")
 
    def forward(self, x):
        x = self.init_conv(x)

        skips = []
        for down in self.down_convs:
            x = down(x)
            skips.insert(0, x)
            x = nn.AvgPool2d((2, 2))(x)

        for skip, up in zip(skips, self.up_convs):
            x = nn.Upsample(scale_factor=2)(x)
            x = torch.cat([skip, x], dim=1)
            x = up(x)
        
        x = self.final_conv(x)
        return x

