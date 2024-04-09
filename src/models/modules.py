import torch
from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    ConvTranspose2d
)


class ConvBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, repeat: int) -> None:
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(repeat):
            layers += [
                Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(out_channels),
                ReLU(inplace=True),
            ]
            in_channels = out_channels

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DecoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.trans_conv = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv_block = ConvBlock(in_channels, out_channels, repeat=2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.trans_conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x)
        return x
