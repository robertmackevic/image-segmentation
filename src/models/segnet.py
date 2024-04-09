from torch import Tensor
from torch.nn import (
    Module,
    MaxPool2d,
    MaxUnpool2d,
    Softmax
)

from src.models.modules import ConvBlock
from src.utils import load_config


class SegNet(Module):
    def __init__(self) -> None:
        super(SegNet, self).__init__()
        config = load_config()
        in_channels = config.channels
        out_channels = len(config.classes) + 1

        self.enc_block_1 = ConvBlock(in_channels, 64, repeat=2)
        self.enc_block_2 = ConvBlock(64, 128, repeat=2)
        self.enc_block_3 = ConvBlock(128, 256, repeat=3)
        self.enc_block_4 = ConvBlock(256, 512, repeat=3)
        self.enc_block_5 = ConvBlock(512, 512, repeat=3)

        self.dec_block_1 = ConvBlock(512, 512, repeat=3)
        self.dec_block_2 = ConvBlock(512, 256, repeat=3)
        self.dec_block_3 = ConvBlock(256, 128, repeat=3)
        self.dec_block_4 = ConvBlock(128, 64, repeat=2)
        self.dec_block_5 = ConvBlock(64, out_channels, repeat=2)

        self.pool = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = MaxUnpool2d(kernel_size=2, stride=2)
        self.softmax = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        size_1 = x.size()
        x, indices_1 = self.pool(self.enc_block_1(x))
        size_2 = x.size()
        x, indices_2 = self.pool(self.enc_block_2(x))
        size_3 = x.size()
        x, indices_3 = self.pool(self.enc_block_3(x))
        size_4 = x.size()
        x, indices_4 = self.pool(self.enc_block_4(x))
        size_5 = x.size()
        x, indices_5 = self.pool(self.enc_block_5(x))

        # Decoder
        x = self.dec_block_1(self.unpool(x, indices_5, output_size=size_5))
        x = self.dec_block_2(self.unpool(x, indices_4, output_size=size_4))
        x = self.dec_block_3(self.unpool(x, indices_3, output_size=size_3))
        x = self.dec_block_4(self.unpool(x, indices_2, output_size=size_2))
        x = self.dec_block_5(self.unpool(x, indices_1, output_size=size_1))
        return self.softmax(x)
