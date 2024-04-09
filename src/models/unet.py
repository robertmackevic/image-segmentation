from torch import Tensor
from torch.nn import (
    Module,
    MaxPool2d,
    Conv2d,
    Softmax
)

from src.models.modules import ConvBlock, DecoderBlock


class UNet(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.enc_block_1 = ConvBlock(in_channels, 64, repeat=2)
        self.enc_block_2 = ConvBlock(64, 128, repeat=2)
        self.enc_block_3 = ConvBlock(128, 256, repeat=2)
        self.enc_block_4 = ConvBlock(256, 512, repeat=2)
        self.enc_block_5 = ConvBlock(512, 1024, repeat=2)

        self.dec_block_1 = DecoderBlock(1024, 512)
        self.dec_block_2 = DecoderBlock(512, 256)
        self.dec_block_3 = DecoderBlock(256, 128)
        self.dec_block_4 = DecoderBlock(128, 64)
        self.final = Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.softmax = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc_1 = self.enc_block_1(x)
        enc_2 = self.enc_block_2(self.pool(enc_1))
        enc_3 = self.enc_block_3(self.pool(enc_2))
        enc_4 = self.enc_block_4(self.pool(enc_3))
        x = self.enc_block_5(self.pool(enc_4))

        # Decoder
        x = self.dec_block_1(x, enc_4)
        x = self.dec_block_2(x, enc_3)
        x = self.dec_block_3(x, enc_2)
        x = self.dec_block_4(x, enc_1)
        x = self.final(x)
        return self.softmax(x)
