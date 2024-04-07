from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    MaxUnpool2d,
    Softmax
)
from torch.nn.init import xavier_uniform_
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class SegNet(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SegNet, self).__init__()
        self.enc_block_1 = self._create_conv_block(in_channels, 64, 2)
        self.enc_block_2 = self._create_conv_block(64, 128, 2)
        self.enc_block_3 = self._create_conv_block(128, 256, 3)
        self.enc_block_4 = self._create_conv_block(256, 512, 3)
        self.enc_block_5 = self._create_conv_block(512, 512, 3)

        self.dec_block_5 = self._create_conv_block(512, 512, 3)
        self.dec_block_4 = self._create_conv_block(512, 256, 3)
        self.dec_block_3 = self._create_conv_block(256, 128, 3)
        self.dec_block_2 = self._create_conv_block(128, 64, 2)
        self.dec_block_1 = self._create_conv_block(64, out_channels, 2)

        self.pool = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = MaxUnpool2d(kernel_size=2, stride=2)
        self.softmax = Softmax(dim=1)
        self._init_weights()

    @staticmethod
    def _create_conv_block(in_channels: int, out_channels: int, num_layers: int) -> Sequential:
        layers = []
        for _ in range(num_layers):
            layers += [
                Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(out_channels),
                ReLU()
            ]
            in_channels = out_channels
        return Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, Conv2d):
                xavier_uniform_(module.weight.data)

        vgg16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

        self.enc_block_1[0].weight.data = vgg16.features[0].weight.data
        self.enc_block_1[1].weight.data = vgg16.features[1].weight.data
        self.enc_block_1[3].weight.data = vgg16.features[3].weight.data
        self.enc_block_1[4].weight.data = vgg16.features[4].weight.data

        self.enc_block_2[0].weight.data = vgg16.features[7].weight.data
        self.enc_block_2[1].weight.data = vgg16.features[8].weight.data
        self.enc_block_2[3].weight.data = vgg16.features[10].weight.data
        self.enc_block_2[4].weight.data = vgg16.features[11].weight.data

        self.enc_block_3[0].weight.data = vgg16.features[14].weight.data
        self.enc_block_3[1].weight.data = vgg16.features[15].weight.data
        self.enc_block_3[3].weight.data = vgg16.features[17].weight.data
        self.enc_block_3[4].weight.data = vgg16.features[18].weight.data
        self.enc_block_3[6].weight.data = vgg16.features[20].weight.data
        self.enc_block_3[7].weight.data = vgg16.features[21].weight.data

        self.enc_block_4[0].weight.data = vgg16.features[24].weight.data
        self.enc_block_4[1].weight.data = vgg16.features[25].weight.data
        self.enc_block_4[3].weight.data = vgg16.features[27].weight.data
        self.enc_block_4[4].weight.data = vgg16.features[28].weight.data
        self.enc_block_4[6].weight.data = vgg16.features[30].weight.data
        self.enc_block_4[7].weight.data = vgg16.features[31].weight.data

        self.enc_block_5[0].weight.data = vgg16.features[34].weight.data
        self.enc_block_5[1].weight.data = vgg16.features[35].weight.data
        self.enc_block_5[3].weight.data = vgg16.features[37].weight.data
        self.enc_block_5[4].weight.data = vgg16.features[38].weight.data
        self.enc_block_5[6].weight.data = vgg16.features[40].weight.data
        self.enc_block_5[7].weight.data = vgg16.features[41].weight.data

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
        x = self.dec_block_5(self.unpool(x, indices_5, output_size=size_5))
        x = self.dec_block_4(self.unpool(x, indices_4, output_size=size_4))
        x = self.dec_block_3(self.unpool(x, indices_3, output_size=size_3))
        x = self.dec_block_2(self.unpool(x, indices_2, output_size=size_2))
        x = self.dec_block_1(self.unpool(x, indices_1, output_size=size_1))
        return self.softmax(x)