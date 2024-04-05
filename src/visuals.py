from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

from src.model import SegNet
from src.utils import load_weights, load_config


class Segmentor:
    def __init__(self, model_weights_path: Optional[Path] = None) -> None:
        self.config = load_config()
        self.num_classes = len(self.config.classes)
        self.model = SegNet(in_channels=3, out_channels=self.num_classes)

        if model_weights_path is not None:
            self.model = load_weights(model_weights_path, self.model)

        self.colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def segment(self) -> None:
        pass

    def segment_url(self) -> None:
        pass

    def visualize_mask(self, image: Tensor, masks: Tensor) -> None:
        image = image.permute(1, 2, 0).cpu().numpy()
        masks = masks.permute(1, 2, 0).cpu().numpy()

        for i in range(self.num_classes):
            mask = masks[:, :, i]
            image[mask > 0] = (image[mask > 0] * 0.5 + self.colors[i % self.colors.shape[0]] * 0.5)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Semantic Segmentation Mask")
        plt.show()
