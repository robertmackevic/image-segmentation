from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor, no_grad
from torch.nn import Module

from src.utils import load_config, compose_transform, get_available_device


class Segmentor:
    def __init__(self, model: Module) -> None:
        self.device = get_available_device()
        self.model = model.to(self.device)
        self.config = load_config()
        self.num_classes = len(self.config.classes)
        self.transform = compose_transform((self.config.image_size, self.config.image_size))
        self.colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def segment(self, image: Tensor) -> None:
        self.model.eval()
        with no_grad():
            image = image.unsqueeze(0).to(self.device)
            predicted_labels = self.model(image).argmax(dim=1)
            masks = [torch.eq(predicted_labels, idx)[0] for idx in range(len(self.config.classes))]
            self.visualize_mask(image[0], torch.stack(masks))

    def segment_url(self, url: str) -> None:
        image = Image.open(BytesIO(requests.get(url).content))
        image = self.transform(image)
        self.segment(image)

    def visualize_mask(self, image: Tensor, masks: Tensor) -> None:
        image = image.permute(1, 2, 0).cpu().numpy()
        masks = masks.permute(1, 2, 0).cpu().numpy()

        for i in range(self.num_classes):
            mask = masks[:, :, i]
            image[mask > 0] = (image[mask > 0] * 0.6 + self.colors[i % self.colors.shape[0]] * 0.4)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Semantic Segmentation Mask")
        plt.show()
