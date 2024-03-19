from argparse import Namespace
from functools import reduce
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from fiftyone import Sample, Dataset as FODataset
from numpy.typing import NDArray
from torch import BoolTensor, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

from src.utils import get_logger


class COCODataset(Dataset):
    def __init__(self, data: FODataset, config: Namespace) -> None:
        self.config = config
        self.samples: List[Dict[str, Any]] = []
        self.resolution = (self.config.image_size, self.config.image_size)
        self.transform = Compose([
            Resize(self.resolution),
            ToTensor()
        ])

        # noinspection PyTypeChecker
        for sample in data.select_fields(["filepath", "ground_truth"]):
            self.samples.append({
                "image_filepath": sample.filepath,
                "masks": self._create_semantic_masks(sample)
            })

    def _create_semantic_masks(self, sample: Sample) -> Dict[str, NDArray]:
        return {
            label: self._reduce_to_single_mask([
                instance.to_segmentation(frame_size=self.resolution).mask.astype(bool)
                for instance in sample.ground_truth.detections
                if instance.label == label
            ])
            for label in self.config.classes
        }

    def _reduce_to_single_mask(self, masks: List[NDArray]) -> NDArray:
        num_masks = len(masks)
        return (
            np.zeros(self.resolution, dtype=bool)
            if num_masks == 0 else
            masks[0]
            if num_masks == 1 else
            reduce(np.logical_or, masks[1:], masks[0])
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[index]
        image = self.transform(Image.open(sample["image_filepath"]))
        masks = torch.stack([BoolTensor(mask) for mask in sample["masks"].values()], dim=0)
        return image, masks

    def summary(self) -> None:
        logger = get_logger()
        logger.info(f"Number of samples: {len(self)}")

        num_pixels_per_class = {
            label: sum(np.count_nonzero(sample["masks"][label]) for sample in self.samples)
            for label in self.config.classes
        }

        total_num_pixels = sum(num_pixels_per_class.values())
        logger.info(f"Total number of annotated pixels: {total_num_pixels}")

        for label, num_pixels in num_pixels_per_class.items():
            percentage = num_pixels / total_num_pixels * 100
            logger.info(f"Number of pixels of class `{label}`: {num_pixels} | {percentage:.2f}%")

        logger.info(f"Mean annotated pixels per sample: {total_num_pixels / len(self):.2f}\n")
