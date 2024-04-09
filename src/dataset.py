from argparse import Namespace
from functools import reduce
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from fiftyone import Sample, Dataset as FODataset
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.utils import get_logger, compose_transform, load_config


class COCODataset(Dataset):
    def __init__(self, data: FODataset, summarize: bool = True, config: Namespace = load_config()) -> None:
        self.config = config
        self.samples: List[Dict[str, Any]] = []
        self.resolution = (self.config.image_size, self.config.image_size)
        self.transform = compose_transform(self.resolution)
        logger = get_logger()

        # noinspection PyTypeChecker
        for sample in data.select_fields(["filepath", "ground_truth"]):
            self.samples.append({
                "image_filepath": sample.filepath,
                "masks": self._create_semantic_masks(sample)
            })

        num_pixels_per_class = {
            label: sum(np.count_nonzero(sample["masks"][label]) for sample in self.samples)
            for label in self.config.classes
        }

        total_pixels = self.resolution[0] * self.resolution[1] * len(self)
        total_labeled = sum(num_pixels_per_class.values())
        percent_labeled = total_labeled / total_pixels * 100
        total_unlabeled = total_pixels - total_labeled
        percent_unlabeled = total_unlabeled / total_pixels * 100
        weight_coefficient = (len(self.config.classes) + 1)
        unlabeled_weight = total_pixels / (total_unlabeled * weight_coefficient)
        labeled_weight = total_pixels / (total_labeled * weight_coefficient)

        if summarize:
            logger.info(f"Number of samples: {len(self)}")
            logger.info(f"Labeled: {total_labeled} pixels | {percent_labeled:.2f}% | {labeled_weight:.3f} w.")
            logger.info(f"Unlabeled: {total_unlabeled} pixels | {percent_unlabeled:.2f}% | {unlabeled_weight:.3f} w.")
            logger.info(f"Mean labeled pixels per sample: {total_labeled / len(self):.2f}")

        class_weights = []
        for label, num_pixels in num_pixels_per_class.items():
            weight = total_pixels / (num_pixels_per_class[label] * weight_coefficient)
            class_weights.append(weight)

            if summarize:
                percentage = num_pixels / total_labeled * 100
                logger.info(f"Class `{label}`: {num_pixels} pixels | {percentage:.2f}% | {weight:.3f} w.")

        class_weights.append(unlabeled_weight)
        self.class_weights = Tensor(class_weights)

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

        background_mask, masks = torch.ones(self.resolution), []
        for mask in sample["masks"].values():
            masks.append(torch.tensor(mask))
            background_mask[mask == 1] = 0

        masks.append(background_mask)
        masks = torch.stack(masks, dim=0)

        return image, masks


class COCODataloader(DataLoader):
    def __init__(self, dataset: COCODataset, *args, **kwargs) -> None:
        super().__init__(dataset, *args, **kwargs)
        self.class_weights = dataset.class_weights
