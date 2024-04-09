import json
import logging
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from fiftyone import zoo, Dataset as FODataset
from torch.nn import Module
from torchvision.transforms import Compose, Resize, ToTensor

from src.paths import CONFIG_FILE


def load_config() -> Namespace:
    with open(CONFIG_FILE, "r") as config:
        return Namespace(**json.load(config))


def save_config(config: Namespace, filepath: Path) -> None:
    with open(filepath, "w") as file:
        json.dump(vars(config), file, indent=4)


def download_coco_data(config: Namespace, split: str, **kwargs) -> FODataset:
    return zoo.load_zoo_dataset(
        name="coco-2017",
        label_types=["segmentations"],
        split=split,
        classes=config.classes,
        seed=config.seed,
        **kwargs
    )


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger()
    return logger


def get_available_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: Module) -> int:
    return sum(p.numel() for p in module.parameters())


def compose_transform(resolution: Tuple[int, int]) -> Compose:
    return Compose([
        lambda image: image.convert("RGB"),
        Resize(resolution),
        ToTensor()
    ])


def load_weights(filepath: Path, model: Module) -> Module:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    return model


def save_weights(filepath: Path, model: Module) -> None:
    torch.save(model.state_dict(), filepath)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setRNGSeed(seed)
