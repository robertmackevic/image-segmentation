import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import torch
from fiftyone import zoo, Dataset as FODataset

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
