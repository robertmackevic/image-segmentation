from argparse import Namespace
from os import makedirs, listdir
from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.paths import RUNS_DIR
from src.utils import get_available_device, get_logger, save_config


class Trainer:
    def __init__(self, config: Namespace, dataloaders: Tuple[DataLoader, DataLoader]) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger()
        self.train_dl, self.val_dl = dataloaders

        makedirs(RUNS_DIR, exist_ok=True)
        self.model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"
        self.summary_writer_train = SummaryWriter(log_dir=str(self.model_dir / "train"))
        self.summary_writer_eval = SummaryWriter(log_dir=str(self.model_dir / "eval"))
        makedirs(self.summary_writer_train.log_dir, exist_ok=True)
        makedirs(self.summary_writer_eval.log_dir, exist_ok=True)
        save_config(config, self.model_dir / "config.json")
