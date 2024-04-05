from os import makedirs, listdir
from statistics import mean
from typing import Tuple, List, Dict, Optional

from torch import no_grad, Tensor, logical_and, logical_or
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.paths import RUNS_DIR
from src.utils import (
    get_available_device,
    get_logger,
    load_config,
    save_config,
    count_layers,
    count_parameters,
    save_weights
)


class Trainer:
    def __init__(self, model: Module, dataloaders: Tuple[DataLoader, DataLoader]) -> None:
        self.config = load_config()
        self.device = get_available_device()
        self.logger = get_logger()
        self.train_dl, self.val_dl = dataloaders

        makedirs(RUNS_DIR, exist_ok=True)
        self.model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"
        self.summary_writer_train = SummaryWriter(log_dir=str(self.model_dir / "train"))
        self.summary_writer_eval = SummaryWriter(log_dir=str(self.model_dir / "eval"))
        makedirs(self.summary_writer_train.log_dir, exist_ok=True)
        makedirs(self.summary_writer_eval.log_dir, exist_ok=True)
        save_config(self.config, self.model_dir / "config.json")

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = CrossEntropyLoss()

        self.logger.info(f"Number of trainable parameters: {count_parameters(self.model)}")
        self.logger.info(f"Number of layers: {count_layers(self.model)}")

    def fit(self) -> Module:
        best_fit = 0

        for epoch in range(1, self.config.epochs + 1):
            losses = self._train_for_epoch()
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            self.log_losses(losses, phase="train", epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                losses, iou, m_iou = self.eval(self.val_dl)
                self.log_losses(losses, phase="eval", epoch=epoch)
                self.log_iou(iou, m_iou, epoch=epoch)

                if m_iou > best_fit:
                    best_fit = m_iou
                    self.logger.info(f"Saving best weights with mIoU: {m_iou:.3f}")
                    save_weights(self.model_dir / "weights_best.pth", self.model)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(self.model_dir / f"weights_{epoch}.pth", self.model)

        self.logger.info("Saving final model weights")
        save_weights(self.model_dir / "weights_final.pth", self.model)

        return self.model

    def eval(self, dataloader: DataLoader) -> Tuple[List[float], Dict[str, float], float]:
        self.model.eval()
        losses = []
        iou = {label: [] for label in self.config.classes}

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()

            source = batch[0].to(self.device)
            target = batch[1].to(self.device)

            with no_grad():
                output = self.model(source)
                loss = self.loss_fn(output, target)
                losses.append(loss.item())

            for idx, label in enumerate(self.config.classes):
                iou[label].append(self.compute_iou(output[:, idx, ...], target[:, idx, ...]))

        iou = {label: mean(values) for label, values in iou.items()}
        m_iou = mean(iou.values())
        return losses, iou, m_iou

    def _train_for_epoch(self) -> List[float]:
        self.model.train()
        losses = []

        for batch in tqdm(self.train_dl):
            self.optimizer.zero_grad()

            source = batch[0].to(self.device)
            target = batch[1].to(self.device)
            output = self.model(source)

            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return losses

    def log_losses(self, losses: List[float], phase: str, epoch: Optional[int] = None) -> None:
        loss = mean(losses)

        if epoch is not None:
            summary_writer = self.summary_writer_train if phase == "train" else self.summary_writer_eval
            summary_writer.add_scalar(tag="loss", scalar_value=loss, global_step=epoch)

        self.logger.info(f"[{phase}] loss: {loss:.3f}")

    def log_iou(self, iou: Dict[str, float], m_iou: float, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.summary_writer_eval.add_scalar(tag="mIoU", scalar_value=m_iou, global_step=epoch)

        self.logger.info(f"mIoU: {m_iou:.3f}")

        for label, value in iou.items():
            self.logger.info(f"\tIoU {label}: {value:.3f}")

    def compute_iou(self, output: Tensor, target: Tensor) -> float:
        output = (output > self.config.conf).float()
        intersection = logical_and(output, target).sum()
        union = logical_or(output, target).sum()
        iou = intersection.float() / (union.float() + 1e-8)
        return iou.item()
