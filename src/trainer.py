from os import makedirs, listdir
from statistics import mean
from typing import Tuple, List, Optional

import torch
from torch import no_grad
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.dataset import COCODataloader
from src.metrics import Metrics
from src.paths import RUNS_DIR
from src.utils import (
    get_available_device,
    get_logger,
    load_config,
    save_config,
    count_parameters,
    save_weights
)


class Trainer:
    def __init__(self, model: Module, dataloaders: Tuple[COCODataloader, COCODataloader]) -> None:
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
        self.loss_fn = CrossEntropyLoss(weight=self.train_dl.class_weights.to(self.device))

        self.logger.info(f"Number of trainable parameters: {count_parameters(self.model)}")

    def fit(self) -> Module:
        best_score = 0
        best_score_metric = "IoU"

        for epoch in range(1, self.config.epochs + 1):
            losses = self._train_for_epoch()
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            self.log_losses(losses, phase="train", epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                losses, metrics = self.eval(self.val_dl)
                self.log_losses(losses, phase="eval", epoch=epoch)
                self.log_metrics(metrics, epoch=epoch)

                score = metrics.total_metrics[best_score_metric]

                if score > best_score:
                    best_score = score
                    self.logger.info(f"Saving best weights with {best_score_metric}: {score:.3f}")
                    save_weights(self.model_dir / "weights_best.pth", self.model)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(self.model_dir / f"weights_{epoch}.pth", self.model)

        return self.model

    def eval(self, dataloader: COCODataloader) -> Tuple[List[float], Metrics]:
        self.model.eval()
        losses = []
        metrics = Metrics(self.config.classes)

        for batch in tqdm(dataloader):
            source = batch[0].to(self.device)
            target = batch[1].to(self.device)

            with no_grad():
                output = self.model(source)
                loss = self.loss_fn(output, target)
                losses.append(loss.item())

            predicted_labels = output.argmax(dim=1, keepdim=False)
            for label_idx, label in enumerate(self.config.classes):
                metrics.data[label].compute(
                    output=torch.eq(predicted_labels, label_idx).float(),
                    target=target[:, label_idx, ...]
                )

        return losses, metrics

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

    def log_metrics(self, metrics: Metrics, epoch: Optional[int] = None) -> None:
        metrics.compute_class_metrics()
        metrics.compute_total_metrics()

        if epoch is not None:
            for key, value in metrics.total_metrics.items():
                self.summary_writer_eval.add_scalar(tag=key, scalar_value=value, global_step=epoch)

        message = "[overall] "
        for key, value in metrics.total_metrics.items():
            message += f"{key}: {value:.3f} "
        self.logger.info(message)

        for label in self.config.classes:
            message = f"[{label}] "
            for key, value in metrics.class_metrics[label].items():
                message += f"{key}: {value:.3f} "
            self.logger.info(message)
