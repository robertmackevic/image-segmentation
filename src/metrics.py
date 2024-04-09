from dataclasses import dataclass
from statistics import mean
from typing import List, Tuple

from torch import Tensor, logical_not, logical_and


@dataclass(frozen=False)
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def compute(self, output: Tensor, target: Tensor) -> None:
        n_output, n_target = logical_not(output), logical_not(target)
        self.tp += logical_and(output, target).sum().item()
        self.tn += logical_and(n_output, n_target).sum().item()
        self.fp += logical_and(output, n_target).sum().item()
        self.fn += logical_and(n_output, target).sum().item()

    def get(self) -> Tuple[int, int, int, int]:
        return self.tp, self.tn, self.fp, self.fn


class Metrics:
    def __init__(self, classes: List[str]) -> None:
        self.metrics = ["Accuracy", "Precision", "Recall", "F1", "IoU"]
        self.data = {label: ConfusionMatrix() for label in classes}
        self.class_metrics = {label: {metric: 0.0 for metric in self.metrics} for label in classes}
        self.total_metrics = {metric: 0.0 for metric in self.metrics}

    def compute_class_metrics(self) -> None:
        for label in self.class_metrics:
            tp, tn, fp, fn = self.data[label].get()

            accuracy = (tp + tn) / (tp + fn + fp + tn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fn + fp + 1e-8)

            self.class_metrics[label]["Accuracy"] = accuracy
            self.class_metrics[label]["Precision"] = precision
            self.class_metrics[label]["Recall"] = recall
            self.class_metrics[label]["F1"] = f1
            self.class_metrics[label]["IoU"] = iou

    def compute_total_metrics(self) -> None:
        for metric in self.metrics:
            self.total_metrics[metric] = mean(value[metric] for value in self.class_metrics.values())
