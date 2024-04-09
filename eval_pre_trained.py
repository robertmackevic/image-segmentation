from typing import Callable

import torch
from fiftyone import Dataset as FODataset
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_resnet50,
    FCN_ResNet50_Weights,
    DeepLabV3_ResNet50_Weights,
)
from tqdm import tqdm

from src.dataset import COCODataset, COCODataloader
from src.metrics import Metrics
from src.utils import (
    load_config,
    seed_everything,
    download_coco_data,
    get_available_device,
    count_parameters,
)


def load_test_data() -> FODataset:
    config = load_config()
    _data = download_coco_data(config=config, split="train", max_samples=config.num_samples)
    test_start_idx = int(config.num_samples * 0.8) + int(config.num_samples * 0.1)
    return _data[test_start_idx:]


def eval_pre_trained(
        model_init_fn: Callable,
        weights: FCN_ResNet50_Weights | DeepLabV3_ResNet50_Weights,
        test_data: FODataset
) -> None:
    config = load_config()
    seed_everything(config.seed)
    device = get_available_device()

    transform = weights.transforms()
    image_size = transform.resize_size[0]
    transform.resize_size = [image_size, image_size]
    config.image_size = image_size
    label_to_output_idx = {label: idx for idx, label in enumerate(weights.meta["categories"])}

    dataset = COCODataset(test_data, summarize=False, config=config)

    dataset.transform = transform
    dataloader = COCODataloader(dataset, batch_size=16, pin_memory=True)

    model = model_init_fn(weights=weights).to(device)
    print(f"Number of parameters: {count_parameters(model)}")
    model.eval()

    config.classes = ["train", "boat", "aeroplane"]
    metrics = Metrics(config.classes)

    for batch in tqdm(dataloader):
        source = batch[0].to(device)
        target = batch[1].to(device)

        with torch.no_grad():
            output = model(source)["out"].softmax(dim=1)

        for target_label_idx, label in enumerate(config.classes):
            output_label_idx = label_to_output_idx[label]
            metrics.data[label].compute(
                output=(output[:, output_label_idx, ...] > 0.5).float(),
                target=target[:, target_label_idx, ...]
            )

    metrics.compute_class_metrics()
    metrics.compute_total_metrics()

    message = "[overall] "
    for key, value in metrics.total_metrics.items():
        message += f"{key}: {value:.3f} "
    print(message)

    for label in config.classes:
        message = f"[{label}] "
        for key, value in metrics.class_metrics[label].items():
            message += f"{key}: {value:.3f} "
        print(message)


if __name__ == "__main__":
    print("Pytorch version:", torch.__version__)
    print("CUDA enabled:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name())

    data = load_test_data()

    print("\nEvaluating FCN_ResNet50_Weights")
    eval_pre_trained(
        model_init_fn=fcn_resnet50,
        weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        test_data=data
    )
    print("\nEvaluating DeepLabV3_ResNet50_Weights")
    eval_pre_trained(
        model_init_fn=deeplabv3_resnet50,
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        test_data=data
    )
