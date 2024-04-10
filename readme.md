# Dataset
COCO-2017
* 5000 images.
* 3 classes:
  * train
  * boat
  * airplane
* Dataset split 80-10-10
# Trained Model Evaluation
## U-Net
* Original paper: https://arxiv.org/abs/1505.04597
* Parameters: 31,037,828

TRAIN DATA
```
[overall]   Precision: 0.821 Recall: 0.986 F1: 0.895 IoU: 0.811
[train]     Precision: 0.902 Recall: 0.987 F1: 0.943 IoU: 0.892
[boat]      Precision: 0.757 Recall: 0.989 F1: 0.858 IoU: 0.751
[airplane]  Precision: 0.803 Recall: 0.982 F1: 0.884 IoU: 0.791
```
VALIDATION DATA
```
[overall]   Precision: 0.670 Recall: 0.660 F1: 0.665 IoU: 0.503
[train]     Precision: 0.734 Recall: 0.730 F1: 0.732 IoU: 0.577
[boat]      Precision: 0.568 Recall: 0.534 F1: 0.550 IoU: 0.380
[airplane]  Precision: 0.708 Recall: 0.716 F1: 0.712 IoU: 0.553
```
TEST DATA
```
[overall]   Precision: 0.656 Recall: 0.588 F1: 0.619 IoU: 0.460
[train]     Precision: 0.772 Recall: 0.694 F1: 0.731 IoU: 0.576
[boat]      Precision: 0.498 Recall: 0.403 F1: 0.445 IoU: 0.287
[airplane]  Precision: 0.697 Recall: 0.667 F1: 0.681 IoU: 0.517
```
## SetNet
* Original paper: https://arxiv.org/abs/1511.00561
* Parameters: 24,864,608

TRAIN DATA
```
[overall]   Precision: 0.655 Recall: 0.972 F1: 0.780 IoU: 0.643 
[train]     Precision: 0.749 Recall: 0.969 F1: 0.845 IoU: 0.731 
[boat]      Precision: 0.572 Recall: 0.967 F1: 0.718 IoU: 0.561 
[airplane]  Precision: 0.644 Recall: 0.981 F1: 0.778 IoU: 0.636
```
VALIDATION DATA
```
[overall]   Precision: 0.466 Recall: 0.572 F1: 0.511 IoU: 0.351 
[train]     Precision: 0.531 Recall: 0.723 F1: 0.612 IoU: 0.441 
[boat]      Precision: 0.351 Recall: 0.336 F1: 0.343 IoU: 0.207 
[airplane]  Precision: 0.516 Recall: 0.655 F1: 0.577 IoU: 0.406
```
TEST DATA
```
[overall]   Precision: 0.458 Recall: 0.541 F1: 0.495 IoU: 0.341 
[train]     Precision: 0.597 Recall: 0.669 F1: 0.631 IoU: 0.461 
[boat]      Precision: 0.277 Recall: 0.315 F1: 0.295 IoU: 0.173 
[airplane]  Precision: 0.498 Recall: 0.639 F1: 0.560 IoU: 0.389
```
# Comparison with Pre-Trained Models
## FCN with ResNet-50 Backbone
* Link: https://pytorch.org/vision/stable/models/fcn.html
* Parameters: 35,322,218

TEST DATA
```
[overall]   Precision: 0.922 Recall: 0.771 F1: 0.837 IoU: 0.724
[train]     Precision: 0.950 Recall: 0.778 F1: 0.855 IoU: 0.747
[boat]      Precision: 0.894 Recall: 0.660 F1: 0.759 IoU: 0.612
[airplane]  Precision: 0.921 Recall: 0.874 F1: 0.897 IoU: 0.813
```
## DeepLabV3 with ResNet-50 Backbone
* Link: https://pytorch.org/vision/stable/models/deeplabv3.html
* Parameters: 42,004,074

TEST DATA
```
[overall]   Precision: 0.898 Recall: 0.881 F1: 0.889 IoU: 0.803
[train]     Precision: 0.931 Recall: 0.930 F1: 0.931 IoU: 0.871
[boat]      Precision: 0.861 Recall: 0.798 F1: 0.828 IoU: 0.707
[airplane]  Precision: 0.901 Recall: 0.914 F1: 0.907 IoU: 0.830
```