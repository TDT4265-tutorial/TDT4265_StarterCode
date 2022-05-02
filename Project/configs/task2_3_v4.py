from math import gamma
from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes, RetinaNet
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    # backbone,
    # model,
    # data_train,
    # data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    # anchors,
    # loss_objective,
)

from .task2_2 import (
    data_train, 
    data_val
)

from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN

backbone = L(FPN)(pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,  # Add 1 for background
    anchor_prob_initialization=True
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio.
    # All feature maps must have same aspect ratio in order to make task 2.3.3 work
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

