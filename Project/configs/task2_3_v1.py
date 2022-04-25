from math import gamma
from ssd.modeling import backbones, SSDFocalLoss
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    model,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN

backbone = L(FPN)(resnet_variant="resnet18",
                  pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)

""" This code has just the FPN implementation. Nothing is done wrt loss function or optimizer.  There is also no data augmentation."""