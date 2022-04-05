from ssd.modeling import backbones
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
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

backbone = L(FPN)(type="resnet34",
                  pretrained=True,
                  output_channels=[64, 128, 256, 512, 256, 64],
                  image_channels="${train.image_channels}",
                  output_feature_sizes="${anchors.feature_sizes}")