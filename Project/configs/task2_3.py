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

backbone = L(FPN)(resnet_variant="resnet34",
                  pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")