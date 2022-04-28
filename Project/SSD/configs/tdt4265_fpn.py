from ssd.modeling import backbones
from .tdt4265_2_2 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    # backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
import torchvision

from tops.config import LazyCall as L
from ssd.modeling.backbones import Resnet101WithFPN

# backbone = L(Resnet101WithFPN)()
backbone = L(Resnet101WithFPN)()