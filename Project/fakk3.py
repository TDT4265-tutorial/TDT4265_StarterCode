from cv2 import log
from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision

# Notation from "Focal Loss for Dense Object Detection"
A = 6       # Num anchors at each feature map
K = 9       # Number of classes
C = 256     # Number of channels per feature map

a = nn.ModuleList([nn.Conv2d(C, C, kernel_size=3, padding=1), nn.Conv2d(C, C, kernel_size=3, padding=1)])
print(a)
print(a[:])
print(*a[:].bias)

exit()

classification_heads = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, K*A, kernel_size=3, padding=1),
    nn.Sigmoid()
)

regression_heads = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, 4*A, kernel_size=3, padding=1),
    nn.Sigmoid()
)

layers = [regression_heads, classification_heads]


module_children = list(classification_heads.children())
print(module_children)
named_params = list(module_children[-2].named_parameters())

print(named_params)
bias = named_params[1]
print("Bias pre")
print(bias)
print(type(bias))
nn.init.zeros_(bias[1])
print("Eivin")
bias[1][0] = 1
print(bias[1])
bias[1][0:-1:A] = 1

print("Bias post")
print(bias)
print(len(bias[1]))

for layer in layers:
    for param in layer.parameters():
        if param.dim() > 1: nn.init.xavier_uniform_(param)