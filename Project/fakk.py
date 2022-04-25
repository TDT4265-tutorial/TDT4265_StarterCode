from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision

model = torchvision.models.resnet34(pretrained=True)

feature_extractor = nn.Sequential(*list(model.children())[:-2])

feature_extractor.add_module("layer5", nn.Sequential(
    nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        # Downsample
        nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )
    ),
    nn.Sequential(
        nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512)
    ),
    nn.Sequential(
        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256)
    ),
    nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256)
    ),
    nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256)
    ),
    nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256)
    )
))
feature_extractor.add_module("layer6", nn.Sequential(
    nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        # Downsample
        nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
    ),
    nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128)
    ),
    nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64)
    ),
    nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64)
    ),
    nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64)
    ),
    nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64)
    )
))

print(feature_extractor)
print(len(feature_extractor))

a = feature_extractor(torch.randn(1,3,128,1024))
print(a.shape)