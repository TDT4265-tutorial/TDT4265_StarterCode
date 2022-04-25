import torch
import torch.nn as nn
import torchvision
from typing import OrderedDict, Tuple, List


class FPN(nn.Module):
    """
    This is a basic backbone for RetinaNet - Feature Pyramid network based on ResNet.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
                 resnet_variant: str,
                 pretrained: bool,
                 fpn_out_channels: int,
                 output_feature_sizes: List[List[int]]):
        super().__init__()
        
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels for i in range(6)]
        self.output_feature_shape = output_feature_sizes
        
        if resnet_variant == 'resnet18':
            self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet18(pretrained=pretrained).children())[:-2])
        elif resnet_variant == 'resnet34':
            self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())[:-2])
        elif resnet_variant == 'resnet50':
            self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet50(pretrained=pretrained).children())[:-2])
        elif resnet_variant == 'resnet101':
            self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet101(pretrained=pretrained).children())[:-2])
        elif resnet_variant == 'resnet152':
            self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet152(pretrained=pretrained).children())[:-2])
        else:
            raise ValueError(f'Unsupport resnet_resnet_variant "{resnet_variant}".')
        
        self.resnet_out_channels = [64, 128, 256, 512, 256, 64] if resnet_variant[-2:] in ['18', '34'] else [256, 512, 1024, 2048, 1024, 256]
        
        self.feature_extractor.add_module("layer5", nn.Sequential(
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-3], self.resnet_out_channels[-3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-3], self.resnet_out_channels[-3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-3]),
                # Downsample
                nn.Sequential(
                    nn.Conv2d(self.resnet_out_channels[-3], self.resnet_out_channels[-3]*2, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(self.resnet_out_channels[-3]*2),
                )
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-3]*2, self.resnet_out_channels[-3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-3], self.resnet_out_channels[-3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-3])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-3], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-2])
            )
        ))
        self.feature_extractor.add_module("layer6", nn.Sequential(
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-2], self.resnet_out_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]*2),
                # Downsample
                nn.Sequential(
                    nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1]*2, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(self.resnet_out_channels[-1]*2),
                )
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]*2)
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1]*2, self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1])
            ),
            nn.Sequential(
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.resnet_out_channels[-1], self.resnet_out_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.resnet_out_channels[-1])
            )
        ))
        
        self.fpn = torchvision.ops.FeaturePyramidNetwork(self.resnet_out_channels, self.fpn_out_channels)
    
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        #out_features = []
        
        # Ignore four first "layers"/operations
        for i in range(4):
            x = self.feature_extractor[i](x)
        
        pyramid = OrderedDict()
        
        # Pass x through the resnet
        pyramid['feat0'] = self.feature_extractor[4](x)
        pyramid['feat1'] = self.feature_extractor[5](pyramid['feat0'])
        pyramid['feat2'] = self.feature_extractor[6](pyramid['feat1'])
        pyramid['feat3'] = self.feature_extractor[7](pyramid['feat2'])
        pyramid['feat4'] = self.feature_extractor[8](pyramid['feat3'])
        pyramid['feat5'] = self.feature_extractor[9](pyramid['feat4'])
        
        out_features = self.fpn(pyramid).values()
        
        #feature_thingy_2(out_features)
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (self.fpn_out_channels, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

#    def forward(self, x):
#        """
#        The forward functiom should output features with shape:
#            [shape(-1, output_channels[0], 38, 38),
#            shape(-1, output_channels[1], 19, 19),
#            shape(-1, output_channels[2], 10, 10),
#            shape(-1, output_channels[3], 5, 5),
#            shape(-1, output_channels[3], 3, 3),
#            shape(-1, output_channels[4], 1, 1)]
#        We have added assertion tests to check this, iteration through out_features,
#        where out_features[0] should have the shape:
#            shape(-1, output_channels[0], 38, 38),
#        """
#        out_features = []
#        for i in range(5):
#            x = self.feature_extractor[i](x)
#        
#        out_features.append(self.feature_extractor[4](x))
#
#        for i in range(5, len(self.feature_extractor)):
#            out_features.append(self.feature_extractor[i](out_features[i-5]))
#
#        for idx, feature in enumerate(out_features):
#            out_channel = self.out_channels[idx]
#            h, w = self.output_feature_shape[idx]
#            expected_shape = (out_channel, h, w)
#            assert feature.shape[1:] == expected_shape, \
#                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
#        assert len(out_features) == len(self.output_feature_shape),\
#            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
#        return tuple(out_features)
