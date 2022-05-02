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
                 pretrained: bool,
                 fpn_out_channels: int,
                 output_feature_sizes: List[List[int]]):
        super().__init__()
        
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels for i in range(6)]
        self.output_feature_shape = output_feature_sizes
        
        self.resnet_out_channels = [64, 128, 256, 512, 1024, 2048]
        # Get a pretrained ResNet34 model
        self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())[:-2])

        self.feature_extractor.add_module("layer5", nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),    # Downsample
            nn.ReLU()
        ))
        self.feature_extractor.add_module("layer6", nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),    # Downsample
            nn.ReLU()
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
        
        for idx, feature in enumerate(out_features):
            h, w = self.output_feature_shape[idx]
            expected_shape = (self.fpn_out_channels, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

