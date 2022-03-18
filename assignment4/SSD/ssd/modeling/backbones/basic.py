# import torch
# from torch import nn

# class BasicModel(torch.nn.Module):
#     """
#     This is a basic backbone for SSD.
#     The feature extractor outputs a list of 6 feature maps, with the sizes:
#     [shape(-1, output_channels[0], 38, 38),
#      shape(-1, output_channels[1], 19, 19),
#      shape(-1, output_channels[2], 10, 10),
#      shape(-1, output_channels[3], 5, 5),
#      shape(-1, output_channels[3], 3, 3),
#      shape(-1, output_channels[4], 1, 1)]
#      where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
#         self.output_channels = output_channels
#         image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
#         self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
#         #Basics backbone 
# #         self.map1 = nn.Sequential(
# #             nn.Conv2d(image_channels, 32, 3, 1, 1),
# #             nn.MaxPool2d(2,2),
# #             nn.ReLU(),

# #             nn.Conv2d(32, 64, 3, 1, 1),
# #             nn.MaxPool2d(2,2),
# #             nn.ReLU(),
            
# #             nn.Conv2d(64, 64,3, 1, 1),
# #             nn.ReLU(),
            
# #             nn.Conv2d(64, output_channels[0], 3, 2, 1)
# #         )

# #         self.map2 = nn.Sequential(
# #             nn.ReLU(),
# #             nn.Conv2d(output_channels[0], 128, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(128, output_channels[1], 3, 2, 1)
# #         )

# #         self.map3 = nn.Sequential(
# #             nn.ReLU(),
# #             nn.Conv2d(output_channels[1], 256, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(256, output_channels[2], 3, 2, 1)
# #         )

# #         self.map4 = nn.Sequential(
# #             nn.ReLU(),
# #             nn.Conv2d(output_channels[2], 128, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(128, output_channels[3], 3, 2, 1)
# #         )

# #         self.map5 = nn.Sequential(
# #             nn.ReLU(),
# #             nn.Conv2d(output_channels[3], 128, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(128, output_channels[4], 3, 2, 1)
# #         )

# #         self.map6 = nn.Sequential(
# #             nn.ReLU(),
# #             nn.Conv2d(output_channels[4], 128, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(128, output_channels[5], 3, 1, 0)
# #         )


        
#         #Task 4c model (86.41%) 8000 iterations
# #         self.map1 = nn.Sequential(
# #             nn.Conv2d(image_channels, 32, 3, 1, 1),
# #             nn.MaxPool2d(2,2),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(32, 64, 3, 1, 1),
# #             nn.BatchNorm2d(64),
# #             nn.Dropout(p=0.02),
# #             nn.MaxPool2d(2,2),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(64, 64, 3, 1, 1),
# #             nn.BatchNorm2d(64),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(64, 64, 3, 1, 1),
# #             nn.BatchNorm2d(64),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(64, output_channels[0], 3, 2, 1)
# #         )

# #         self.map2 = nn.Sequential(
            
# #             nn.LeakyReLU(),
# #             nn.Conv2d(output_channels[0], 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.Dropout(p=0.02),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, output_channels[1], 3, 2, 1)
# #         )

# #         self.map3 = nn.Sequential(
            
# #             nn.LeakyReLU(),
# #             nn.Conv2d(output_channels[1], 256, 3, 1, 1),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(256, 256, 3, 1, 1),
# #             nn.BatchNorm2d(256),
# #             nn.Dropout(p=0.02),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(256, 256, 3, 1, 1),
# #             nn.BatchNorm2d(256),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(256, 256, 3, 1, 1),
# #             nn.BatchNorm2d(256),
# #             nn.LeakyReLU(),
            
# #             nn.BatchNorm2d(256),
# #             nn.Conv2d(256, output_channels[2], 3, 2, 1)
# #         )

# #         self.map4 = nn.Sequential(
            
# #             nn.LeakyReLU(),
# #             nn.Conv2d(output_channels[2], 128, 3, 1, 1),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.Dropout(p=0.02),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128,128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.BatchNorm2d(128),
# #             nn.Conv2d(128, output_channels[3], 3, 2, 1)
# #         )

# #         self.map5 = nn.Sequential(
            
# #             nn.LeakyReLU(),
# #             nn.Conv2d(output_channels[3], 128, 3, 1, 1),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.Dropout(p=0.01),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.BatchNorm2d(128),
# #             nn.Conv2d(128, output_channels[4], 3, 2, 1)
# #         )

# #         self.map6 = nn.Sequential(
            
# #             nn.LeakyReLU(),
# #             nn.Conv2d(output_channels[4], 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.Dropout(p=0.01),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, 128, 3, 1, 1),
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(),
            
# #             nn.Conv2d(128, output_channels[5], 3, 1, 0)
# #         )
        
        
        
#         #90.2% mAP with 15K iterations, smaller boxes and larger output channels
        
#         self.map1 = nn.Sequential(
#             nn.Conv2d(image_channels, 32, 3, 1, 1),
#             nn.MaxPool2d(2,2),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2,2),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(64, output_channels[0], 3, 2, 1)
#         )

#         self.map2 = nn.Sequential(
            
#             nn.LeakyReLU(),
#             nn.Conv2d(output_channels[0], 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, output_channels[1], 3, 2, 1)
#         )

#         self.map3 = nn.Sequential(
            
#             nn.LeakyReLU(),
#             nn.Conv2d(output_channels[1], 256, 3, 1, 1),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),
            
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, output_channels[2], 3, 2, 1)
#         )

#         self.map4 = nn.Sequential(
            
#             nn.LeakyReLU(),
#             nn.Conv2d(output_channels[2], 128, 3, 1, 1),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(128,128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, output_channels[3], 3, 2, 1)
#         )

#         self.map5 = nn.Sequential(
            
#             nn.LeakyReLU(),
#             nn.Conv2d(output_channels[3], 128, 3, 1, 1),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, output_channels[4], 3, 2, 1)
#         )

#         self.map6 = nn.Sequential(
            
#             nn.LeakyReLU(),
#             nn.Conv2d(output_channels[4], 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
            
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(128, output_channels[5], 3, 1, 0)
#         )

#     def forward(self, x):
#         """
#         The forward functiom should output features with shape:
#             [shape(-1, output_channels[0], 38, 38),
#             shape(-1, output_channels[1], 19, 19),
#             shape(-1, output_channels[2], 10, 10),
#             shape(-1, output_channels[3], 5, 5),
#             shape(-1, output_channels[3], 3, 3),
#             shape(-1, output_channels[4], 1, 1)]
#         We have added assertion tests to check this, iteration through out_features,
#         where out_features[0] should have the shape:
#             shape(-1, output_channels[0], 38, 38),
#         """

#         out_features = []
        
#         x = self.map1(x)
#         out_features.append(x)
#         x = self.map2(x)
#         out_features.append(x)
#         x = self.map3(x)
#         out_features.append(x)
#         x = self.map4(x)
#         out_features.append(x)
#         x = self.map5(x)
#         out_features.append(x)
#         x = self.map6(x)
#         out_features.append(x)

#         for idx, feature in enumerate(out_features):
#             w, h = self.output_feature_shape[idx]
#             expected_shape = (self.output_channels[idx], h, w)
#             assert feature.shape[1:] == expected_shape, \
#                 f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
#         return tuple(out_features)






# vår kode:



import torch
from torch import nn
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.model1 = nn.Sequential(

            # Resolution 38x38
            nn.Conv2d(image_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels[0], 3, 2, 1),
            nn.ReLU()
        )

        self.model2 = nn.Sequential(

            # Resolution 19x19
            nn.ReLU(),
            nn.Conv2d(output_channels[0], 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels[1], 3, 2, 1),
            nn.ReLU()
        )

        self.model3 = nn.Sequential(

            # Resolution 10x10
            nn.ReLU(),
            nn.Conv2d(output_channels[1], 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, output_channels[2], 3, 2, 1),
            nn.ReLU()
        )

        self.model4 = nn.Sequential(

            # Resolution 5x5
            nn.ReLU(),
            nn.Conv2d(output_channels[2], 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels[3], 3, 2, 1),
            nn.ReLU()
        )

        self.model5 = nn.Sequential(

            # Resolution 3x3
            nn.ReLU(),
            nn.Conv2d(output_channels[3], 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels[4], 3, 2, 1),
            nn.ReLU()
        )

        self.model6 = nn.Sequential(

            # Resolution 1x1
            nn.ReLU(),
            nn.Conv2d(output_channels[4], 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels[5], 3, 1, 0),
            nn.ReLU()
        )


    def forward(self, x):
        """
        The forward function should output features with shape:
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
        out_features = []

        x = self.model1(x)
        out_features.append(x)
        x = self.model2(x)
        out_features.append(x)
        x = self.model3(x)
        out_features.append(x)
        x = self.model4(x)
        out_features.append(x)
        x = self.model5(x)
        out_features.append(x)
        x = self.model6(x)
        out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

