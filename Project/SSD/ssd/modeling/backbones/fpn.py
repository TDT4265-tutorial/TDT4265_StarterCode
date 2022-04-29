from queue import PriorityQueue
import torch
from torchvision.models import resnet101
from torchvision.models.resnet import BasicBlock
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from typing import Tuple, List
from torch import nn





class Layer(nn.Sequential):
    def __init__(self,channels,layer_index):
        # [1, 512, 4, 32]
        super().__init__(
            nn.ReLU(),
            #[64, 128, 256, 512, 64, 64],
            #i= 512, o=64
            #i=64, o=64
            # next
            # i=64, o=64
            # i=64, o=64
            nn.Conv2d(in_channels=channels[layer_index-1], out_channels=channels[layer_index], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[layer_index], out_channels=channels[layer_index], kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )

# MaskRCNN requires a backbone with an attached FPN
class Resnet101WithFPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # super(Resnet101WithFPN, self).__init__()

        self.out_channels = [256, 256, 256, 2048, 64, 64]
        
        m = resnet101(pretrained=True)

        
        self.extras = nn.ModuleList([
            torch.nn.Sequential(
                BasicBlock (inplanes = self.out_channels[-3], planes = self.out_channels[-2], stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.out_channels[-3], out_channels=self.out_channels[-2], kernel_size=1, stride=2),
                    nn.ReLU())),
                )
            ,
            torch.nn.Sequential(
                BasicBlock (inplanes = self.out_channels[-2], planes = self.out_channels[-1], stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.out_channels[-2], out_channels=self.out_channels[-1], kernel_size=1, stride=2),
                    nn.ReLU())),
                )
        ])



       
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

        
        inp = torch.randn(1, 3, 128, 1024)

        with torch.no_grad():
            out = self.body(inp)
            # print(out)
            x = out["3"]

            in_channels_list = [o.shape[1] for o in out.values()]

        # print(self.extras[0](x))

        with torch.no_grad():
            for i, extra in enumerate(self.extras):
                #print(i)
                x = extra(x)
                #print(x.shape)
                in_channels_list.append(x.shape[1])


        print("---------------------------------------", in_channels_list, "---------------------------------------")

        # Build FPN
        # self.out_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=256)

        self.out_channels = [256, 256, 256,256, 256, 256]
        print("############################################################")
    def forward(self, x):

        features = []
        x = self.body(x)
        #print(x["3"].shape)
        
        #print("\n finishe body \n")

        for i,extra in enumerate(self.extras):
            
            x[f"{i+4}"] = extra(x[f"{i+3}"])
            
            #features.append(x)
        #print("\n finishe extras \n")
        
        x = self.fpn(x)
        #for i in range(6):
           # print(x[f"{i}"].shape)
        
        features.extend(x.values() )


        # for idx, feature in enumerate(features):
        #     out_channel = self.out_channels[idx]
        #     print("out_channel: ", out_channel, "\n")
        #     print("feature: ", feature, "\n")
            # h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     assert feature.items().shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.items().shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #     f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        out_features = tuple(features)
        #print(out_features)
        return tuple(x.values())



# Now we can build our model!
