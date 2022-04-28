import torch
from torchvision.models import resnet101
from torchvision.models.resnet import BasicBlock
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from typing import Tuple, List
from torch import nn


# To assist you in designing the feature extractor you may want to print out
# the available nodes for resnet50.
m = resnet101(pretrained=True)

train_nodes, eval_nodes = get_graph_node_names(resnet101(pretrained=True))
print(train_nodes,eval_nodes)
# The lists returned, are the names of all the graph nodes (in order of
# execution) for the input model traced in train mode and in eval mode
# respectively. You'll find that `train_nodes` and `eval_nodes` are the same
# for this example. But if the model contains control flow that's dependent
# on the training mode, they may be different.

# To specify the nodes you want to extract, you could select the final node
# that appears in each of the main layers:
return_nodes = {
    # node_name: user-specified key for output dict
    'layer1.2.relu_2': 'layer1',
    'layer2.3.relu_2': 'layer2',
    'layer3.5.relu_2': 'layer3',
    'layer4.2.relu_2': 'layer4',
    'layer5.2.relu_2': 'layer5',
    'layer6.2.relu_2': 'layer6',
}

# But `create_feature_extractor` can also accept truncated node specifications
# like "layer1", as it will just pick the last node that's a descendent of
# of the specification. (Tip: be careful with this, especially when a layer
# has multiple outputs. It's not always guaranteed that the last operation
# performed is the one that corresponds to the output you desire. You should
# consult the source code for the input model to confirm.)
return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
    'layer5': 'layer5',
    'layer6': 'layer6',
}

# Now you can build the feature extractor. This returns a module whose forward
# method returns a dictionary like:
# {
#     'layer1': output of layer 1,
#     'layer2': output of layer 2,
#     'layer3': output of layer 3,
#     'layer4': output of layer 4,
# }
#create_feature_extractor(m, return_nodes=return_nodes)

# Let's put all that together to wrap resnet50 with MaskRCNN


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
    def __init__(self,
            image_channels: int,
            output_feature_sizes: List[Tuple[int]],
            model_type='resnet34',
            pretrained=True):
        super(Resnet101WithFPN, self).__init__()
        # super(Resnet101WithFPN, self).__init__()

        # Get a resnet50 backbone
        m = resnet101(pretrained=True)



        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

        # Dry run to get number of channels for FPN
        inp = torch.randn(2, image_channels, 224, 224)
        print(inp.shape)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        print(in_channels_list)
        # Build FPN
        self.out_channels = [256]*6
        # self.out_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=256)


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
        print("############################################################")
    def forward(self, x):

        features = []
        x = self.body(x)

        for extra in self.extras:
            x = extra(x)
            features.append(x)

        x = self.fpn(x)

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
        # print(out_features)
        return out_features



# Now we can build our model!
