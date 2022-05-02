import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class RetinaNet(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            anchor_prob_initialization: bool = True):
        super().__init__()
        """
            Implements the RetinaNet network, based on SSD.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.anchor_prob_initialization = anchor_prob_initialization
        
        # Notation from "Focal Loss for Dense Object Detection"
        self.A = anchors.num_boxes_per_fmap[0]           # Num anchors at each feature map
        self.K = self.num_classes                        # Number of classes
        self.C = self.feature_extractor.fpn_out_channels # Number of channels per feature map

        # Initialize output heads that are applied to each feature map from the backbone.
        self.classification_heads = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.K*self.A, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.regression_heads = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, 4*self.A, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    def _init_weights(self):
        if self.anchor_prob_initialization:
            layers = [*self.regression_heads, *self.classification_heads]
            for layer in layers:
                for param in layer.parameters():
                    # Sorting out the weights
                    if param.dim() > 1: 
                        nn.init.normal_(param, 0, 0.01)
                    # Sorting out the biases
                    else:
                        nn.init.zeros_(param)

            # Extracting last layer of classification heads
            module_children = list(self.classification_heads.children())
            # Extracting the last convolutional layer
            conv_layer = list(module_children[-2].named_parameters())

            bias = conv_layer[1]
            biasArray = torch.zeros(self.K)
            p = 0.99
            b = torch.log(torch.tensor(p*((self.K-1)/(1-p))))
            biasArray[0] = b
            biasArray = biasArray.repeat(self.A)
            bias[1].data = biasArray
        else:
            layers = [self.regression_heads, self.classification_heads]
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for x in features:
            bbox_delta = self.regression_heads(x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads(x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences
    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]