import torch.nn as nn
import torch
import torch.nn.functional as F
from tops import to_cuda


class SSDFocalLoss(nn.Module):
    """
        Implements the loss as Focal loss from the paper
        Focal loss for dense object detection.
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, gamma):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        
        self.gamma = gamma


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        batch_size, num_classes, num_anchors = confs.shape
        alpha = to_cuda(torch.ones(num_classes)*1000)
        alpha[0] = 10
        alpha = alpha.view(1, -1, 1)
        
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        
        p_k = F.softmax(confs, dim=1)
        log_p_k = F.log_softmax(confs, dim=1)
        y_k = F.one_hot(gt_labels, num_classes).transpose(1, 2)

        # print("alpha:  ", alpha.shape)
        # print("p_k:    ", p_k.shape)
        # print("log_p_k:", log_p_k.shape)
        # print("y_k:    ", y_k.shape)

        weight = torch.pow(1. - p_k, self.gamma)
        focal = - y_k * weight * log_p_k
        loss_tmp = torch.sum(alpha * focal, dim=1)
        focal_loss = torch.mean(loss_tmp)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + focal_loss
        # print("Regression Loss:", regression_loss/num_pos)
        # print("Focal Loss:     ", focal_loss)
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=focal_loss,
            total_loss=total_loss
        )
        return total_loss, to_log
