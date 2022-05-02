from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

num_classes = 9

gamma = 2

alpha = torch.ones(num_classes)*1000
alpha[0] = 10
alpha = alpha.view(1, -1, 1)

p_k = torch.rand((32, 9, 65536))
log_p_k = torch.rand((32, 9, 65536))
y_k = torch.rand((32, 9, 65536))

print("== SHAPES ==")
print("alpha:   ", alpha.shape)
print("p_k:     ", p_k.shape)
print("log_p_k: ", log_p_k.shape)
print("y_k:     ", y_k.shape)

weight = torch.pow(1. - p_k, gamma)

print("weight:  ", weight.shape)
print("a*w: ", (alpha*weight).shape)
focal = -alpha * weight * log_p_k
loss_tmp = torch.sum(y_k * focal, dim=1)
focal_loss = torch.mean(loss_tmp)

focal_loss2 = (y_k*focal).sum(dim=1).mean()
print("focal_loss: ", focal_loss)
print("focal_loss2:", focal_loss2)

print("========== ZAIM ==========")
alpha = torch.ones(num_classes)*1000
alpha[0] = 10

weight_balanced = torch.pow(1.0 - p_k, gamma)
focal = -alpha.unsqueeze(0).unsqueeze(-1) * weight_balanced
classification_loss = focal * y_k * log_p_k
print("classification_loss.shape:", classification_loss.shape)
classification_loss = classification_loss.sum(dim=1).mean()

print(classification_loss)