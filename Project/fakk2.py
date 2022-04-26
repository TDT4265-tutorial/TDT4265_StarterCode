from cv2 import log
from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision

n = 65520
b = 32
c = 9

pk = torch.rand((b, c, n))
log_pk = torch.rand((b, c, n))
yk = torch.rand((b, c, n))
alpha = torch.rand((1,9))
alpha = alpha.view(1, -1, 1)


gamma = 2

print("Shapes:")
print("pk     ", pk.shape)
print("log_pk ", log_pk.shape)
print("yk     ", yk.shape)
print("alpha  ", alpha.shape)

print("MAGI:")
focal_loss = - alpha @ ((1-pk) ** gamma).transpose(1,2) @ yk @ log_pk

print()
print("Test 1")

t = alpha @ ((1-pk) ** gamma)
print("alpha @ ((1-pk) ** gamma):", (alpha @ ((1-pk) ** gamma)).shape)
print("t:  ", t.shape)
#torch.outer(alpha, (1-pk) ** gamma)

print()
print("Test 2")
#t2 = t.transpose(0,2)@ yk
yk2 = yk.transpose(1,2)
print("yk2:   ", yk2.shape)
t2 = t @ yk2
print("alpha @ ((1-pk) ** gamma) @ yk", (t @ yk2).shape)

print()
print("Test 3")
focal_loss = t2 @ log_pk
print("focal_loss:", focal_loss.shape)