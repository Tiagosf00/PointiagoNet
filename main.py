from pointnet import ClassificationModule
import torch

x = torch.randn(32, 3, 100)
classification_pointnet = ClassificationModule(100, 10, 2)
x, idx = classification_pointnet.forward(x)

print("2 classes for each of the 32 samples")
print(x)

print("10 critical indexes for each of the 32 samples")
print(idx)
