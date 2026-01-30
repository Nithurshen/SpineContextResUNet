from model import SpineResUNet_LiDilation
import torch

model = SpineResUNet_LiDilation()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")
