from model import SpineResUNet_cotext_off
import torch

model = SpineResUNet_cotext_off()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")
