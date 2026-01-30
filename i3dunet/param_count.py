from model import Plain3DUNet
import torch

# Base filters = 18 is tuned to get ~1.7M parameters with this depth
model = Plain3DUNet(base_filters=18)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: Plain3DUNet (base=18)")
print(f"Total Params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")