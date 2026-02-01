from model import Plain3DUNet
import torch

# Base filters = 9 gives approximately 1.79M parameters
model = Plain3DUNet(base_filters=9)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: Plain3DUNet (base=9)")
print(f"Total Params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")