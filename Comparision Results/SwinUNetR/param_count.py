from model import SwinUNETR_Nano
import torch

# Initialize the model
model = SwinUNETR_Nano()

# Calculate parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"--- SwinUNETR Micro Summary ---")
print(f"Total Params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")

# Optional: Check size in MB (float32 = 4 bytes)
size_mb = total_params * 4 / (1024 * 1024)
print(f"Model Size:   {size_mb:.2f} MB")