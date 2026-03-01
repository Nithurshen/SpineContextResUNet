import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETR_Nano(nn.Module):
    """
    SwinUNETR-Nano (Smallest Viable)
    --------------------------------
    Parameter Count: ~3.7M

    Constraint: feature_size MUST be divisible by 12 in MONAI.
    Therefore, feature_size=12 is the absolute minimum.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(SwinUNETR_Nano, self).__init__()

        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=12,
            depths=[1, 1, 1, 1],
            num_heads=[3, 6, 12, 24],
            window_size=(4, 4, 4),
            use_checkpoint=False,
            spatial_dims=3,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward(self, x):
        return self.swin(x)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = SwinUNETR_Nano().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SwinUNETR-Micro Parameters: {total_params / 1e6:.2f}M")

    test_in = torch.randn(1, 1, 128, 128, 64).to(device)
    test_out = model(test_in)
    print(f"Input: {test_in.shape} -> Output: {test_out.shape}")
