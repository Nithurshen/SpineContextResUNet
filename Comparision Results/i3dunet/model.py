import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard Plain Block: (Conv3d => BN => ReLU) * 2
    No residual connections.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Plain3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=9):
        super(Plain3DUNet, self).__init__()

        # --- Encoder ---
        self.inc = DoubleConv(in_channels, base_filters)
        
        self.pool1 = nn.MaxPool3d(2)
        self.down1 = DoubleConv(base_filters, base_filters * 2)
        
        self.pool2 = nn.MaxPool3d(2)
        self.down2 = DoubleConv(base_filters * 2, base_filters * 4)
        
        self.pool3 = nn.MaxPool3d(2)
        self.down3 = DoubleConv(base_filters * 4, base_filters * 8)

        # --- Bottleneck ---
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)

        # --- Decoder ---
        self.up1 = nn.ConvTranspose3d(
            base_filters * 16, base_filters * 8, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_filters * 16, base_filters * 8)
        
        self.up2 = nn.ConvTranspose3d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.up3 = nn.ConvTranspose3d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.up4 = nn.ConvTranspose3d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_filters * 2, base_filters)

        # --- Final ---
        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        
        p1 = self.pool1(x1)
        x2 = self.down1(p1)
        
        p2 = self.pool2(x2)
        x3 = self.down2(p2)
        
        p3 = self.pool3(x3)
        x4 = self.down3(p3)

        p4 = self.pool4(x4)
        b = self.bottleneck(p4)

        # Decoder (with interpolation for shape mismatch)
        d1 = self.up1(b)
        if d1.shape != x4.shape:
            d1 = F.interpolate(d1, size=x4.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x4], dim=1))
        
        d2 = self.up2(d1)
        if d2.shape != x3.shape:
            d2 = F.interpolate(d2, size=x3.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x3], dim=1))
        
        d3 = self.up3(d2)
        if d3.shape != x2.shape:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        
        d4 = self.up4(d3)
        if d4.shape != x1.shape:
            d4 = F.interpolate(d4, size=x1.shape[2:], mode="trilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, x1], dim=1))

        return self.sigmoid(self.final(d4))