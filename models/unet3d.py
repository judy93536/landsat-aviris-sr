"""
3D U-Net for Joint Spatial-Spectral Super-Resolution

Directly learns from 7 Landsat bands → 198 AVIRIS bands
using 3D convolutions that operate on (H, W, C) jointly.

This captures spatial-spectral correlations that the 2-stage approach misses.

Key advantages:
- End-to-end optimization
- Learns coupled spatial-spectral features
- Single-pass inference

Memory consideration:
- 3D convolutions are memory-intensive
- Use smaller batch sizes (1-2)
- Consider smaller patch sizes or fewer channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Encoder block: 2x Conv3D + MaxPool (spatial only)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
        # Pool only spatially (H, W), not spectrally (C)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block: Upsample + Concat + 2x Conv3D (spatial only)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample only spatially (H, W), not spectrally (C)
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                     kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.conv1 = Conv3DBlock(in_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for joint spatial-spectral super-resolution.

    Architecture:
    - Encoder: Progressively downsample spatial dimensions while increasing channels
    - Bottleneck: Deepest representation
    - Decoder: Upsample spatial dimensions while decreasing channels
    - Skip connections: Preserve spatial-spectral details

    Input shape: (B, 1, H, W, 7) - treating bands as depth dimension
    Output shape: (B, 1, H, W, 198)
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=198,
        base_features=32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Treat input as (B, 1, H, W, C) where C is spectral dimension
        # This allows 3D conv to operate on spatial (H,W) and spectral (C) jointly

        # Initial conv (no pooling)
        self.init_conv = Conv3DBlock(1, base_features, kernel_size=3, padding=1)

        # Encoder (downsampling spatial only, not spectral yet)
        self.down1 = DownBlock(base_features, base_features * 2)
        self.down2 = DownBlock(base_features * 2, base_features * 4)
        self.down3 = DownBlock(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv3DBlock(base_features * 8, base_features * 16),
            Conv3DBlock(base_features * 16, base_features * 16)
        )

        # Decoder (upsampling spatial)
        self.up1 = UpBlock(base_features * 16, base_features * 8)
        self.up2 = UpBlock(base_features * 8, base_features * 4)
        self.up3 = UpBlock(base_features * 4, base_features * 2)

        # Final conv to get desired output bands
        self.final_conv = nn.Conv3d(base_features * 2, 1, kernel_size=1)

        # Spectral expansion layer
        # After spatial processing, expand from 7 to 198 bands
        self.spectral_expand = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (B, C_in, H, W) with C_in=7

        Returns:
        --------
        out : torch.Tensor
            Output tensor (B, C_out, H, W) with C_out=198
        """
        B, C_in, H, W = x.shape

        # Reshape to 3D: (B, 1, H, W, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, 7)

        # U-Net encoder
        x = self.init_conv(x)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # U-Net decoder
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)

        # Final conv
        x = self.final_conv(x)  # (B, 1, H, W, 7)

        # Remove channel dimension and permute back
        x = x.squeeze(1)  # (B, H, W, 7)
        x = x.permute(0, 3, 1, 2)  # (B, 7, H, W)

        # Spectral expansion (7 → 198 bands)
        # Reshape for 1D conv along spectral dimension
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (B, 7, H*W)
        x = self.spectral_expand(x)  # (B, 198, H*W)
        x = x.view(B, self.out_channels, H, W)  # (B, 198, H, W)

        return x


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net with fewer layers and channels.

    More memory-efficient for limited GPU resources.
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=198,
        base_features=16
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Simplified architecture
        self.init_conv = Conv3DBlock(1, base_features)

        # Shallow encoder
        self.down1 = DownBlock(base_features, base_features * 2)
        self.down2 = DownBlock(base_features * 2, base_features * 4)

        # Bottleneck
        self.bottleneck = Conv3DBlock(base_features * 4, base_features * 8)

        # Decoder
        self.up1 = UpBlock(base_features * 8, base_features * 4)
        self.up2 = UpBlock(base_features * 4, base_features * 2)

        # Final
        self.final_conv = nn.Conv3d(base_features * 2, 1, kernel_size=1)

        # Spectral expansion
        self.spectral_expand = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C_in, H, W = x.shape

        # Reshape to 3D
        x = x.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, 7)

        # Encoder
        x = self.init_conv(x)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)

        # Final
        x = self.final_conv(x)
        x = x.squeeze(1).permute(0, 3, 1, 2)

        # Spectral expansion
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = self.spectral_expand(x)
        x = x.view(B, self.out_channels, H, W)

        return x


# Test
if __name__ == "__main__":
    print("Testing 3D U-Net...")

    # Test full model
    model = UNet3D(
        in_channels=7,
        out_channels=198,
        base_features=32
    )

    x = torch.randn(1, 7, 256, 256)
    print(f"Input shape: {x.shape}")

    y = model(x)
    print(f"Output shape: {y.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("\n" + "="*50)
    print("Testing Lightweight 3D U-Net...")

    # Test lightweight
    model_light = LightweightUNet3D(
        in_channels=7,
        out_channels=198,
        base_features=16
    )

    y_light = model_light(x)
    print(f"Output shape: {y_light.shape}")

    n_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"Lightweight parameters: {n_params_light:,}")
