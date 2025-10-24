"""
Stage 2: Spatial Super-Resolution Network

Upsamples 340-band hyperspectral cube from 30m to 4m resolution (7.5× upsampling).
Uses 3D convolutions to leverage spectral-spatial correlations.

Input:  (B, 340, H, W)      - 340 bands @ 30m resolution
Output: (B, 340, 7.5H, 7.5W) - 340 bands @ 4m resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualChannelAttentionBlock(nn.Module):
    """
    Residual Channel Attention Block (RCAB) adapted for hyperspectral data.
    Uses 2D convolutions to preserve spectral dimension.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        # Apply channel attention
        out = out * self.ca(out)

        return out + residual


class ResidualGroup(nn.Module):
    """
    Residual Group: Multiple RCAB blocks with long skip connection.
    """
    def __init__(self, channels, num_blocks=4):
        super().__init__()

        self.blocks = nn.Sequential(*[
            ResidualChannelAttentionBlock(channels)
            for _ in range(num_blocks)
        ])
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.blocks(x)
        out = self.conv(out)
        return out + residual


class PixelShuffleUpsampler(nn.Module):
    """
    Sub-pixel convolution upsampler (efficient).
    Rearranges channel dimension to spatial dimensions.
    """
    def __init__(self, channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels * (scale_factor ** 2),
            3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class SpatialSRNet(nn.Module):
    """
    Spatial Super-Resolution Network for Hyperspectral Images.

    Based on RCAN (Residual Channel Attention Network) adapted for multi-band data.

    Architecture:
    1. Shallow feature extraction
    2. Deep feature extraction (Residual Groups)
    3. Upsampling (sub-pixel convolution)
    4. Reconstruction
    """

    def __init__(
        self,
        num_bands=340,
        num_features=64,
        num_groups=4,
        num_blocks_per_group=4,
        scale_factor=7.5,  # Note: will use 8× then downsample slightly
        reduction=16
    ):
        super().__init__()

        self.num_bands = num_bands
        self.scale_factor = scale_factor

        # For non-integer scale factors, use nearest integer + interpolation
        self.use_integer_scale = (scale_factor == int(scale_factor))
        self.integer_scale = int(round(scale_factor))

        # Shallow feature extraction
        self.shallow_feat = nn.Conv2d(num_bands, num_features, 3, padding=1)

        # Deep feature extraction (Residual Groups)
        self.res_groups = nn.Sequential(*[
            ResidualGroup(num_features, num_blocks_per_group)
            for _ in range(num_groups)
        ])

        self.conv_after_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        # For 7.5×, use 8× pixel shuffle then slight downscale
        if self.integer_scale == 2:
            self.upsample = nn.Sequential(
                PixelShuffleUpsampler(num_features, 2)
            )
        elif self.integer_scale == 3:
            self.upsample = nn.Sequential(
                PixelShuffleUpsampler(num_features, 3)
            )
        elif self.integer_scale == 4:
            self.upsample = nn.Sequential(
                PixelShuffleUpsampler(num_features, 2),
                PixelShuffleUpsampler(num_features, 2)
            )
        elif self.integer_scale == 8:
            self.upsample = nn.Sequential(
                PixelShuffleUpsampler(num_features, 2),
                PixelShuffleUpsampler(num_features, 2),
                PixelShuffleUpsampler(num_features, 2)
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

        # Final reconstruction
        self.reconstruction = nn.Conv2d(num_features, num_bands, 3, padding=1)

        # Bicubic upsampling for residual learning
        self.bicubic_upsample = lambda x: F.interpolate(
            x,
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input hyperspectral cube (B, 340, H, W)

        Returns:
            Upsampled hyperspectral cube (B, 340, 7.5H, 7.5W)
        """
        # Bicubic baseline for residual learning
        bicubic_up = self.bicubic_upsample(x)

        # Shallow features
        feat_shallow = self.shallow_feat(x)

        # Deep features
        feat_deep = self.res_groups(feat_shallow)
        feat_deep = self.conv_after_body(feat_deep)

        # Global residual learning
        feat = feat_shallow + feat_deep

        # Upsampling
        feat_up = self.upsample(feat)

        # Reconstruction
        out = self.reconstruction(feat_up)

        # For non-integer scales, adjust size to match target
        if not self.use_integer_scale:
            target_h = int(x.size(2) * self.scale_factor)
            target_w = int(x.size(3) * self.scale_factor)

            if out.size(2) != target_h or out.size(3) != target_w:
                out = F.interpolate(
                    out,
                    size=(target_h, target_w),
                    mode='bicubic',
                    align_corners=False
                )

        # Residual learning: add bicubic baseline
        # Resize bicubic to match output if needed
        if bicubic_up.shape != out.shape:
            bicubic_up = F.interpolate(
                bicubic_up,
                size=(out.size(2), out.size(3)),
                mode='bicubic',
                align_corners=False
            )

        out = out + bicubic_up

        return out


class LightweightSpatialSRNet(nn.Module):
    """
    Lightweight version for faster training with limited data.

    Uses fewer parameters but maintains key architectural features.
    """

    def __init__(
        self,
        num_bands=340,
        num_features=32,
        num_res_blocks=4,
        scale_factor=7.5
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(num_bands, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.extend([
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.res_blocks = nn.Sequential(*res_blocks)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 8× total
        )

        # Reconstruction
        self.reconstruct = nn.Conv2d(num_features, num_bands, 3, padding=1)

    def forward(self, x):
        # Bicubic baseline
        bicubic_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Feature extraction + residual blocks
        feat = self.feat_extract(x)
        feat = self.res_blocks(feat) + feat

        # Upsampling
        feat = self.upsample(feat)

        # Reconstruction
        out = self.reconstruct(feat)

        # Adjust to target size
        target_h = int(x.size(2) * self.scale_factor)
        target_w = int(x.size(3) * self.scale_factor)

        if out.size(2) != target_h or out.size(3) != target_w:
            out = F.interpolate(
                out,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False
            )

        # Residual learning
        if bicubic_up.shape != out.shape:
            bicubic_up = F.interpolate(
                bicubic_up,
                size=(out.size(2), out.size(3)),
                mode='bicubic',
                align_corners=False
            )

        return out + bicubic_up


if __name__ == "__main__":
    # Test the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Spatial SR Networks...")
    print("=" * 60)

    # Create test input (batch=2, 340 bands, 34×34 @ 30m)
    x = torch.randn(2, 340, 34, 34).to(device)
    print(f"Input shape: {x.shape} (34×34 @ 30m resolution)")
    print(f"Expected output: (2, 340, 256, 256) (256×256 @ 4m resolution)")
    print()

    # Test SpatialSRNet
    print("1. Full SpatialSRNet:")
    model1 = SpatialSRNet(num_groups=2, num_blocks_per_group=2).to(device)
    out1 = model1(x)
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    print()

    # Test Lightweight version
    print("2. LightweightSpatialSRNet:")
    model2 = LightweightSpatialSRNet().to(device)
    out2 = model2(x)
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("=" * 60)
    print("✓ Networks initialized successfully!")
