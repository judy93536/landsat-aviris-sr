"""
Stage 2: Spatial Refinement Network

Refines spatial details in the Stage 1 output using residual learning.

Input:  Stage 1 output (198 bands @ 256×256, but blurry from upsampled Landsat)
Output: Refined 198 bands @ 256×256 with sharp spatial details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism (RCAN-style)."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualChannelAttentionBlock(nn.Module):
    """Residual block with channel attention (RCAB)."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.channel_attention = ChannelAttention(num_channels, reduction)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.channel_attention(out)
        out += residual
        return out


class SpatialRefinementNet(nn.Module):
    """
    Spatial refinement network for Stage 2.

    Uses residual learning to refine spatial details from Stage 1 output.
    Architecture inspired by RCAN (Residual Channel Attention Network).

    Key idea: Learn the residual (detail) to add to Stage 1 output,
    rather than predicting the full output directly.
    """

    def __init__(
        self,
        num_bands=198,
        num_features=64,
        num_rcab=8,
        reduction=16
    ):
        super().__init__()

        self.num_bands = num_bands

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_bands, num_features, 3, padding=1)

        # Residual channel attention blocks
        self.rcabs = nn.ModuleList([
            ResidualChannelAttentionBlock(num_features, reduction)
            for _ in range(num_rcab)
        ])

        # Residual connection across all RCABs
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Reconstruction (predict residual)
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_bands, 3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Stage 1 output (B, 198, H, W)

        Returns:
        --------
        out : torch.Tensor
            Refined output (B, 198, H, W)
        """
        # Store input for residual connection
        stage1_output = x

        # Shallow feature extraction
        fea = self.conv_first(x)
        residual = fea

        # Deep feature extraction with RCABs
        for rcab in self.rcabs:
            fea = rcab(fea)

        # Residual connection across RCABs
        fea = self.conv_mid(fea)
        fea += residual

        # Predict residual details
        residual_details = self.conv_last(fea)

        # Add residual to input (residual learning)
        out = stage1_output + residual_details

        return out


class LightweightSpatialRefinementNet(nn.Module):
    """
    Lightweight spatial refinement network.

    Faster and smaller version for quick experiments.
    """

    def __init__(
        self,
        num_bands=198,
        num_features=32,
        num_blocks=4
    ):
        super().__init__()

        self.num_bands = num_bands

        # Feature extraction
        self.conv_first = nn.Conv2d(num_bands, num_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Simple residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, 3, padding=1)
            )
            for _ in range(num_blocks)
        ])

        # Reconstruction
        self.conv_last = nn.Conv2d(num_features, num_bands, 3, padding=1)

    def forward(self, x):
        stage1_output = x

        # Feature extraction
        fea = self.relu(self.conv_first(x))

        # Residual blocks
        for block in self.res_blocks:
            residual = fea
            fea = block(fea)
            fea += residual
            fea = self.relu(fea)

        # Predict residual
        residual_details = self.conv_last(fea)

        # Add to input
        out = stage1_output + residual_details

        return out


# Test
if __name__ == "__main__":
    # Test full model
    model = SpatialRefinementNet(
        num_bands=198,
        num_features=64,
        num_rcab=8
    )

    x = torch.randn(2, 198, 256, 256)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("\n" + "="*50)

    # Test lightweight model
    model_light = LightweightSpatialRefinementNet(
        num_bands=198,
        num_features=32,
        num_blocks=4
    )

    y_light = model_light(x)

    print(f"Lightweight output shape: {y_light.shape}")

    n_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"Lightweight parameters: {n_params_light:,}")
