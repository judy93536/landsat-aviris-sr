"""
Stage 1: Spectral Super-Resolution Network

Expands 7 Landsat bands to 340 AVIRIS-like hyperspectral bands.
Uses spectral unmixing + learned spectral correlations.

Input:  (B, 7, H, W)    - 7 Landsat bands
Output: (B, 340, H, W)  - 340 AVIRIS-like bands (same spatial resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralResidualBlock(nn.Module):
    """
    Residual block operating in spectral dimension.
    Uses 1D convolutions along spectral axis.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class SpectralAttention(nn.Module):
    """
    Channel (spectral) attention mechanism.
    Learns importance weights for each spectral band.
    """
    def __init__(self, num_bands):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_bands, num_bands // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // 4, num_bands),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L) where C is bands, L is spatial dimension flattened
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SpectralSRNet(nn.Module):
    """
    Spectral Super-Resolution Network.

    Architecture:
    1. Initial spectral expansion (7 → 340 bands)
    2. Spectral residual blocks for refinement
    3. Spectral attention for band weighting
    4. Skip connection for stability
    """

    def __init__(
        self,
        in_bands=7,
        out_bands=340,
        hidden_dim=128,
        num_res_blocks=8,
        use_attention=True
    ):
        super().__init__()

        self.in_bands = in_bands
        self.out_bands = out_bands

        # Initial spectral expansion
        # Uses learned linear combination of input bands
        self.spectral_expand = nn.Sequential(
            nn.Conv2d(in_bands, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_bands, 3, padding=1)
        )

        # Spectral refinement blocks
        # Operate on spectral dimension to learn spectral correlations
        self.res_blocks = nn.ModuleList([
            SpectralResidualBlock(out_bands)
            for _ in range(num_res_blocks)
        ])

        # Spectral attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpectralAttention(out_bands)

        # Final refinement
        self.final_conv = nn.Conv2d(out_bands, out_bands, 3, padding=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input Landsat bands (B, 7, H, W)

        Returns:
            Reconstructed hyperspectral cube (B, 340, H, W)
        """
        # Initial spectral expansion
        out = self.spectral_expand(x)
        residual = out

        # Reshape for 1D spectral processing
        # (B, C, H, W) → (B, C, H*W)
        b, c, h, w = out.shape
        out_1d = out.view(b, c, h * w)

        # Apply spectral residual blocks
        for block in self.res_blocks:
            out_1d = block(out_1d)

        # Apply spectral attention
        if self.use_attention:
            out_1d = self.attention(out_1d)

        # Reshape back to 2D
        out = out_1d.view(b, c, h, w)

        # Skip connection + final refinement
        out = self.final_conv(out + residual)

        return out


class SpectralUnmixingNet(nn.Module):
    """
    Alternative: Spectral unmixing-based approach.

    Learns endmember spectra and abundance maps.
    More interpretable but may be less flexible.
    """

    def __init__(
        self,
        in_bands=7,
        out_bands=340,
        num_endmembers=16
    ):
        super().__init__()

        self.num_endmembers = num_endmembers

        # Abundance estimation from input bands
        self.abundance_net = nn.Sequential(
            nn.Conv2d(in_bands, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_endmembers, 3, padding=1),
            nn.Softmax(dim=1)  # Abundances sum to 1
        )

        # Learned endmember spectra (num_endmembers × out_bands)
        self.endmembers = nn.Parameter(
            torch.randn(num_endmembers, out_bands)
        )

    def forward(self, x):
        """
        Forward pass using linear mixing model.

        Args:
            x: Input Landsat bands (B, 7, H, W)

        Returns:
            Reconstructed spectra (B, 340, H, W)
        """
        # Estimate abundance maps
        abundances = self.abundance_net(x)  # (B, num_endmembers, H, W)

        # Linear spectral mixing
        # out = sum(abundance_i * endmember_i)
        b, _, h, w = abundances.shape
        abundances_flat = abundances.view(b, self.num_endmembers, -1)

        # Matrix multiply: (num_endmembers, out_bands) × (num_endmembers, H*W)
        spectra = torch.matmul(
            self.endmembers.t(),
            abundances_flat
        )  # (out_bands, H*W)

        out = spectra.view(b, -1, h, w)  # (B, out_bands, H, W)

        return out


if __name__ == "__main__":
    # Test the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Spectral SR Network...")
    print("=" * 60)

    # Create test input (batch=2, 7 bands, 256×256)
    x = torch.randn(2, 7, 256, 256).to(device)
    print(f"Input shape: {x.shape}")

    # Test SpectralSRNet
    model1 = SpectralSRNet().to(device)
    out1 = model1(x)
    print(f"SpectralSRNet output: {out1.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")

    print()

    # Test SpectralUnmixingNet
    model2 = SpectralUnmixingNet().to(device)
    out2 = model2(x)
    print(f"SpectralUnmixingNet output: {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("=" * 60)
    print("✓ Networks initialized successfully!")
