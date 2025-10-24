"""
Loss Functions for Hyperspectral Super-Resolution

Specialized loss functions for evaluating spectral and spatial fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralAngleMapper(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.

    Measures spectral similarity by computing the angle between spectra.
    Lower values indicate more similar spectral signatures.

    Range: [0, π/2] radians (or [0, 90] degrees)
    """

    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Compute SAM loss.

        Args:
            pred: Predicted hyperspectral cube (B, C, H, W)
            target: Ground truth hyperspectral cube (B, C, H, W)

        Returns:
            SAM loss (scalar if reduction='mean', else (B, H, W))
        """
        # Normalize along spectral dimension
        # (B, C, H, W) -> reshape to (B*H*W, C)
        b, c, h, w = pred.shape

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)

        # Compute dot product and magnitudes
        dot_product = torch.sum(pred_flat * target_flat, dim=1)
        pred_norm = torch.norm(pred_flat, dim=1)
        target_norm = torch.norm(target_flat, dim=1)

        # Compute cosine similarity
        cos_angle = dot_product / (pred_norm * target_norm + self.eps)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

        # Compute angle in radians
        angle = torch.acos(cos_angle)

        # Reshape back if needed
        if self.reduction == 'none':
            angle = angle.reshape(b, h, w)
        elif self.reduction == 'mean':
            angle = torch.mean(angle)
        elif self.reduction == 'sum':
            angle = torch.sum(angle)

        return angle


class SpectralGradientLoss(nn.Module):
    """
    Spectral Gradient Loss.

    Encourages smooth spectral curves by penalizing large gradients
    along the spectral dimension.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute spectral gradient loss.

        Args:
            pred: Predicted hyperspectral cube (B, C, H, W)
            target: Ground truth hyperspectral cube (B, C, H, W)

        Returns:
            Spectral gradient loss
        """
        # Compute spectral gradients (difference between adjacent bands)
        pred_grad = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_grad = target[:, 1:, :, :] - target[:, :-1, :, :]

        # L1 loss on gradients
        loss = F.l1_loss(pred_grad, target_grad, reduction=self.reduction)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for hyperspectral super-resolution.

    Combines multiple loss terms with adjustable weights:
    - L1/L2 reconstruction loss
    - Spectral Angle Mapper (SAM)
    - Spectral gradient loss
    """

    def __init__(
        self,
        l1_weight=1.0,
        l2_weight=0.0,
        sam_weight=0.1,
        spectral_grad_weight=0.1,
        reduction='mean'
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sam_weight = sam_weight
        self.spectral_grad_weight = spectral_grad_weight

        self.sam = SpectralAngleMapper(reduction=reduction)
        self.spectral_grad = SpectralGradientLoss(reduction=reduction)

    def forward(self, pred, target):
        """
        Compute combined loss.

        Args:
            pred: Predicted hyperspectral cube (B, C, H, W)
            target: Ground truth hyperspectral cube (B, C, H, W)

        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # L1 reconstruction loss
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(pred, target)
            loss_dict['l1'] = l1_loss.item()
            total_loss += self.l1_weight * l1_loss

        # L2 reconstruction loss
        if self.l2_weight > 0:
            l2_loss = F.mse_loss(pred, target)
            loss_dict['l2'] = l2_loss.item()
            total_loss += self.l2_weight * l2_loss

        # Spectral Angle Mapper
        if self.sam_weight > 0:
            sam_loss = self.sam(pred, target)
            loss_dict['sam'] = sam_loss.item()
            total_loss += self.sam_weight * sam_loss

        # Spectral gradient loss
        if self.spectral_grad_weight > 0:
            sg_loss = self.spectral_grad(pred, target)
            loss_dict['spectral_grad'] = sg_loss.item()
            total_loss += self.spectral_grad_weight * sg_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class SpatialLoss(nn.Module):
    """
    Loss for spatial super-resolution stage.

    Combines pixel-wise reconstruction with perceptual similarity.
    """

    def __init__(
        self,
        l1_weight=1.0,
        l2_weight=0.0,
        sam_weight=0.05,
        reduction='mean'
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sam_weight = sam_weight

        self.sam = SpectralAngleMapper(reduction=reduction)

    def forward(self, pred, target):
        """
        Compute spatial SR loss.

        Args:
            pred: Predicted HR hyperspectral cube (B, C, H, W)
            target: Ground truth HR hyperspectral cube (B, C, H, W)

        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # L1 reconstruction loss
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(pred, target)
            loss_dict['l1'] = l1_loss.item()
            total_loss += self.l1_weight * l1_loss

        # L2 reconstruction loss
        if self.l2_weight > 0:
            l2_loss = F.mse_loss(pred, target)
            loss_dict['l2'] = l2_loss.item()
            total_loss += self.l2_weight * l2_loss

        # Spectral consistency
        if self.sam_weight > 0:
            sam_loss = self.sam(pred, target)
            loss_dict['sam'] = sam_loss.item()
            total_loss += self.sam_weight * sam_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Hyperspectral Loss Functions...")
    print("=" * 60)

    # Create test data
    pred = torch.randn(2, 340, 64, 64).to(device)
    target = torch.randn(2, 340, 64, 64).to(device)

    print(f"Test input shape: {pred.shape}")
    print()

    # Test SAM
    print("1. Spectral Angle Mapper:")
    sam = SpectralAngleMapper().to(device)
    sam_loss = sam(pred, target)
    print(f"   SAM loss: {sam_loss.item():.4f} radians ({np.rad2deg(sam_loss.item()):.2f}°)")
    print()

    # Test Spectral Gradient Loss
    print("2. Spectral Gradient Loss:")
    sg = SpectralGradientLoss().to(device)
    sg_loss = sg(pred, target)
    print(f"   Spectral gradient loss: {sg_loss.item():.6f}")
    print()

    # Test Combined Loss
    print("3. Combined Loss:")
    combined = CombinedLoss().to(device)
    total_loss, loss_dict = combined(pred, target)
    print(f"   Total loss: {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        if k != 'total':
            print(f"   - {k}: {v:.6f}")
    print()

    # Test Spatial Loss
    print("4. Spatial Loss:")
    spatial = SpatialLoss().to(device)
    spatial_loss, spatial_dict = spatial(pred, target)
    print(f"   Total loss: {spatial_loss.item():.6f}")
    for k, v in spatial_dict.items():
        if k != 'total':
            print(f"   - {k}: {v:.6f}")

    print("=" * 60)
    print("✓ Loss functions tested successfully!")
