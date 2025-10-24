#!/usr/bin/env python3
"""
Quick test of the training pipeline.

Tests that all components work together:
- Data loading
- Model forward pass
- Loss computation
- Gradient flow

This is NOT actual training - just a sanity check.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stage1_spectral.spectral_sr_net import SpectralSRNet
from models.stage2_spatial.spatial_sr_net import SpatialSRNet
from utils.dataloader import create_dataloaders, find_all_patch_files, split_train_val
from utils.losses import CombinedLoss, SpatialLoss


def test_stage1():
    """Test Stage 1: Spectral SR"""
    print("\n" + "="*60)
    print("Testing Stage 1: Spectral Super-Resolution")
    print("="*60)

    # Find data
    data_dir = Path("outputs/dataset_small")
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False

    patch_files = find_all_patch_files(data_dir)
    if len(patch_files) == 0:
        print("ERROR: No patch files found!")
        return False

    print(f"Found {len(patch_files)} patch files")

    # Split data
    train_files, val_files = split_train_val(patch_files, val_fraction=0.2)

    # Create dataloaders
    train_loader, val_loader, train_dataset = create_dataloaders(
        train_files,
        val_files,
        batch_size=2,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpectralSRNet(
        in_bands=7,
        out_bands=340,
        hidden_dim=64,  # Smaller for testing
        num_res_blocks=4
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create loss and optimizer
    criterion = CombinedLoss(
        l1_weight=1.0,
        sam_weight=0.1,
        spectral_grad_weight=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Test forward pass
    print("\nTesting forward pass...")
    batch = next(iter(train_loader))
    landsat = batch['landsat'].to(device)
    aviris = batch['aviris'].to(device)

    print(f"  Input shape: {landsat.shape}")
    print(f"  Target shape: {aviris.shape}")

    pred_aviris = model(landsat)
    print(f"  Output shape: {pred_aviris.shape}")

    # Test loss computation
    print("\nTesting loss computation...")
    loss, loss_dict = criterion(pred_aviris, aviris)
    print(f"  Total loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        if k != 'total':
            print(f"  - {k}: {v:.6f}")

    # Test backward pass
    print("\nTesting backward pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  ✓ Gradient flow successful")

    print("\n✓ Stage 1 pipeline test passed!")
    return True


def test_stage2():
    """Test Stage 2: Spatial SR"""
    print("\n" + "="*60)
    print("Testing Stage 2: Spatial Super-Resolution")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model (use smaller version for testing)
    from models.stage2_spatial.spatial_sr_net import LightweightSpatialSRNet

    model = LightweightSpatialSRNet(
        num_bands=340,
        num_features=16,  # Smaller for testing
        num_res_blocks=2,
        scale_factor=7.5
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create loss and optimizer
    criterion = SpatialLoss(
        l1_weight=1.0,
        sam_weight=0.05
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Test forward pass with synthetic input
    print("\nTesting forward pass...")
    # Simulate Stage 1 output: 340 bands at low resolution (34×34)
    x_lr = torch.randn(2, 340, 34, 34).to(device)

    print(f"  Input shape: {x_lr.shape} (34×34 @ 30m)")

    pred_hr = model(x_lr)
    print(f"  Output shape: {pred_hr.shape}")

    # Create target with same size as output
    x_hr_target = torch.randn_like(pred_hr)

    # Test loss computation
    print("\nTesting loss computation...")
    loss, loss_dict = criterion(pred_hr, x_hr_target)
    print(f"  Total loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        if k != 'total':
            print(f"  - {k}: {v:.6f}")

    # Test backward pass
    print("\nTesting backward pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  ✓ Gradient flow successful")

    print("\n✓ Stage 2 pipeline test passed!")
    return True


def main():
    print("="*60)
    print("Training Pipeline Sanity Check")
    print("="*60)

    # Test Stage 1
    stage1_ok = test_stage1()

    # Test Stage 2
    stage2_ok = test_stage2()

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Stage 1: {'✓ PASS' if stage1_ok else '✗ FAIL'}")
    print(f"  Stage 2: {'✓ PASS' if stage2_ok else '✗ FAIL'}")
    print("="*60)

    if stage1_ok and stage2_ok:
        print("\n✓ All tests passed! Ready to start training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
