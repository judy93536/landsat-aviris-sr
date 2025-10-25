#!/usr/bin/env python3
"""
Stage 2 Training: Spatial Super-Resolution

This script trains the Stage 2 network for spatial super-resolution.
Uses the trained Stage 1 model to generate 340-band inputs, then trains
Stage 2 to refine spatial details to match ground truth AVIRIS.

Pipeline:
  Landsat (7 bands) → Stage 1 (frozen) → 340 bands → Stage 2 (trainable) → 340 bands @ 4m
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stage1_spectral.spectral_sr_net import SpectralSRNet
from models.stage2_spatial.spatial_sr_net import SpatialSRNet
from utils.dataloader import create_dataloaders
from utils.losses import SpatialLoss


def load_stage1_model(checkpoint_path, device):
    """Load pre-trained Stage 1 model (frozen)."""
    print(f"Loading Stage 1 model from: {checkpoint_path}")

    model = SpectralSRNet(
        in_bands=7,
        out_bands=340,
        hidden_dim=128,
        num_res_blocks=8,
        use_attention=True
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Freeze Stage 1

    for param in model.parameters():
        param.requires_grad = False

    print(f"  Stage 1 loaded (frozen)")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")

    return model


def train_epoch(stage1_model, stage2_model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    stage2_model.train()
    stage1_model.eval()  # Keep Stage 1 frozen

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (landsat, aviris) in enumerate(train_loader):
        landsat = landsat.to(device)
        aviris = aviris.to(device)

        # Stage 1: Generate 340 bands (frozen, no gradients)
        with torch.no_grad():
            stage1_out = stage1_model(landsat)

        # Stage 2: Spatial refinement (trainable)
        stage2_out = stage2_model(stage1_out)

        # Compute loss
        loss = criterion(stage2_out, aviris)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  [{epoch}][{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

    return total_loss / num_batches


def validate(stage1_model, stage2_model, val_loader, criterion, device):
    """Validate the model."""
    stage1_model.eval()
    stage2_model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for landsat, aviris in val_loader:
            landsat = landsat.to(device)
            aviris = aviris.to(device)

            # Stage 1 + Stage 2 pipeline
            stage1_out = stage1_model(landsat)
            stage2_out = stage2_model(stage1_out)

            loss = criterion(stage2_out, aviris)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2: Spatial SR')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data (HDF5 files)')
    parser.add_argument('--output-dir', type=str, default='outputs/stage2_training',
                        help='Output directory for checkpoints and logs')

    # Stage 1 checkpoint
    parser.add_argument('--stage1-checkpoint', type=str, required=True,
                        help='Path to trained Stage 1 checkpoint')

    # Model
    parser.add_argument('--model', type=str, default='spatial_sr',
                        choices=['spatial_sr'],
                        help='Model architecture (default: spatial_sr)')
    parser.add_argument('--num-features', type=int, default=64,
                        help='Number of feature channels (default: 64)')
    parser.add_argument('--num-groups', type=int, default=4,
                        help='Number of residual groups (default: 4)')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help='Number of RCAB blocks per group (default: 4)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Fraction of data for validation (default: 0.2)')

    # Misc
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard'), exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\n" + "=" * 70)
    print("Stage 2 Training: Spatial Super-Resolution")
    print("=" * 70)

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers
    )

    # Load Stage 1 model (frozen)
    stage1_model = load_stage1_model(args.stage1_checkpoint, device)

    # Create Stage 2 model
    print(f"\nCreating Stage 2 model: {args.model}")

    # Note: For now, we'll train at the same resolution (no actual upsampling)
    # This is because the data is already aligned. Stage 2 learns to refine spatial details.
    stage2_model = SpatialSRNet(
        num_bands=340,
        num_features=args.num_features,
        num_groups=args.num_groups,
        num_blocks_per_group=args.num_blocks,
        scale_factor=1.0,  # No upsampling, just spatial refinement
        reduction=16
    )

    stage2_model = stage2_model.to(device)

    num_params = sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = SpatialLoss()
    optimizer = optim.Adam(stage2_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        stage2_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))

    # Training loop
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            stage1_model, stage2_model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(stage1_model, stage2_model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch {epoch} ({epoch_time:.1f}s):")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss:   {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(
            args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch:03d}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': stage2_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': stage2_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

        print()

    print("=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {os.path.join(args.output_dir, 'checkpoints')}")
    print("=" * 70)

    writer.close()


if __name__ == '__main__':
    main()
