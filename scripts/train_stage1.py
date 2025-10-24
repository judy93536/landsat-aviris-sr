#!/usr/bin/env python3
"""
Training script for Stage 1: Spectral Super-Resolution

Trains SpectralSRNet to expand 7 Landsat bands to 340 AVIRIS bands.

Usage:
    python scripts/train_stage1.py \
        --data-dir outputs/dataset_small \
        --output-dir outputs/stage1_training \
        --epochs 100 \
        --batch-size 4
"""

import sys
import argparse
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stage1_spectral.spectral_sr_net import SpectralSRNet, SpectralUnmixingNet
from utils.dataloader import create_dataloaders, find_all_patch_files, split_train_val
from utils.losses import CombinedLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.

    Parameters:
    -----------
    model : nn.Module
        SpectralSRNet model
    dataloader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number

    Returns:
    --------
    avg_loss : float
        Average loss for the epoch
    loss_dict : dict
        Dictionary with individual loss components
    """
    model.train()

    total_loss = 0.0
    total_loss_dict = {}

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        landsat = batch['landsat'].to(device)  # (B, 7, H, W)
        aviris = batch['aviris'].to(device)     # (B, 340, H, W)

        # Forward pass
        optimizer.zero_grad()
        pred_aviris = model(landsat)

        # Compute loss
        loss, loss_dict = criterion(pred_aviris, aviris)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        for k, v in loss_dict.items():
            total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v

        # Print progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.6f} "
                  f"(L1: {loss_dict['l1']:.6f}, "
                  f"SAM: {loss_dict['sam']:.6f})")

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in total_loss_dict.items()}

    return avg_loss, avg_loss_dict


def validate(model, dataloader, criterion, device):
    """
    Validate model.

    Parameters:
    -----------
    model : nn.Module
        SpectralSRNet model
    dataloader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to validate on

    Returns:
    --------
    avg_loss : float
        Average validation loss
    loss_dict : dict
        Dictionary with individual loss components
    """
    model.eval()

    total_loss = 0.0
    total_loss_dict = {}

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            landsat = batch['landsat'].to(device)
            aviris = batch['aviris'].to(device)

            # Forward pass
            pred_aviris = model(landsat)

            # Compute loss
            loss, loss_dict = criterion(pred_aviris, aviris)

            # Accumulate loss
            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in total_loss_dict.items()}

    return avg_loss, avg_loss_dict


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 1: Spectral Super-Resolution"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs/dataset_small",
        help="Directory containing patch HDF5 files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage1_training",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spectral_sr",
        choices=["spectral_sr", "spectral_unmixing"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1 Training: Spectral Super-Resolution")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Find patch files
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    patch_files = find_all_patch_files(data_dir)
    print(f"\nFound {len(patch_files)} patch files:")
    for f in patch_files:
        print(f"  - {f.name}")
    print()

    if len(patch_files) == 0:
        print("ERROR: No patch files found!")
        sys.exit(1)

    # Split into train/val
    train_files, val_files = split_train_val(
        patch_files,
        val_fraction=args.val_fraction
    )
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, train_dataset = create_dataloaders(
        train_files,
        val_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print()

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "spectral_sr":
        model = SpectralSRNet(
            in_bands=7,
            out_bands=340,
            hidden_dim=128,
            num_res_blocks=8
        )
    else:  # spectral_unmixing
        model = SpectralUnmixingNet(
            in_bands=7,
            out_bands=340,
            num_endmembers=16
        )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Parameters: {num_params:,}")
    print()

    # Create loss and optimizer
    criterion = CombinedLoss(
        l1_weight=1.0,
        l2_weight=0.0,
        sam_weight=0.1,
        spectral_grad_weight=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        print()

    # TensorBoard writer
    writer = SummaryWriter(output_dir / "tensorboard")

    # Training loop
    print("Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_loss_dict = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        for k in train_loss_dict:
            if k != 'total':
                writer.add_scalar(f'Train/{k}', train_loss_dict[k], epoch)
                writer.add_scalar(f'Val/{k}', val_loss_dict[k], epoch)

        epoch_time = time.time() - epoch_start_time

        # Print summary
        print(f"Epoch {epoch} ({epoch_time:.1f}s):")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss:   {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_checkpoint_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

        print()

    print("=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    writer.close()


if __name__ == "__main__":
    main()
