"""
Train 3D U-Net for Joint Spatial-Spectral Super-Resolution

Single end-to-end model that directly learns:
  Landsat (7 bands @ 256×256) → AVIRIS (198 bands @ 256×256)

Using 3D convolutions to capture spatial-spectral correlations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.unet3d import UNet3D, LightweightUNet3D
from utils.losses import CombinedLoss


class SimpleDataset(Dataset):
    """Simple dataset for 3D U-Net training."""

    def __init__(self, h5_file, normalize=True):
        self.h5_file = h5_file
        self.normalize = normalize

        with h5py.File(h5_file, 'r') as f:
            self.n_patches = f.attrs['n_patches']

            # Load into memory
            print("Loading dataset...")
            self.landsat_data = f['landsat'][:]
            self.aviris_data = f['aviris'][:]

            if normalize:
                self.aviris_p2 = np.percentile(self.aviris_data, 2)
                self.aviris_p98 = np.percentile(self.aviris_data, 98)
                self.landsat_p2 = np.percentile(self.landsat_data, 2)
                self.landsat_p98 = np.percentile(self.landsat_data, 98)

        print(f"Loaded {self.n_patches} patches")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        landsat = self.landsat_data[idx]
        aviris = self.aviris_data[idx]

        if self.normalize:
            landsat = (landsat - self.landsat_p2) / (self.landsat_p98 - self.landsat_p2 + 1e-8)
            aviris = (aviris - self.aviris_p2) / (self.aviris_p98 - self.aviris_p2 + 1e-8)
            landsat = np.clip(landsat, 0, 1)
            aviris = np.clip(aviris, 0, 1)

        # Convert to torch (C, H, W)
        landsat = torch.from_numpy(landsat.transpose(2, 0, 1)).float()
        aviris = torch.from_numpy(aviris.transpose(2, 0, 1)).float()

        return landsat, aviris


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    with tqdm(dataloader, desc="Training") as pbar:
        for landsat, aviris in pbar:
            landsat = landsat.to(device)
            aviris = aviris.to(device)

            # Forward
            output = model(landsat)
            loss, loss_dict = criterion(output, aviris)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss_dict)

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for landsat, aviris in dataloader:
            landsat = landsat.to(device)
            aviris = aviris.to(device)

            output = model(landsat)
            loss, _ = criterion(output, aviris)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train 3D U-Net for joint spatial-spectral SR')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/3d_unet_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (small due to 3D memory)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model')
    parser.add_argument('--base-features', type=int, default=16, help='Base feature channels')
    parser.add_argument('--num-workers', type=int, default=0, help='Data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("="*70)
    print("3D U-Net Training: Joint Spatial-Spectral SR")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    full_dataset = SimpleDataset(args.data, normalize=True)

    # Split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    # Create model
    print(f"\nCreating 3D U-Net...")
    if args.lightweight:
        model = LightweightUNet3D(
            in_channels=7, out_channels=198,
            base_features=args.base_features
        ).to(device)
        print("  Using lightweight model")
    else:
        model = UNet3D(
            in_channels=7, out_channels=198,
            base_features=args.base_features
        ).to(device)
        print("  Using full model")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(
        l1_weight=1.0,
        sam_weight=0.1,
        spectral_grad_weight=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print(f"\nStarting training...")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path.name}")

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  ✓ New best! Val loss: {val_loss:.6f}")

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
