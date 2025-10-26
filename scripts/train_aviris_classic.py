"""
Train spectral super-resolution model on AVIRIS Classic dataset.

Input:  7 Landsat bands @ 256×256 (simulated, upsampled from ~126×126)
Output: 198 AVIRIS Classic bands @ 256×256

This is Stage 1 - Spectral SR only (spatial resolution already matched).
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
from models.stage1_spectral.spectral_sr_net import SpectralSRNet
from utils.losses import CombinedLoss


class AVIRISClassicDataset(Dataset):
    """
    Dataset loader for AVIRIS Classic HDF5 files.

    Returns paired patches:
    - landsat: (7, 256, 256) - simulated Landsat
    - aviris: (198, 256, 256) - AVIRIS Classic ground truth
    """

    def __init__(self, h5_file, normalize=True):
        """
        Initialize dataset.

        Parameters:
        -----------
        h5_file : str
            Path to HDF5 file
        normalize : bool
            Apply normalization (robust percentile-based)
        """
        self.h5_file = h5_file
        self.normalize = normalize

        # Open file to get metadata
        with h5py.File(h5_file, 'r') as f:
            self.n_patches = f.attrs['n_patches']
            self.aviris_bands = f.attrs['aviris_bands']
            self.landsat_bands = f.attrs['landsat_bands']

            # Compute normalization statistics
            if normalize:
                print("Computing normalization statistics...")
                aviris_data = f['aviris'][:]
                landsat_data = f['landsat'][:]

                # Use 2nd and 98th percentile for robust normalization
                self.aviris_p2 = np.percentile(aviris_data, 2)
                self.aviris_p98 = np.percentile(aviris_data, 98)
                self.landsat_p2 = np.percentile(landsat_data, 2)
                self.landsat_p98 = np.percentile(landsat_data, 98)

                print(f"  AVIRIS:  [{self.aviris_p2:.4f}, {self.aviris_p98:.4f}]")
                print(f"  Landsat: [{self.landsat_p2:.4f}, {self.landsat_p98:.4f}]")

        print(f"Loaded dataset: {self.n_patches} patches")
        print(f"  Landsat: {self.landsat_bands} bands")
        print(f"  AVIRIS:  {self.aviris_bands} bands")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # Load patches (H, W, C)
            landsat = f['landsat'][idx]  # (256, 256, 7)
            aviris = f['aviris'][idx]    # (256, 256, 198)

        # Normalize
        if self.normalize:
            landsat = (landsat - self.landsat_p2) / (self.landsat_p98 - self.landsat_p2 + 1e-8)
            aviris = (aviris - self.aviris_p2) / (self.aviris_p98 - self.aviris_p2 + 1e-8)

            # Clip to [0, 1]
            landsat = np.clip(landsat, 0, 1)
            aviris = np.clip(aviris, 0, 1)

        # Convert to torch tensors (C, H, W)
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

            # Forward pass
            output = model(landsat)
            loss, loss_dict = criterion(output, aviris)

            # Backward pass
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
    parser = argparse.ArgumentParser(description='Train AVIRIS Classic spectral SR model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/aviris_classic_training',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("="*70)
    print("AVIRIS Classic Spectral SR Training")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Check device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    full_dataset = AVIRISClassicDataset(args.data, normalize=True)

    # Split train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} patches")
    print(f"  Val:   {len(val_dataset)} patches")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print(f"\nCreating model...")
    model = SpectralSRNet(
        in_bands=7,
        out_bands=198,
        hidden_dim=128,
        num_res_blocks=8,
        use_attention=True
    ).to(device)

    # Count parameters
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
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # Training loop
    print(f"\nStarting training...")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print stats
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
            print(f"  Saved checkpoint: {checkpoint_path.name}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  ✓ New best model! Val loss: {val_loss:.6f}")

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
