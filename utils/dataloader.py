"""
Data Loader for Hyperspectral Training Patches

Loads paired Landsat-AVIRIS patches from HDF5 files for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path


class HyperspectralPairDataset(Dataset):
    """
    Dataset for paired Landsat-AVIRIS hyperspectral patches.

    Loads from HDF5 files created by SyntheticDataGenerator.
    """

    def __init__(
        self,
        h5_files,
        normalize=True,
        transform=None
    ):
        """
        Initialize dataset.

        Parameters:
        -----------
        h5_files : list of Path or str
            List of HDF5 files containing patches
        normalize : bool
            Whether to normalize data to [0, 1] range
        transform : callable, optional
            Optional transform to apply to patches
        """
        self.h5_files = [Path(f) for f in h5_files]
        self.normalize = normalize
        self.transform = transform

        # Load all patches into memory (small dataset)
        self.aviris_patches = []
        self.landsat_patches = []

        print(f"Loading {len(self.h5_files)} HDF5 files...")

        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                aviris = f['aviris'][:]  # (N, H, W, C)
                landsat = f['landsat'][:]  # (N, H, W, C)

                # Convert to torch tensors and reorder to (N, C, H, W)
                aviris = torch.from_numpy(aviris).float().permute(0, 3, 1, 2)
                landsat = torch.from_numpy(landsat).float().permute(0, 3, 1, 2)

                self.aviris_patches.append(aviris)
                self.landsat_patches.append(landsat)

        # Concatenate all patches
        self.aviris_patches = torch.cat(self.aviris_patches, dim=0)
        self.landsat_patches = torch.cat(self.landsat_patches, dim=0)

        print(f"  Loaded {len(self.aviris_patches)} patches")
        print(f"  AVIRIS shape: {self.aviris_patches.shape}")
        print(f"  Landsat shape: {self.landsat_patches.shape}")

        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        print("Computing normalization statistics...")

        # Use robust statistics (percentiles) to avoid outlier influence
        # Convert to numpy for efficient percentile calculation
        aviris_np = self.aviris_patches.numpy()
        landsat_np = self.landsat_patches.numpy()

        self.aviris_min = torch.tensor(np.percentile(aviris_np, 1))
        self.aviris_max = torch.tensor(np.percentile(aviris_np, 99))

        self.landsat_min = torch.tensor(np.percentile(landsat_np, 1))
        self.landsat_max = torch.tensor(np.percentile(landsat_np, 99))

        print(f"  AVIRIS range: [{self.aviris_min:.2f}, {self.aviris_max:.2f}]")
        print(f"  Landsat range: [{self.landsat_min:.2f}, {self.landsat_max:.2f}]")

    def __len__(self):
        return len(self.aviris_patches)

    def __getitem__(self, idx):
        """
        Get a single training pair.

        Returns:
        --------
        dict with keys:
            'landsat': (7, H, W) - Low-resolution multispectral input
            'aviris': (340, H, W) - High-resolution hyperspectral target
        """
        aviris = self.aviris_patches[idx]
        landsat = self.landsat_patches[idx]

        # Normalize to [0, 1] range
        if self.normalize:
            aviris = (aviris - self.aviris_min) / (self.aviris_max - self.aviris_min + 1e-8)
            landsat = (landsat - self.landsat_min) / (self.landsat_max - self.landsat_min + 1e-8)

            # Clip to [0, 1]
            aviris = torch.clamp(aviris, 0, 1)
            landsat = torch.clamp(landsat, 0, 1)

        # Apply transforms if any
        if self.transform:
            aviris, landsat = self.transform(aviris, landsat)

        return {
            'landsat': landsat,
            'aviris': aviris
        }


def create_dataloaders(
    train_files,
    val_files,
    batch_size=4,
    num_workers=4,
    normalize=True
):
    """
    Create train and validation dataloaders.

    Parameters:
    -----------
    train_files : list
        List of training HDF5 files
    val_files : list
        List of validation HDF5 files
    batch_size : int
        Batch size for training
    num_workers : int
        Number of data loading workers
    normalize : bool
        Whether to normalize data

    Returns:
    --------
    train_loader, val_loader : DataLoader
        Training and validation dataloaders
    train_dataset : Dataset
        Training dataset (to access normalization stats)
    """
    # Create datasets
    train_dataset = HyperspectralPairDataset(
        train_files,
        normalize=normalize
    )

    val_dataset = HyperspectralPairDataset(
        val_files,
        normalize=normalize
    )

    # Copy normalization stats to validation set
    if normalize:
        val_dataset.aviris_min = train_dataset.aviris_min
        val_dataset.aviris_max = train_dataset.aviris_max
        val_dataset.landsat_min = train_dataset.landsat_min
        val_dataset.landsat_max = train_dataset.landsat_max

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset


def find_all_patch_files(output_dir, pattern="*.h5"):
    """
    Find all patch HDF5 files in a directory.

    Parameters:
    -----------
    output_dir : Path or str
        Directory containing patch files
    pattern : str
        Glob pattern for patch files

    Returns:
    --------
    files : list of Path
        List of HDF5 patch files
    """
    output_dir = Path(output_dir)
    # Search recursively
    files = sorted(output_dir.rglob(pattern))
    return files


def split_train_val(files, val_fraction=0.2, seed=42):
    """
    Split files into train and validation sets.

    Parameters:
    -----------
    files : list
        List of file paths
    val_fraction : float
        Fraction of data to use for validation
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    train_files, val_files : list
        Training and validation file lists
    """
    np.random.seed(seed)

    files = list(files)
    np.random.shuffle(files)

    n_val = max(1, int(len(files) * val_fraction))
    val_files = files[:n_val]
    train_files = files[n_val:]

    return train_files, val_files


if __name__ == "__main__":
    # Test data loader
    print("Testing Hyperspectral Data Loader...")
    print("=" * 60)

    # Find patch files
    output_dir = Path("outputs/dataset_small")

    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        print("Please run dataset generation first.")
        exit(1)

    patch_files = find_all_patch_files(output_dir)
    print(f"Found {len(patch_files)} patch files:")
    for f in patch_files:
        print(f"  - {f.name}")
    print()

    if len(patch_files) == 0:
        print("No patch files found. Please run dataset generation first.")
        exit(1)

    # Split into train/val
    train_files, val_files = split_train_val(patch_files, val_fraction=0.2)
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print()

    # Create dataloaders
    train_loader, val_loader, train_dataset = create_dataloaders(
        train_files,
        val_files,
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Test loading a batch
    print("Loading first batch...")
    batch = next(iter(train_loader))

    print(f"  Landsat shape: {batch['landsat'].shape}")
    print(f"  AVIRIS shape: {batch['aviris'].shape}")
    print(f"  Landsat range: [{batch['landsat'].min():.4f}, {batch['landsat'].max():.4f}]")
    print(f"  AVIRIS range: [{batch['aviris'].min():.4f}, {batch['aviris'].max():.4f}]")

    print("=" * 60)
    print("✓ Data loader tested successfully!")
