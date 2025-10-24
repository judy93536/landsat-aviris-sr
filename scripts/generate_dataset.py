#!/usr/bin/env python3
"""
Generate full training dataset from AVIRIS imagery.

Usage:
    python scripts/generate_dataset.py \
        --aviris-dir /path/to/aviris \
        --output-dir outputs/synthetic \
        --patch-size 256 \
        --augment

This script processes multiple AVIRIS files to create a large training dataset.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.landsat_srf import load_aviris_wavelengths
from data.synthetic import SyntheticDataGenerator
from data.augmentation import AugmentationPipeline


def find_aviris_files(aviris_dir, extensions=('.h5', '.hdf', '.hdf5', '.npy')):
    """
    Find all AVIRIS files in directory.

    Parameters:
    -----------
    aviris_dir : Path
        Directory containing AVIRIS files
    extensions : tuple
        File extensions to search for

    Returns:
    --------
    files : list of Path
        List of AVIRIS file paths
    """
    aviris_dir = Path(aviris_dir)
    files = []

    for ext in extensions:
        files.extend(aviris_dir.glob(f"**/*{ext}"))

    return sorted(files)


def merge_patch_datasets(patch_files, output_file):
    """
    Merge multiple patch HDF5 files into single dataset.

    Parameters:
    -----------
    patch_files : list of Path
        List of HDF5 patch files to merge
    output_file : Path
        Output merged dataset file
    """
    print(f"\nMerging {len(patch_files)} patch files...")

    # First pass: count total patches
    total_patches = 0
    aviris_shape = None
    landsat_shape = None

    for pf in patch_files:
        with h5py.File(pf, 'r') as f:
            n = f.attrs['n_patches']
            total_patches += n

            if aviris_shape is None:
                aviris_shape = f.attrs['aviris_shape']
                landsat_shape = f.attrs['landsat_shape']

    print(f"  Total patches: {total_patches}")

    # Create merged dataset
    with h5py.File(output_file, 'w') as f_out:
        # Create datasets
        aviris_ds = f_out.create_dataset(
            'aviris',
            shape=(total_patches,) + tuple(aviris_shape),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        landsat_ds = f_out.create_dataset(
            'landsat',
            shape=(total_patches,) + tuple(landsat_shape),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        # Copy patches
        idx = 0
        for pf in tqdm(patch_files, desc="Merging"):
            with h5py.File(pf, 'r') as f_in:
                n = f_in['aviris'].shape[0]
                aviris_ds[idx:idx+n] = f_in['aviris'][:]
                landsat_ds[idx:idx+n] = f_in['landsat'][:]
                idx += n

        # Store metadata
        f_out.attrs['n_patches'] = total_patches
        f_out.attrs['aviris_shape'] = aviris_shape
        f_out.attrs['landsat_shape'] = landsat_shape
        f_out.attrs['n_source_files'] = len(patch_files)

    print(f"  Saved merged dataset to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training dataset from AVIRIS imagery"
    )
    parser.add_argument(
        "--aviris-dir",
        type=str,
        required=True,
        help="Directory containing AVIRIS data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/synthetic",
        help="Output directory for generated patches"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size at AVIRIS resolution"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for patch extraction (default: patch_size // 2)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply augmentation to patches"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all patches into single dataset file"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of AVIRIS files to process (for testing)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Synthetic Dataset Generation")
    print("=" * 70)

    # Find AVIRIS files
    aviris_dir = Path(args.aviris_dir)
    if not aviris_dir.exists():
        print(f"ERROR: AVIRIS directory not found: {aviris_dir}")
        sys.exit(1)

    aviris_files = find_aviris_files(aviris_dir)
    print(f"\nFound {len(aviris_files)} AVIRIS files in {aviris_dir}")

    if len(aviris_files) == 0:
        print("ERROR: No AVIRIS files found!")
        sys.exit(1)

    if args.max_files is not None:
        aviris_files = aviris_files[:args.max_files]
        print(f"  Limited to {len(aviris_files)} files for testing")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    print("\nInitializing synthetic data generator...")
    generator = SyntheticDataGenerator(
        spatial_downsampling_factor=7.5,
        psf_sigma=2.0,
        landsat_sensor="LC08"
    )

    # Initialize augmentation if requested
    if args.augment:
        print("Augmentation enabled")
        augmentation = AugmentationPipeline(
            geometric=True,
            noise=True,
            artifacts=True,
            atmospheric=False  # Phase 3 - enable later
        )

    # Process each AVIRIS file
    stride = args.stride if args.stride is not None else args.patch_size // 2

    patch_files = []
    total_patches = 0

    for i, aviris_file in enumerate(aviris_files):
        print(f"\n[{i+1}/{len(aviris_files)}] Processing: {aviris_file.name}")

        try:
            # Generate patches
            patches = generator.process_aviris_file(
                aviris_file,
                wavelengths=None,  # Will use default
                patch_size=args.patch_size,
                stride=stride,
                output_dir=None  # Don't save yet
            )

            if len(patches) == 0:
                print("  WARNING: No valid patches extracted, skipping")
                continue

            # Apply augmentation if requested
            if args.augment:
                print(f"  Applying augmentation to {len(patches)} patches...")
                wavelengths = load_aviris_wavelengths(
                    n_bands=patches[0]['aviris'].shape[2]
                )

                for patch in tqdm(patches, desc="Augmenting", leave=False):
                    aug_aviris, aug_landsat = augmentation.augment_pair(
                        patch['aviris'],
                        patch['landsat'],
                        wavelengths=wavelengths
                    )
                    patch['aviris'] = aug_aviris
                    patch['landsat'] = aug_landsat

            # Save patches for this file
            output_file = output_dir / f"{aviris_file.stem}_patches.h5"
            generator.save_patches(patches, output_file)

            # Create thumbnails
            thumbnail_dir = output_dir / f"{aviris_file.stem}_thumbnails"
            generator.create_thumbnails(patches, thumbnail_dir, wavelengths=wavelengths)

            patch_files.append(output_file)
            total_patches += len(patches)

        except Exception as e:
            print(f"  ERROR processing {aviris_file.name}: {e}")
            continue

    # Merge datasets if requested
    if args.merge and len(patch_files) > 0:
        merged_file = output_dir / "training_dataset.h5"
        merge_patch_datasets(patch_files, merged_file)

    # Summary
    print("\n" + "=" * 70)
    print("Dataset Generation Summary:")
    print(f"  Processed files: {len(patch_files)} / {len(aviris_files)}")
    print(f"  Total patches: {total_patches}")
    print(f"  Output directory: {output_dir}")
    if args.merge and len(patch_files) > 0:
        print(f"  Merged dataset: {merged_file}")
    print("=" * 70)
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
