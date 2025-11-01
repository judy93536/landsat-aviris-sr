#!/usr/bin/env python3
"""
Merge all VNIR-5x datasets into a single training file.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def merge_datasets(input_dir, output_file):
    """Merge all VNIR-5x HDF5 files into one."""

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    # Find all VNIR-5x datasets
    h5_files = sorted(input_dir.glob("*_patches_vnir_5x.h5"))

    print("="*70)
    print("Merging VNIR-5x Datasets")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"\nFound {len(h5_files)} datasets:")

    # Count total patches
    total_patches = 0
    file_info = []

    for f in h5_files:
        with h5py.File(f, 'r') as hf:
            n_patches = hf.attrs['num_patches']
            total_patches += n_patches
            file_info.append((f.name, n_patches))
            print(f"  {f.name}: {n_patches} patches")

    print(f"\nTotal patches: {total_patches}")

    # Load first file to get metadata
    with h5py.File(h5_files[0], 'r') as f:
        wavelengths = f['wavelengths'][:]
        sample_landsat = f['landsat'][0]
        sample_aviris = f['aviris'][0]
        patch_size = f.attrs['patch_size']
        num_bands = f.attrs['num_bands']
        aviris_bands_vnir = f.attrs['aviris_bands_vnir']

    print(f"\nDataset properties:")
    print(f"  Patch size: {patch_size}×{patch_size}")
    print(f"  Landsat bands: {num_bands}")
    print(f"  AVIRIS VNIR bands: {aviris_bands_vnir}")
    print(f"  Landsat shape: {sample_landsat.shape}")
    print(f"  AVIRIS shape: {sample_aviris.shape}")

    # Create merged file
    print(f"\nMerging datasets...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as out_f:
        # Create datasets with appropriate shapes
        landsat_shape = (total_patches, patch_size, patch_size, num_bands)
        aviris_shape = (total_patches, patch_size, patch_size, aviris_bands_vnir)

        landsat_dset = out_f.create_dataset(
            'landsat', shape=landsat_shape, dtype=np.float32,
            compression='gzip', compression_opts=4
        )

        aviris_dset = out_f.create_dataset(
            'aviris', shape=aviris_shape, dtype=np.float32,
            compression='gzip', compression_opts=4
        )

        # Copy wavelengths
        out_f.create_dataset('wavelengths', data=wavelengths)

        # Merge all files
        idx = 0
        for h5_file in tqdm(h5_files, desc="Merging files"):
            with h5py.File(h5_file, 'r') as in_f:
                n = in_f.attrs['num_patches']

                # Copy data
                landsat_dset[idx:idx+n] = in_f['landsat'][:]
                aviris_dset[idx:idx+n] = in_f['aviris'][:]

                idx += n

        # Save metadata
        out_f.attrs['num_patches'] = total_patches
        out_f.attrs['patch_size'] = patch_size
        out_f.attrs['num_bands'] = num_bands
        out_f.attrs['aviris_bands_vnir'] = aviris_bands_vnir
        out_f.attrs['mode'] = 'VNIR-5band-5x-merged'
        out_f.attrs['landsat_gsd'] = 15.0
        out_f.attrs['aviris_gsd'] = 3.0
        out_f.attrs['downsample_factor'] = 5
        out_f.attrs['num_source_files'] = len(h5_files)
        out_f.attrs['source_files'] = ','.join([f.name for f in h5_files])

    file_size_gb = output_file.stat().st_size / (1024**3)
    print(f"\n{'='*70}")
    print("Merge complete!")
    print(f"  Output: {output_file}")
    print(f"  Total patches: {total_patches}")
    print(f"  File size: {file_size_gb:.2f} GB")
    print(f"  Landsat shape: {landsat_shape}")
    print(f"  AVIRIS shape: {aviris_shape}")
    print("="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Merge VNIR-5x datasets')
    parser.add_argument('--input-dir', type=Path,
                       default=Path('outputs/aviris_ng_vnir_5x'),
                       help='Directory containing VNIR-5x datasets')
    parser.add_argument('--output-file', type=Path,
                       default=Path('outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5'),
                       help='Output merged file')

    args = parser.parse_args()

    merge_datasets(args.input_dir, args.output_file)
