#!/usr/bin/env python3
"""
Generate VNIR-only dataset with 5× spatial downsample.

Simplified task:
- Spectral: 5 Landsat bands → 120 AVIRIS VNIR bands (400-1000nm)
- Spatial: 3m → 15m (5× downsample instead of 10×)
- Landsat patches: 51×51 → upsampled to 256×256 for alignment
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import spectral.io.envi as envi


def spectral_binning_vnir(aviris_cube, aviris_wavelengths):
    """
    Bin AVIRIS VNIR bands to 5 Landsat bands.

    Landsat-8 OLI VNIR bands:
    - Band 1 (Coastal): 433-453nm, center 443nm
    - Band 2 (Blue):    452-512nm, center 482nm
    - Band 3 (Green):   533-590nm, center 561nm
    - Band 4 (Red):     636-673nm, center 655nm
    - Band 5 (NIR):     851-879nm, center 865nm
    """

    landsat_bands = [
        (433, 453, 'Coastal'),
        (452, 512, 'Blue'),
        (533, 590, 'Green'),
        (636, 673, 'Red'),
        (851, 879, 'NIR')
    ]

    h, w, _ = aviris_cube.shape
    landsat_cube = np.zeros((h, w, 5), dtype=np.float32)

    for i, (wl_min, wl_max, name) in enumerate(landsat_bands):
        # Find AVIRIS bands within this Landsat band
        mask = (aviris_wavelengths >= wl_min) & (aviris_wavelengths <= wl_max)

        if not np.any(mask):
            print(f"  Warning: No AVIRIS bands found for {name} ({wl_min}-{wl_max}nm)")
            continue

        # Average matching bands
        landsat_cube[:, :, i] = np.mean(aviris_cube[:, :, mask], axis=2)

    return landsat_cube


def spatial_downsample_5x(cube):
    """
    Downsample spatially by 5× (3m → 15m).

    Args:
        cube: (H, W, C) array

    Returns:
        Downsampled cube (H/5, W/5, C)
    """
    h, w, c = cube.shape
    new_h, new_w = h // 5, w // 5

    # Use zoom for each band
    downsampled = np.zeros((new_h, new_w, c), dtype=cube.dtype)

    for i in range(c):
        downsampled[:, :, i] = zoom(cube[:, :, i], 0.2, order=1)  # 1/5 = 0.2, bilinear

    return downsampled


def upsample_bicubic(cube, target_size):
    """
    Upsample back to target size using bicubic interpolation.

    Args:
        cube: (H, W, C) small array
        target_size: Target spatial size (e.g., 256)

    Returns:
        Upsampled cube (target_size, target_size, C)
    """
    h, w, c = cube.shape
    zoom_factor = target_size / h

    upsampled = np.zeros((target_size, target_size, c), dtype=cube.dtype)

    for i in range(c):
        upsampled[:, :, i] = zoom(cube[:, :, i], zoom_factor, order=3)  # cubic

    return upsampled


def generate_dataset(aviris_file, output_file, patch_size=256, stride=128, quality_threshold=0.9):
    """
    Generate VNIR-only dataset with 5× spatial downsample.
    """

    print(f"Processing: {aviris_file}")
    print("="*70)

    # Load AVIRIS data
    img = envi.open(str(aviris_file) + '.hdr', str(aviris_file))
    aviris_data = img.load()

    print(f"  Shape: {aviris_data.shape}")
    print(f"  Bands: {aviris_data.shape[2]}")

    # Get wavelengths
    wavelengths = np.array([float(w) for w in img.metadata['wavelength']])

    # Extract VNIR bands (400-1000nm)
    vnir_mask = (wavelengths >= 400) & (wavelengths <= 1000)
    aviris_vnir = aviris_data[:, :, vnir_mask]
    wavelengths_vnir = wavelengths[vnir_mask]

    print(f"  VNIR bands: {aviris_vnir.shape[2]} (400-1000nm)")
    print(f"  Wavelength range: {wavelengths_vnir.min():.1f} - {wavelengths_vnir.max():.1f} nm")

    # Extract patches
    h, w, _ = aviris_vnir.shape

    patches_aviris = []
    patches_landsat = []

    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1
    total_possible = n_h * n_w

    print(f"\n  Extracting {patch_size}×{patch_size} patches (stride={stride})...")
    print(f"  Grid: {n_h} × {n_w} = {total_possible} patches")

    for i in tqdm(range(0, h - patch_size + 1, stride), desc="Rows"):
        for j in range(0, w - patch_size + 1, stride):
            # Extract AVIRIS VNIR patch
            aviris_patch = aviris_vnir[i:i+patch_size, j:j+patch_size, :]

            # Quality check (skip if too many invalid pixels)
            valid_mask = np.all(aviris_patch > -0.1, axis=2)  # Valid reflectance
            valid_fraction = valid_mask.sum() / (patch_size * patch_size)

            if valid_fraction < quality_threshold:
                continue

            # Normalize to [0, 1] range
            aviris_patch = np.clip(aviris_patch, 0, 1)

            # Create Landsat simulation
            # 1. Spectral binning: 120 VNIR bands → 5 Landsat bands
            landsat_full = spectral_binning_vnir(aviris_patch, wavelengths_vnir)

            # 2. Spatial downsample: 256×256 → 51×51 (5×)
            landsat_small = spatial_downsample_5x(landsat_full)

            # 3. Upsample back to 256×256 (bicubic)
            landsat_patch = upsample_bicubic(landsat_small, patch_size)

            patches_aviris.append(aviris_patch)
            patches_landsat.append(landsat_patch)

    n_patches = len(patches_aviris)
    print(f"\n  Accepted patches: {n_patches} / {total_possible} ({n_patches/total_possible*100:.1f}%)")

    if n_patches == 0:
        print("  ERROR: No valid patches found!")
        return

    # Save to HDF5
    print(f"\n  Saving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Stack patches
        aviris_stack = np.stack(patches_aviris, axis=0)
        landsat_stack = np.stack(patches_landsat, axis=0)

        # Create datasets
        f.create_dataset('aviris', data=aviris_stack, compression='gzip', compression_opts=4)
        f.create_dataset('landsat', data=landsat_stack, compression='gzip', compression_opts=4)
        f.create_dataset('wavelengths', data=wavelengths_vnir)

        # Metadata
        f.attrs['num_patches'] = n_patches
        f.attrs['patch_size'] = patch_size
        f.attrs['stride'] = stride
        f.attrs['quality_threshold'] = quality_threshold
        f.attrs['mode'] = 'VNIR-5band-5x'
        f.attrs['num_bands'] = 5
        f.attrs['aviris_bands_vnir'] = aviris_vnir.shape[2]
        f.attrs['landsat_gsd'] = 15.0  # 3m * 5
        f.attrs['aviris_gsd'] = 3.0
        f.attrs['downsample_factor'] = 5

        file_size_gb = output_file.stat().st_size / (1024**3)
        print(f"  ✓ Saved: {n_patches} patches, {file_size_gb:.2f} GB")
        print(f"  Landsat shape: {landsat_stack.shape}")
        print(f"  AVIRIS shape: {aviris_stack.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate VNIR-only dataset with 5× spatial downsample')
    parser.add_argument('--aviris-file', type=Path, required=True,
                       help='Path to AVIRIS RFL file (without .hdr extension)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/aviris_ng_vnir_5x'),
                       help='Output directory')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--quality-threshold', type=float, default=0.9)

    args = parser.parse_args()

    # Check if file exists
    aviris_file = args.aviris_file
    if not aviris_file.exists():
        print(f"ERROR: AVIRIS file not found: {aviris_file}")
        return

    # Output file - extract flight name from file path
    flight_name = aviris_file.stem.split('_')[0]  # Get ang20230627t232845 from filename
    output_file = args.output_dir / f"{flight_name}_patches_vnir_5x.h5"

    # Generate
    generate_dataset(aviris_file, output_file,
                    args.patch_size, args.stride, args.quality_threshold)

    print("\n✓ Dataset generation complete!")


if __name__ == '__main__':
    main()
