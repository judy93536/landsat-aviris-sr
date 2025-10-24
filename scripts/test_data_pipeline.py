#!/usr/bin/env python3
"""
Test script for the synthetic data generation pipeline.

Usage:
    python scripts/test_data_pipeline.py [--aviris-file PATH] [--output-dir PATH]

This script tests the complete pipeline:
1. Load AVIRIS data (or generate synthetic)
2. Generate synthetic Landsat
3. Extract patches
4. Apply augmentation
5. Save results
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.landsat_srf import LandsatSRF, load_aviris_wavelengths
from data.synthetic import SyntheticDataGenerator, load_patch_dataset
from data.augmentation import AugmentationPipeline


def create_test_aviris_data(height=512, width=512, n_bands=224):
    """
    Create synthetic AVIRIS-like test data with realistic spectral signatures.

    Returns:
    --------
    aviris_cube : np.ndarray
        Synthetic AVIRIS data (shape: [height, width, n_bands])
    wavelengths : np.ndarray
        Wavelength array
    """
    print(f"Creating synthetic AVIRIS test data: {height}x{width}x{n_bands}")

    wavelengths = load_aviris_wavelengths(n_bands=n_bands)
    aviris_cube = np.zeros((height, width, n_bands), dtype=np.float32)

    # Create different land cover types with characteristic spectra
    for i in range(height):
        for j in range(width):
            # Determine land cover type based on position
            if i < height // 3:
                # Vegetation: low red, high NIR
                spectrum = np.ones(n_bands) * 300
                red_idx = np.abs(wavelengths - 650).argmin()
                nir_idx = np.abs(wavelengths - 850).argmin()
                spectrum[red_idx:red_idx+20] *= 0.4  # Red absorption
                spectrum[nir_idx:] *= 2.0  # NIR plateau

            elif i < 2 * height // 3:
                # Soil: gradual increase to SWIR
                spectrum = 200 + (wavelengths - wavelengths[0]) / 10
                spectrum += np.random.randn(n_bands) * 20

            else:
                # Water: absorption in NIR/SWIR
                spectrum = np.ones(n_bands) * 500
                nir_idx = np.abs(wavelengths - 850).argmin()
                spectrum[nir_idx:] *= 0.1

            # Add noise
            spectrum += np.random.randn(n_bands) * 30

            aviris_cube[i, j, :] = np.clip(spectrum, 0, None)

    print(f"  Data range: {aviris_cube.min():.1f} - {aviris_cube.max():.1f}")
    print(f"  Data mean: {aviris_cube.mean():.1f}")

    return aviris_cube, wavelengths


def visualize_results(aviris_cube, synthetic_landsat, wavelengths, output_dir=None):
    """
    Create visualization of results.

    Parameters:
    -----------
    aviris_cube : np.ndarray
        AVIRIS data
    synthetic_landsat : np.ndarray
        Synthetic Landsat data
    wavelengths : np.ndarray
        Wavelength array
    output_dir : Path or None
        Directory to save plots
    """
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # AVIRIS RGB composite (using bands closest to R, G, B)
    r_idx = np.abs(wavelengths - 650).argmin()
    g_idx = np.abs(wavelengths - 550).argmin()
    b_idx = np.abs(wavelengths - 450).argmin()

    aviris_rgb = np.stack([
        aviris_cube[:, :, r_idx],
        aviris_cube[:, :, g_idx],
        aviris_cube[:, :, b_idx]
    ], axis=2)
    aviris_rgb = (aviris_rgb - aviris_rgb.min()) / (aviris_rgb.max() - aviris_rgb.min())

    axes[0, 0].imshow(aviris_rgb)
    axes[0, 0].set_title("AVIRIS RGB (4m)")
    axes[0, 0].axis('off')

    # Synthetic Landsat RGB (bands 3, 2, 1 for LC08)
    if synthetic_landsat.shape[2] >= 3:
        landsat_rgb = np.stack([
            synthetic_landsat[:, :, 3],  # Red
            synthetic_landsat[:, :, 2],  # Green
            synthetic_landsat[:, :, 1]   # Blue
        ], axis=2)
        landsat_rgb = (landsat_rgb - landsat_rgb.min()) / (landsat_rgb.max() - landsat_rgb.min())

        axes[0, 1].imshow(landsat_rgb)
        axes[0, 1].set_title("Synthetic Landsat RGB (30m)")
        axes[0, 1].axis('off')

    # Spectral signatures from center pixel
    center_y, center_x = aviris_cube.shape[0] // 2, aviris_cube.shape[1] // 2
    aviris_spectrum = aviris_cube[center_y, center_x, :]

    axes[0, 2].plot(wavelengths, aviris_spectrum, linewidth=0.5)
    axes[0, 2].set_xlabel("Wavelength (nm)")
    axes[0, 2].set_ylabel("Radiance")
    axes[0, 2].set_title("AVIRIS Spectrum (center pixel)")
    axes[0, 2].grid(True, alpha=0.3)

    # Landsat bands overlaid
    srf = LandsatSRF(sensor="LC08")
    landsat_bands_info = srf.get_all_bands_info()

    center_y_lr = synthetic_landsat.shape[0] // 2
    center_x_lr = synthetic_landsat.shape[1] // 2

    for band_num in range(min(7, synthetic_landsat.shape[2])):
        band_info = landsat_bands_info[band_num + 1]
        value = synthetic_landsat[center_y_lr, center_x_lr, band_num]
        axes[1, 0].bar(
            band_num,
            value,
            label=f"B{band_num+1}: {band_info['name']}"
        )

    axes[1, 0].set_xlabel("Band Number")
    axes[1, 0].set_ylabel("Radiance")
    axes[1, 0].set_title("Synthetic Landsat Bands")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Statistical comparison
    axes[1, 1].hist(aviris_cube.flatten(), bins=100, alpha=0.5, label="AVIRIS", density=True)
    axes[1, 1].hist(synthetic_landsat.flatten(), bins=100, alpha=0.5, label="Landsat", density=True)
    axes[1, 1].set_xlabel("Radiance")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Radiance Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # SRF plot
    srf_wavelengths = wavelengths
    axes[1, 2].set_xlabel("Wavelength (nm)")
    axes[1, 2].set_ylabel("Spectral Response")
    axes[1, 2].set_title("Landsat-8 SRFs")

    for band_num in [1, 2, 3, 4, 5, 6, 7]:
        srf_values = srf.get_srf(band_num, srf_wavelengths)
        band_info = landsat_bands_info[band_num]
        axes[1, 2].plot(
            srf_wavelengths,
            srf_values,
            label=f"B{band_num}",
            linewidth=1
        )

    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir) / "pipeline_test_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
    else:
        print("  Visualization created (not saved)")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Test synthetic data generation pipeline"
    )
    parser.add_argument(
        "--aviris-file",
        type=str,
        default=None,
        help="Path to AVIRIS data file (.h5, .npy). If not provided, creates synthetic data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test",
        help="Output directory for results"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size at AVIRIS resolution"
    )
    parser.add_argument(
        "--test-augmentation",
        action="store_true",
        help="Test augmentation pipeline"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Landsat-AVIRIS Synthetic Data Pipeline Test")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create AVIRIS data
    if args.aviris_file is not None:
        print(f"\nLoading AVIRIS data from: {args.aviris_file}")
        generator = SyntheticDataGenerator()
        aviris_cube = generator.load_aviris(args.aviris_file)
        wavelengths = load_aviris_wavelengths(n_bands=aviris_cube.shape[2])
    else:
        print("\nNo AVIRIS file provided, creating synthetic test data...")
        aviris_cube, wavelengths = create_test_aviris_data()

    # Initialize generator
    print("\nInitializing synthetic data generator...")
    generator = SyntheticDataGenerator(
        spatial_downsampling_factor=7.5,
        psf_sigma=2.0,
        landsat_sensor="LC08"
    )

    # Generate synthetic Landsat
    print("\nGenerating synthetic Landsat...")
    synthetic_landsat = generator.generate_synthetic_landsat(
        aviris_cube,
        wavelengths
    )

    # Extract patches
    print("\nExtracting patches...")
    patches = generator.extract_patches(
        aviris_cube,
        synthetic_landsat,
        patch_size_hr=args.patch_size,
        stride=args.patch_size // 2  # 50% overlap
    )

    # Save patches
    if len(patches) > 0:
        patch_file = output_dir / "test_patches.h5"
        generator.save_patches(patches, patch_file)

        # Test loading
        print("\nTesting patch loading...")
        aviris_patches, landsat_patches, metadata = load_patch_dataset(patch_file)
        print(f"  Loaded {aviris_patches.shape[0]} patches")
        print(f"  AVIRIS patch shape: {aviris_patches.shape[1:]}")
        print(f"  Landsat patch shape: {landsat_patches.shape[1:]}")

    # Test augmentation
    if args.test_augmentation and len(patches) > 0:
        print("\nTesting augmentation pipeline...")
        pipeline = AugmentationPipeline(
            geometric=True,
            noise=True,
            artifacts=True,
            atmospheric=False
        )

        test_patch = patches[0]
        aug_aviris, aug_landsat = pipeline.augment_pair(
            test_patch['aviris'],
            test_patch['landsat'],
            wavelengths=wavelengths
        )

        print(f"  Original Landsat mean: {test_patch['landsat'].mean():.2f}")
        print(f"  Augmented Landsat mean: {aug_landsat.mean():.2f}")

    # Create visualization
    print("\nGenerating visualization...")
    fig = visualize_results(
        aviris_cube,
        synthetic_landsat,
        wavelengths,
        output_dir=output_dir
    )

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  AVIRIS shape: {aviris_cube.shape}")
    print(f"  Synthetic Landsat shape: {synthetic_landsat.shape}")
    print(f"  Number of patches: {len(patches)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)
    print("\nTest complete!")


if __name__ == "__main__":
    main()
