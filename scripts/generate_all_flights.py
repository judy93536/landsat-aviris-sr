"""
Generate training dataset from all AVIRIS Classic flights.

Processes each flight separately and saves to individual .h5 files.
Applies strict quality control to handle known data issues.
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
from datetime import datetime
import sys
from tqdm import tqdm
import scipy.ndimage as ndimage

sys.path.append(str(Path(__file__).parent.parent))
from data.aviris_classic_loader import AVIRISClassicImage


def landsat_spectral_bins():
    """Define Landsat-8 OLI spectral bands (in nm)."""
    return {
        'coastal': (430, 450),    # Band 1
        'blue': (450, 510),        # Band 2
        'green': (530, 590),       # Band 3
        'red': (640, 670),         # Band 4
        'nir': (850, 880),         # Band 5
        'swir1': (1570, 1650),     # Band 6
        'swir2': (2110, 2290),     # Band 7
    }


def bin_to_landsat_bands(cube, wavelengths):
    """
    Bin AVIRIS bands to 7 Landsat equivalent bands.

    Parameters:
    -----------
    cube : np.ndarray
        (H, W, 198) AVIRIS cube
    wavelengths : np.ndarray
        (198,) wavelengths in nm

    Returns:
    --------
    landsat_7band : np.ndarray
        (H, W, 7) Landsat equivalent
    """
    bands = landsat_spectral_bins()
    h, w, _ = cube.shape
    landsat = np.zeros((h, w, 7), dtype=np.float32)

    for i, (band_name, (wl_min, wl_max)) in enumerate(bands.items()):
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if mask.sum() > 0:
            landsat[:, :, i] = cube[:, :, mask].mean(axis=2)
        else:
            # No bands in range - use nearest
            nearest_idx = np.argmin(np.abs(wavelengths - (wl_min + wl_max) / 2))
            landsat[:, :, i] = cube[:, :, nearest_idx]

    return landsat


def simulate_landsat(aviris_cube, wavelengths, aviris_gsd=14.8, landsat_gsd=30.0):
    """
    Simulate Landsat from AVIRIS by:
    1. Spectral binning (198 → 7 bands)
    2. Spatial downsampling (14.8m → 30m)
    3. Upsampling back to original size for alignment

    Parameters:
    -----------
    aviris_cube : np.ndarray
        (H, W, 198) AVIRIS reflectance
    wavelengths : np.ndarray
        (198,) wavelengths
    aviris_gsd : float
        AVIRIS ground sampling distance (m)
    landsat_gsd : float
        Landsat ground sampling distance (m)

    Returns:
    --------
    landsat_sim : np.ndarray
        (H, W, 7) simulated Landsat at original spatial resolution
    """
    h, w, _ = aviris_cube.shape

    # 1. Spectral binning
    landsat_7band = bin_to_landsat_bands(aviris_cube, wavelengths)

    # 2. Spatial downsampling
    zoom_factor = aviris_gsd / landsat_gsd  # ~0.493
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)

    downsampled = np.zeros((new_h, new_w, 7), dtype=np.float32)

    for i in range(7):
        zoom_h = new_h / h
        zoom_w = new_w / w
        downsampled[:, :, i] = ndimage.zoom(
            landsat_7band[:, :, i],
            (zoom_h, zoom_w),
            order=1  # Bilinear
        )

    # 3. Upsample back to original size for pixel alignment
    upsampled = np.zeros((h, w, 7), dtype=np.float32)

    for i in range(7):
        zoom_h = h / new_h
        zoom_w = w / new_w
        upsampled[:, :, i] = ndimage.zoom(
            downsampled[:, :, i],
            (zoom_h, zoom_w),
            order=3  # Bicubic
        )

    return upsampled


def assess_patch_quality(patch, min_quality=0.85):
    """
    Assess patch quality with strict criteria.

    Rejects patches with:
    - ANY extreme outliers (< -2 or > 2)
    - More than 10% moderate negatives (< -0.5)
    - More than 5% high values (> 1.5)

    Parameters:
    -----------
    patch : np.ndarray
        (H, W, C) patch
    min_quality : float
        Minimum acceptable quality score

    Returns:
    --------
    quality : float
        Quality score [0, 1]
    """
    # Check for extreme outliers (STRICT: must be zero)
    extreme_neg = (patch < -2).sum()
    extreme_high = (patch > 2).sum()

    if extreme_neg > 0 or extreme_high > 0:
        return 0.0  # Reject immediately

    total_pixels = patch.size

    # Moderate quality issues
    moderate_neg = (patch < -0.5).sum()
    moderate_neg_frac = moderate_neg / total_pixels

    high_values = (patch > 1.5).sum()
    high_values_frac = high_values / total_pixels

    # Check thresholds
    if moderate_neg_frac > 0.10:  # Max 10% moderate negatives
        return 0.0

    if high_values_frac > 0.05:  # Max 5% high values
        return 0.0

    # Compute quality score
    valid_range = ((patch >= -0.5) & (patch <= 1.5)).sum()
    quality = valid_range / total_pixels

    return quality


def extract_patches_from_flight(aviris_file, patch_size=256, stride=None,
                                quality_threshold=0.90, max_patches=None):
    """
    Extract valid patches from a single flight.

    Parameters:
    -----------
    aviris_file : str
        Path to AVIRIS Classic *_corr_*_img file
    patch_size : int
        Patch size (default 256)
    stride : int
        Stride for patch extraction (default: patch_size // 2 for 50% overlap)
    quality_threshold : float
        Minimum quality to accept patch
    max_patches : int
        Maximum patches to extract (None = all valid)

    Returns:
    --------
    patches : dict
        {'landsat': list, 'aviris': list, 'locations': list, 'qualities': list}
    """
    print(f"\nProcessing: {aviris_file}")

    # Load AVIRIS Classic data
    loader = AVIRISClassicImage(aviris_file)
    good_bands = loader.get_good_bands()
    cube = loader.read_cube(bands=good_bands)
    wavelengths = loader.get_wavelengths(bands=good_bands)

    print(f"  Loaded cube: {cube.shape}")
    print(f"  Using {len(wavelengths)} bands")
    print(f"  Data range: [{cube.min():.2f}, {cube.max():.2f}]")

    # Simulate Landsat
    print("  Simulating Landsat...")
    landsat = simulate_landsat(cube, wavelengths)

    # Extract patches
    if stride is None:
        stride = patch_size // 2  # 50% overlap

    h, w, _ = cube.shape
    patches = {
        'landsat': [],
        'aviris': [],
        'locations': [],
        'qualities': []
    }

    print(f"  Extracting {patch_size}×{patch_size} patches (stride={stride})...")

    total_attempts = 0
    valid_patches = 0

    # Use tqdm for progress
    y_positions = list(range(0, h - patch_size + 1, stride))

    for y in tqdm(y_positions, desc="  Scanning rows"):
        for x in range(0, w - patch_size + 1, stride):
            total_attempts += 1

            # Extract patches
            aviris_patch = cube[y:y+patch_size, x:x+patch_size, :]
            landsat_patch = landsat[y:y+patch_size, x:x+patch_size, :]

            # Assess quality
            quality = assess_patch_quality(aviris_patch, min_quality=quality_threshold)

            if quality >= quality_threshold:
                patches['aviris'].append(aviris_patch)
                patches['landsat'].append(landsat_patch)
                patches['locations'].append((y, x))
                patches['qualities'].append(quality)
                valid_patches += 1

                # Check max patches limit
                if max_patches and valid_patches >= max_patches:
                    print(f"  Reached max patches limit ({max_patches})")
                    break

        if max_patches and valid_patches >= max_patches:
            break

    print(f"  Results: {valid_patches}/{total_attempts} valid patches "
          f"({100*valid_patches/max(total_attempts,1):.1f}% acceptance)")

    if valid_patches > 0:
        print(f"  Quality: mean={np.mean(patches['qualities']):.4f}, "
              f"min={np.min(patches['qualities']):.4f}")

    return patches, wavelengths


def save_flight_dataset(patches, wavelengths, output_file, flight_name):
    """
    Save patches to HDF5 file.

    Parameters:
    -----------
    patches : dict
        Extracted patches
    wavelengths : np.ndarray
        Wavelengths array
    output_file : str
        Output .h5 file path
    flight_name : str
        Flight identifier
    """
    n_patches = len(patches['aviris'])

    if n_patches == 0:
        print(f"  WARNING: No valid patches found for {flight_name}")
        return

    print(f"  Saving {n_patches} patches to {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Convert lists to arrays
        landsat_array = np.array(patches['landsat'], dtype=np.float32)
        aviris_array = np.array(patches['aviris'], dtype=np.float32)
        locations_array = np.array(patches['locations'], dtype=np.int32)
        qualities_array = np.array(patches['qualities'], dtype=np.float32)

        # Save datasets
        f.create_dataset('landsat', data=landsat_array, compression='gzip', compression_opts=4)
        f.create_dataset('aviris', data=aviris_array, compression='gzip', compression_opts=4)
        f.create_dataset('locations', data=locations_array)
        f.create_dataset('qualities', data=qualities_array)

        # Metadata
        f.attrs['n_patches'] = n_patches
        f.attrs['patch_size'] = 256
        f.attrs['landsat_bands'] = 7
        f.attrs['aviris_bands'] = 198
        f.attrs['landsat_gsd'] = 30.0
        f.attrs['aviris_gsd'] = 14.8
        f.attrs['wavelengths'] = wavelengths
        f.attrs['flight_name'] = flight_name
        f.attrs['mean_quality'] = np.mean(qualities_array)
        f.attrs['generation_time'] = datetime.now().isoformat()

        # Data statistics
        f.attrs['landsat_min'] = landsat_array.min()
        f.attrs['landsat_max'] = landsat_array.max()
        f.attrs['aviris_min'] = aviris_array.min()
        f.attrs['aviris_max'] = aviris_array.max()

    print(f"  ✓ Saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Generate datasets from all AVIRIS Classic flights')
    parser.add_argument('--aviris-root', type=str,
                       default='/iex_data/AVIRIS_Classic',
                       help='Root directory containing all flights')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/full_dataset/flights',
                       help='Output directory for individual flight .h5 files')
    parser.add_argument('--patch-size', type=int, default=256,
                       help='Patch size')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride (default: patch_size // 2)')
    parser.add_argument('--quality-threshold', type=float, default=0.90,
                       help='Minimum quality threshold')
    parser.add_argument('--max-patches-per-flight', type=int, default=None,
                       help='Max patches per flight (None = all valid)')

    args = parser.parse_args()

    print("="*70)
    print("AVIRIS Classic Full Dataset Generation")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Root: {args.aviris_root}")
    print(f"  Output: {args.output_dir}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Stride: {args.stride or args.patch_size // 2}")
    print(f"  Quality threshold: {args.quality_threshold}")
    print(f"  Max per flight: {args.max_patches_per_flight or 'unlimited'}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all flights - look for *_corr_*_img files
    aviris_root = Path(args.aviris_root)
    flights = []

    for parent_dir in sorted(aviris_root.glob('f*')):
        if parent_dir.is_dir():
            # Look for *_corr_*_img files in _rfl_* subdirectories
            corr_files = list(parent_dir.glob('*_rfl_*/*_corr_*_img'))
            if corr_files:
                flights.append(corr_files[0])  # Take first match

    print(f"\nFound {len(flights)} flights:")
    for flight in flights:
        # Get parent directory name (flight ID)
        flight_id = flight.parent.parent.name
        print(f"  - {flight_id}")

    # Process each flight
    print("\n" + "="*70)
    print("Processing flights...")
    print("="*70)

    summary = []

    for i, flight_file in enumerate(flights, 1):
        # flight_file is the *_corr_*_img file
        flight_id = flight_file.parent.parent.name
        print(f"\n[{i}/{len(flights)}] Flight: {flight_id}")

        try:
            # Extract patches
            patches, wavelengths = extract_patches_from_flight(
                str(flight_file),
                patch_size=args.patch_size,
                stride=args.stride,
                quality_threshold=args.quality_threshold,
                max_patches=args.max_patches_per_flight
            )

            # Save to .h5 using parent directory name (flight ID)
            output_file = output_dir / f'{flight_id}_patches.h5'
            save_flight_dataset(patches, wavelengths, str(output_file), flight_id)

            # Track summary
            summary.append({
                'flight': flight_id,
                'patches': len(patches['aviris']),
                'mean_quality': np.mean(patches['qualities']) if patches['qualities'] else 0,
                'file': output_file.name
            })

        except Exception as e:
            print(f"  ERROR processing {flight_id}: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                'flight': flight_id,
                'patches': 0,
                'mean_quality': 0,
                'file': 'FAILED'
            })

    # Print summary
    print("\n" + "="*70)
    print("Generation Complete!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSummary:")
    print(f"{'Flight':<25} {'Patches':>10} {'Quality':>10} {'File':<30}")
    print("-"*70)

    total_patches = 0
    for item in summary:
        print(f"{item['flight']:<25} {item['patches']:>10} "
              f"{item['mean_quality']:>10.4f} {item['file']:<30}")
        total_patches += item['patches']

    print("-"*70)
    print(f"{'TOTAL':<25} {total_patches:>10}")
    print("\nAll individual flight .h5 files saved to:")
    print(f"  {output_dir}")
    print("\nNext step: Merge these files with scripts/merge_datasets.py")
    print("="*70)


if __name__ == "__main__":
    main()
