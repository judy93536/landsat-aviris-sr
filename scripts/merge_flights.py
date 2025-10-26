"""
Merge multiple AVIRIS Classic flight HDF5 files into one combined dataset.

Use on Lambda Labs after downloading individual flights from S3.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


def merge_flights(input_dir, output_file):
    """
    Merge all .h5 files in input_dir into one combined dataset.

    Parameters:
    -----------
    input_dir : str
        Directory containing individual flight .h5 files
    output_file : str
        Output path for merged dataset
    """
    print("="*70)
    print("Merging AVIRIS Classic Flights")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")

    # Find all .h5 files
    input_path = Path(input_dir)
    h5_files = sorted(input_path.glob("*.h5"))

    if len(h5_files) == 0:
        print(f"\nERROR: No .h5 files found in {input_dir}")
        return

    print(f"\nFound {len(h5_files)} files:")
    for f in h5_files:
        print(f"  - {f.name}")

    # Collect all patches
    all_landsat = []
    all_aviris = []
    all_locations = []
    all_qualities = []
    flight_sources = []

    total_patches = 0

    print("\nLoading patches from each flight...")
    for h5_file in h5_files:
        print(f"\n  {h5_file.name}...")

        try:
            with h5py.File(h5_file, 'r') as f:
                landsat = f['landsat'][:]
                aviris = f['aviris'][:]
                locations = f['locations'][:]
                qualities = f['qualities'][:]

                n_patches = len(aviris)
                print(f"    Loaded {n_patches} patches")
                print(f"    Landsat: {landsat.shape}, AVIRIS: {aviris.shape}")

                all_landsat.append(landsat)
                all_aviris.append(aviris)
                all_locations.append(locations)
                all_qualities.append(qualities)

                # Track which flight each patch came from
                flight_name = f.attrs.get('flight_name', h5_file.stem)
                flight_sources.extend([flight_name] * n_patches)

                total_patches += n_patches

        except Exception as e:
            print(f"    ERROR loading {h5_file.name}: {e}")
            continue

    if total_patches == 0:
        print("\nERROR: No patches loaded!")
        return

    print(f"\n{'='*70}")
    print(f"Total patches to merge: {total_patches}")
    print(f"{'='*70}")

    # Concatenate all arrays
    print("\nConcatenating arrays...")
    combined_landsat = np.concatenate(all_landsat, axis=0)
    combined_aviris = np.concatenate(all_aviris, axis=0)
    combined_locations = np.concatenate(all_locations, axis=0)
    combined_qualities = np.concatenate(all_qualities, axis=0)

    print(f"  Combined Landsat: {combined_landsat.shape}")
    print(f"  Combined AVIRIS: {combined_aviris.shape}")
    print(f"  Total patches: {len(combined_aviris)}")

    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}...")

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Save datasets with compression
        f.create_dataset('landsat', data=combined_landsat,
                        compression='gzip', compression_opts=4)
        f.create_dataset('aviris', data=combined_aviris,
                        compression='gzip', compression_opts=4)
        f.create_dataset('locations', data=combined_locations)
        f.create_dataset('qualities', data=combined_qualities)

        # Save flight sources as strings (variable length)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('flight_sources', data=flight_sources, dtype=dt)

        # Metadata
        f.attrs['n_patches'] = total_patches
        f.attrs['n_flights'] = len(h5_files)
        f.attrs['patch_size'] = 256
        f.attrs['landsat_bands'] = 7
        f.attrs['aviris_bands'] = 198
        f.attrs['landsat_gsd'] = 30.0
        f.attrs['aviris_gsd'] = 14.8
        f.attrs['merge_time'] = datetime.now().isoformat()
        f.attrs['source_files'] = [str(f.name) for f in h5_files]

        # Data statistics
        f.attrs['landsat_min'] = float(combined_landsat.min())
        f.attrs['landsat_max'] = float(combined_landsat.max())
        f.attrs['aviris_min'] = float(combined_aviris.min())
        f.attrs['aviris_max'] = float(combined_aviris.max())
        f.attrs['mean_quality'] = float(np.mean(combined_qualities))
        f.attrs['min_quality'] = float(np.min(combined_qualities))

    print(f"\n✓ Successfully saved merged dataset")
    print(f"\nDataset Summary:")
    print(f"  Total patches: {total_patches}")
    print(f"  From {len(h5_files)} flights")
    print(f"  Mean quality: {np.mean(combined_qualities):.4f}")
    print(f"  Landsat range: [{combined_landsat.min():.2f}, {combined_landsat.max():.2f}]")
    print(f"  AVIRIS range: [{combined_aviris.min():.2f}, {combined_aviris.max():.2f}]")

    # Calculate file size
    file_size_gb = Path(output_file).stat().st_size / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple flight HDF5 files into one dataset'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing individual flight .h5 files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged .h5 file path')

    args = parser.parse_args()

    merge_flights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
