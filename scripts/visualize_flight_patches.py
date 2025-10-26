"""
Visualize AVIRIS Flight with Patch Locations and Diversity Analysis

Creates RGB preview of entire scene at 1/2 resolution with patch locations overlaid,
plus statistics on patch diversity to help decide if we need all patches.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.aviris_classic_loader import AVIRISClassicImage


def compute_patch_diversity(patches_file):
    """
    Compute diversity statistics for patches.

    Returns:
    --------
    stats : dict
        Diversity metrics (mean, std, spectral variance, spatial content, etc.)
    """
    with h5py.File(patches_file, 'r') as f:
        aviris = f['aviris'][:]
        locations = f['locations'][:]
        qualities = f['qualities'][:]

    n_patches = len(aviris)

    # Compute per-patch statistics
    patch_means = np.mean(aviris, axis=(1, 2))  # (N, 198)
    patch_stds = np.std(aviris, axis=(1, 2))    # (N, 198)

    # Overall spectral mean for each patch
    patch_spectral_mean = np.mean(patch_means, axis=1)  # (N,)
    patch_spectral_std = np.mean(patch_stds, axis=1)    # (N,)

    # Spatial content (edge energy as proxy for detail)
    spatial_energy = []
    for i in range(n_patches):
        # Use NIR band for spatial content
        nir_band = aviris[i, :, :, 100]  # Approximate NIR
        # Compute gradient magnitude
        gy, gx = np.gradient(nir_band)
        energy = np.sqrt(gx**2 + gy**2).mean()
        spatial_energy.append(energy)

    spatial_energy = np.array(spatial_energy)

    # Compute pairwise distances between patches (sample subset if too many)
    n_sample = min(200, n_patches)
    indices = np.random.choice(n_patches, n_sample, replace=False)
    sampled_means = patch_means[indices]

    # Spectral angle distance
    from scipy.spatial.distance import pdist
    spectral_distances = pdist(sampled_means, metric='cosine')

    stats = {
        'n_patches': n_patches,
        'mean_reflectance': np.mean(patch_spectral_mean),
        'std_reflectance': np.std(patch_spectral_mean),
        'mean_spatial_energy': np.mean(spatial_energy),
        'std_spatial_energy': np.std(spatial_energy),
        'mean_spectral_distance': np.mean(spectral_distances),
        'std_spectral_distance': np.std(spectral_distances),
        'quality_mean': np.mean(qualities),
        'quality_min': np.min(qualities),
        'locations': locations,
        'patch_spectral_mean': patch_spectral_mean,
        'patch_spatial_energy': spatial_energy,
        'qualities': qualities,
    }

    return stats


def visualize_flight_with_patches(aviris_file, patches_file, output_path, downsample=2):
    """
    Create RGB visualization of full flight with patch locations overlaid.

    Parameters:
    -----------
    aviris_file : str
        Path to AVIRIS *_corr_*_img file
    patches_file : str
        Path to extracted patches .h5 file
    output_path : str
        Output image path
    downsample : int
        Downsampling factor for visualization (2 = half resolution)
    """
    print(f"\nLoading AVIRIS scene from {aviris_file}...")

    # Load AVIRIS data
    loader = AVIRISClassicImage(aviris_file)
    good_bands = loader.get_good_bands()
    cube = loader.read_cube(bands=good_bands)
    wavelengths = loader.get_wavelengths(bands=good_bands)

    print(f"  Scene size: {cube.shape}")

    # Downsample for visualization
    h, w, c = cube.shape
    new_h = h // downsample
    new_w = w // downsample

    print(f"  Downsampling to {new_h} x {new_w} for visualization...")

    # Simple downsampling by averaging
    cube_down = cube[::downsample, ::downsample, :]

    # Create RGB (approximate R, G, B bands)
    rgb_indices = []
    target_wavelengths = [640, 550, 470]  # R, G, B

    for target_wl in target_wavelengths:
        idx = np.argmin(np.abs(wavelengths - target_wl))
        rgb_indices.append(idx)

    print(f"  RGB bands: {wavelengths[rgb_indices]}")

    rgb = cube_down[:, :, rgb_indices]

    # Normalize for display (2-98 percentile stretch)
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    # Load patch statistics
    print(f"\nComputing patch diversity statistics...")
    stats = compute_patch_diversity(patches_file)

    print(f"\nPatch Diversity Analysis:")
    print(f"  Total patches: {stats['n_patches']}")
    print(f"  Mean reflectance: {stats['mean_reflectance']:.4f} ± {stats['std_reflectance']:.4f}")
    print(f"  Spatial energy: {stats['mean_spatial_energy']:.4f} ± {stats['std_spatial_energy']:.4f}")
    print(f"  Spectral diversity: {stats['mean_spectral_distance']:.4f} ± {stats['std_spectral_distance']:.4f}")
    print(f"  Quality: {stats['quality_mean']:.4f} (min: {stats['quality_min']:.4f})")

    # Create visualization
    print(f"\nGenerating visualization...")

    fig = plt.figure(figsize=(20, 10))

    # Left: Full scene RGB with patch locations
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title(f'AVIRIS Classic Scene (1/{downsample} resolution)\n{h}x{w} original',
                  fontsize=12, fontweight='bold')

    # Overlay patch locations
    patch_size = 256
    for loc in stats['locations']:
        y, x = loc
        # Scale to downsampled coordinates
        rect = Rectangle((x / downsample, y / downsample),
                        patch_size / downsample, patch_size / downsample,
                        linewidth=0.5, edgecolor='yellow', facecolor='none', alpha=0.6)
        ax1.add_patch(rect)

    ax1.set_xlabel(f'{stats["n_patches"]} patches extracted (256×256, stride=128)', fontsize=10)
    ax1.axis('off')

    # Middle: Patch diversity scatter (reflectance vs spatial content)
    ax2 = plt.subplot(1, 3, 2)
    scatter = ax2.scatter(stats['patch_spectral_mean'], stats['patch_spatial_energy'],
                         c=stats['qualities'], cmap='viridis', s=20, alpha=0.6)
    ax2.set_xlabel('Mean Reflectance', fontsize=10)
    ax2.set_ylabel('Spatial Content (Edge Energy)', fontsize=10)
    ax2.set_title('Patch Diversity\n(color = quality)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Quality')

    # Right: Histograms
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(stats['patch_spectral_mean'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Mean Reflectance', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.set_title('Reflectance Distribution', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 6)
    ax4.hist(stats['patch_spatial_energy'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax4.set_xlabel('Spatial Energy', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('Spatial Content Distribution', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Flight Scene Overview and Patch Diversity Analysis',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Visualize AVIRIS flight scene with patch locations and diversity analysis'
    )
    parser.add_argument('--aviris-file', type=str, required=True,
                       help='Path to AVIRIS *_corr_*_img file')
    parser.add_argument('--patches-file', type=str, required=True,
                       help='Path to extracted patches .h5 file')
    parser.add_argument('--output', type=str,
                       default='outputs/flight_visualization.png',
                       help='Output visualization path')
    parser.add_argument('--downsample', type=int, default=2,
                       help='Downsampling factor for scene display (default: 2)')

    args = parser.parse_args()

    print("="*70)
    print("AVIRIS Flight Visualization with Patch Diversity Analysis")
    print("="*70)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    stats = visualize_flight_with_patches(
        args.aviris_file,
        args.patches_file,
        str(output_path),
        downsample=args.downsample
    )

    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)

    # Provide recommendations based on diversity
    spectral_diversity = stats['std_reflectance'] / (stats['mean_reflectance'] + 1e-8)
    spatial_diversity = stats['std_spatial_energy'] / (stats['mean_spatial_energy'] + 1e-8)

    print(f"  Spectral diversity: {spectral_diversity:.4f}")
    print(f"  Spatial diversity: {spatial_diversity:.4f}")

    if spectral_diversity < 0.1 and spatial_diversity < 0.3:
        print("\n  ⚠ LOW DIVERSITY: Patches are very similar.")
        print("  Consider increasing stride or filtering for more diverse patches.")
        suggested_keep = max(int(stats['n_patches'] * 0.3), 100)
        print(f"  Suggested: Keep ~{suggested_keep} most diverse patches")
    elif spectral_diversity < 0.2 and spatial_diversity < 0.5:
        print("\n  ⚡ MODERATE DIVERSITY: Some redundancy present.")
        suggested_keep = max(int(stats['n_patches'] * 0.5), 200)
        print(f"  Suggested: Keep ~{suggested_keep} patches or use current set")
    else:
        print("\n  ✓ GOOD DIVERSITY: Patches cover diverse spectral and spatial content.")
        print(f"  Recommended: Keep all {stats['n_patches']} patches")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
