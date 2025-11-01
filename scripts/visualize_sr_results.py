#!/usr/bin/env python3
"""
Visualize super-resolution results: compare Landsat input, predicted AVIRIS, and true AVIRIS.

Creates:
- RGB composite comparisons
- Spectral signature plots
- Error maps (SAM, RMSE)
- Band-by-band analysis
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.unet3d import LightweightUNet3D


def load_model(checkpoint_path, in_channels, out_channels, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")

    # Create model with correct dimensions
    model = LightweightUNet3D(in_channels=in_channels, out_channels=out_channels, base_features=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded (in={in_channels}, out={out_channels})")
    return model


def compute_sam(pred, target):
    """Compute Spectral Angle Mapper in degrees."""
    # Normalize vectors
    pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    cos_sim = np.sum(pred_norm * target_norm, axis=-1)
    cos_sim = np.clip(cos_sim, -1, 1)

    # Convert to degrees
    sam_rad = np.arccos(cos_sim)
    sam_deg = np.degrees(sam_rad)

    return sam_deg


def make_rgb(cube, wavelengths, rgb_bands=[640, 550, 460]):
    """Create RGB composite from hyperspectral cube."""
    # Find closest bands to RGB wavelengths
    r_idx = np.argmin(np.abs(wavelengths - rgb_bands[0]))
    g_idx = np.argmin(np.abs(wavelengths - rgb_bands[1]))
    b_idx = np.argmin(np.abs(wavelengths - rgb_bands[2]))

    rgb = np.stack([cube[:, :, r_idx], cube[:, :, g_idx], cube[:, :, b_idx]], axis=-1)

    # Normalize to [0, 1]
    rgb = np.clip(rgb, 0, 1)

    # Enhance contrast (2nd-98th percentile)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def visualize_patch(landsat, pred_aviris, true_aviris, wavelengths, output_path, patch_idx=0):
    """Create comprehensive visualization for one patch."""

    fig = plt.figure(figsize=(20, 12))

    # ========== Row 1: RGB Composites ==========

    # Landsat RGB (upsampled for visualization)
    ax1 = plt.subplot(3, 4, 1)
    landsat_rgb_bands = [4, 3, 2]  # NIR, Red, Green for false color
    landsat_rgb = np.stack([landsat[:, :, b] for b in landsat_rgb_bands], axis=-1)
    landsat_rgb = (landsat_rgb - landsat_rgb.min()) / (landsat_rgb.max() - landsat_rgb.min() + 1e-8)
    ax1.imshow(landsat_rgb)
    ax1.set_title('Input: Landsat (7 bands)\nFalse Color (NIR-R-G)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Predicted AVIRIS RGB
    ax2 = plt.subplot(3, 4, 2)
    pred_rgb = make_rgb(pred_aviris, wavelengths)
    ax2.imshow(pred_rgb)
    ax2.set_title('Predicted: AVIRIS-NG (359 bands)\nTrue Color RGB', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # True AVIRIS RGB
    ax3 = plt.subplot(3, 4, 3)
    true_rgb = make_rgb(true_aviris, wavelengths)
    ax3.imshow(true_rgb)
    ax3.set_title('Ground Truth: AVIRIS-NG\nTrue Color RGB', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Error RGB
    ax4 = plt.subplot(3, 4, 4)
    error_rgb = np.abs(pred_rgb - true_rgb)
    im4 = ax4.imshow(error_rgb, cmap='hot')
    ax4.set_title('Absolute Error (RGB)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # ========== Row 2: Error Maps ==========

    # RMSE map
    ax5 = plt.subplot(3, 4, 5)
    rmse_map = np.sqrt(np.mean((pred_aviris - true_aviris) ** 2, axis=-1))
    im5 = ax5.imshow(rmse_map, cmap='viridis')
    ax5.set_title(f'RMSE Map\nMean: {rmse_map.mean():.4f}', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # SAM map
    ax6 = plt.subplot(3, 4, 6)
    sam_map = compute_sam(pred_aviris, true_aviris)
    im6 = ax6.imshow(sam_map, cmap='plasma', vmin=0, vmax=20)
    ax6.set_title(f'SAM Map (degrees)\nMean: {sam_map.mean():.2f}°', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # Mean Absolute Error map
    ax7 = plt.subplot(3, 4, 7)
    mae_map = np.mean(np.abs(pred_aviris - true_aviris), axis=-1)
    im7 = ax7.imshow(mae_map, cmap='magma')
    ax7.set_title(f'MAE Map\nMean: {mae_map.mean():.4f}', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    # Histogram of SAM errors
    ax8 = plt.subplot(3, 4, 8)
    ax8.hist(sam_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax8.axvline(sam_map.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sam_map.mean():.2f}°')
    ax8.axvline(np.median(sam_map), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(sam_map):.2f}°')
    ax8.set_xlabel('SAM (degrees)', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('SAM Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3)

    # ========== Row 3: Spectral Signatures ==========

    # Select sample pixels (center, bright, dark, edge)
    h, w = true_aviris.shape[:2]
    sample_pixels = {
        'Center': (h//2, w//2),
        'Bright': np.unravel_index(np.argmax(true_aviris.mean(axis=-1)), (h, w)),
        'Dark': np.unravel_index(np.argmin(true_aviris.mean(axis=-1)), (h, w)),
        'Edge': (h//4, w//4)
    }

    colors = {'Center': 'blue', 'Bright': 'orange', 'Dark': 'green', 'Edge': 'red'}

    # Spectral signature plot
    ax9 = plt.subplot(3, 4, 9)
    for name, (y, x) in sample_pixels.items():
        ax9.plot(wavelengths, true_aviris[y, x], '-', color=colors[name], alpha=0.7, linewidth=2, label=f'{name} (True)')
        ax9.plot(wavelengths, pred_aviris[y, x], '--', color=colors[name], alpha=0.7, linewidth=1.5, label=f'{name} (Pred)')
    ax9.set_xlabel('Wavelength (nm)', fontsize=10)
    ax9.set_ylabel('Reflectance', fontsize=10)
    ax9.set_title('Spectral Signatures (Sample Pixels)', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8, ncol=2)
    ax9.grid(alpha=0.3)

    # Spectral error plot
    ax10 = plt.subplot(3, 4, 10)
    for name, (y, x) in sample_pixels.items():
        error = pred_aviris[y, x] - true_aviris[y, x]
        ax10.plot(wavelengths, error, color=colors[name], alpha=0.7, linewidth=1.5, label=name)
    ax10.axhline(0, color='black', linestyle='-', linewidth=1)
    ax10.set_xlabel('Wavelength (nm)', fontsize=10)
    ax10.set_ylabel('Error (Pred - True)', fontsize=10)
    ax10.set_title('Spectral Error by Wavelength', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=8)
    ax10.grid(alpha=0.3)

    # Band-wise RMSE
    ax11 = plt.subplot(3, 4, 11)
    band_rmse = np.sqrt(np.mean((pred_aviris - true_aviris) ** 2, axis=(0, 1)))
    ax11.plot(wavelengths, band_rmse, linewidth=2, color='darkblue')
    ax11.fill_between(wavelengths, 0, band_rmse, alpha=0.3)
    ax11.set_xlabel('Wavelength (nm)', fontsize=10)
    ax11.set_ylabel('RMSE', fontsize=10)
    ax11.set_title(f'Band-wise RMSE\nMean: {band_rmse.mean():.4f}', fontsize=12, fontweight='bold')
    ax11.grid(alpha=0.3)

    # Statistics table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    stats = [
        ['Metric', 'Value'],
        ['Overall RMSE', f'{rmse_map.mean():.4f}'],
        ['Overall MAE', f'{mae_map.mean():.4f}'],
        ['Mean SAM', f'{sam_map.mean():.2f}°'],
        ['Median SAM', f'{np.median(sam_map):.2f}°'],
        ['SAM < 5°', f'{(sam_map < 5).mean()*100:.1f}%'],
        ['SAM < 10°', f'{(sam_map < 10).mean()*100:.1f}%'],
        ['Max SAM', f'{sam_map.max():.2f}°'],
    ]

    table = ax12.table(cellText=stats, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax12.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle(f'AVIRIS-NG Super-Resolution Results - Patch {patch_idx}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize super-resolution results')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to test data HDF5 file')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sr_visualization'),
                       help='Output directory for visualizations')
    parser.add_argument('--n-patches', type=int, default=5,
                       help='Number of patches to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("AVIRIS-NG Super-Resolution Visualization")
    print("="*70)

    # Load data first to determine dimensions
    print(f"\nLoading test data from {args.data}...")
    with h5py.File(args.data, 'r') as f:
        # Detect dimensions from data
        sample_landsat = f['landsat'][0]
        sample_aviris = f['aviris'][0]
        in_channels = sample_landsat.shape[2]  # (H, W, C)
        out_channels = sample_aviris.shape[2]

    print(f"  Detected dimensions: {in_channels} input → {out_channels} output bands")

    # Load model with correct dimensions
    device = torch.device(args.device)
    model = load_model(args.checkpoint, in_channels, out_channels, device)

    # Load data for processing
    print(f"\nProcessing patches from {args.data}...")
    with h5py.File(args.data, 'r') as f:
        # Get total patches
        n_total = len(f['aviris'])
        print(f"  Total patches: {n_total}")

        # Load wavelengths
        wavelengths = f['wavelengths'][:]
        print(f"  Wavelengths: {len(wavelengths)}")

        # Select random patches
        n_viz = min(args.n_patches, n_total)
        patch_indices = np.random.choice(n_total, n_viz, replace=False)

        print(f"\nProcessing {n_viz} patches...")

        for i, patch_idx in enumerate(patch_indices):
            print(f"\n  Patch {i+1}/{n_viz} (index {patch_idx})...")

            # Load patch
            landsat = f['landsat'][patch_idx]  # (256, 256, 7)
            true_aviris = f['aviris'][patch_idx]  # (256, 256, 359)

            # Run inference
            landsat_tensor = torch.from_numpy(landsat).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                pred_tensor = model(landsat_tensor)

            pred_aviris = pred_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

            # Visualize
            output_path = args.output_dir / f'sr_result_patch_{patch_idx:04d}.png'
            visualize_patch(landsat, pred_aviris, true_aviris, wavelengths, output_path, patch_idx)

    print("\n" + "="*70)
    print("Visualization complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
