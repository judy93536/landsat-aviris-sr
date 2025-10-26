"""
Evaluate 3D U-Net Performance

Generates visualizations comparing 3D U-Net predictions with ground truth.
"""

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.unet3d import UNet3D, LightweightUNet3D


def load_model(checkpoint, device='cuda', lightweight=False, base_features=16):
    """Load 3D U-Net model."""

    if lightweight:
        model = LightweightUNet3D(
            in_channels=7,
            out_channels=198,
            base_features=base_features
        ).to(device)
    else:
        model = UNet3D(
            in_channels=7,
            out_channels=198,
            base_features=base_features
        ).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()

    print(f"✓ Loaded 3D U-Net from {checkpoint}")
    return model


def predict(landsat, model, device='cuda'):
    """Run 3D U-Net prediction."""
    landsat_tensor = torch.from_numpy(landsat.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(landsat_tensor)

    pred = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return pred


def compute_metrics(pred, gt):
    """Compute RMSE and MAE."""
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))
    return rmse, mae


def visualize_results(landsat, pred, aviris_gt, output_path, wavelengths):
    """Generate comprehensive visualization."""

    fig = plt.figure(figsize=(18, 10))

    # RGB indices
    rgb_indices = [50, 30, 20]

    def to_rgb(cube, indices):
        rgb = cube[:, :, indices]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)

    landsat_rgb = landsat[:, :, [3, 2, 1]]
    landsat_rgb = (landsat_rgb - landsat_rgb.min()) / (landsat_rgb.max() - landsat_rgb.min() + 1e-8)
    landsat_rgb = np.clip(landsat_rgb, 0, 1)

    pred_rgb = to_rgb(pred, rgb_indices)
    aviris_rgb = to_rgb(aviris_gt, rgb_indices)

    # Row 1: RGB Comparisons
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(landsat_rgb)
    ax1.set_title('Input: Landsat\n(7 bands)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(pred_rgb)
    rmse, mae = compute_metrics(pred, aviris_gt)
    ax2.set_title(f'3D U-Net Prediction\n(198 bands)\nRMSE: {rmse:.4f}',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(aviris_rgb)
    ax3.set_title('Ground Truth\nAVIRIS Classic', fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    error = np.abs(pred - aviris_gt).mean(axis=2)
    im = ax4.imshow(error, cmap='hot', vmin=0, vmax=0.15)
    ax4.set_title(f'Error Map\nMAE: {mae:.4f}', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)

    # Row 2: Spectral curves at 3 locations
    h, w = aviris_gt.shape[:2]
    locations = [
        (h//4, w//4, 'Top-left'),
        (h//2, w//2, 'Center'),
        (3*h//4, 3*w//4, 'Bottom-right')
    ]

    for idx, (y, x, label) in enumerate(locations):
        ax = plt.subplot(2, 4, 5 + idx)

        ax.plot(wavelengths, aviris_gt[y, x, :], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.plot(wavelengths, pred[y, x, :], 'r-', linewidth=1.5, label='3D U-Net', alpha=0.8)

        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Reflectance', fontsize=10)
        ax.set_title(f'Spectral Response: {label}\nPixel ({y}, {x})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Last panel: Per-band RMSE
    ax8 = plt.subplot(2, 4, 8)
    band_rmse = np.sqrt(np.mean((pred - aviris_gt) ** 2, axis=(0, 1)))
    ax8.plot(wavelengths, band_rmse, 'r-', linewidth=1.5, alpha=0.8)
    ax8.axhline(y=rmse, color='r', linestyle='--', alpha=0.5, label=f'Overall: {rmse:.4f}')
    ax8.set_xlabel('Wavelength (nm)', fontsize=10)
    ax8.set_ylabel('RMSE', fontsize=10)
    ax8.set_title('Per-Band RMSE', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    plt.suptitle('3D U-Net Performance: End-to-End Joint Spatial-Spectral Learning',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print(f"3D U-Net Performance:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D U-Net')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/3d_unet_evaluation',
                       help='Output directory')
    parser.add_argument('--sample-idx', type=int, default=0, help='Which sample to visualize')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model')
    parser.add_argument('--base-features', type=int, default=16, help='Base feature channels')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("="*70)
    print("3D U-Net Evaluation")
    print("="*70)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    print(f"\nLoading 3D U-Net...")
    model = load_model(args.checkpoint, device=device, lightweight=args.lightweight,
                      base_features=args.base_features)

    # Load data
    print(f"\nLoading dataset from {args.data}...")
    with h5py.File(args.data, 'r') as f:
        landsat = f['landsat'][args.sample_idx]
        aviris_gt = f['aviris'][args.sample_idx]
        wavelengths = f.attrs['wavelengths']

        # Normalize
        aviris_p2 = np.percentile(f['aviris'][:], 2)
        aviris_p98 = np.percentile(f['aviris'][:], 98)
        landsat_p2 = np.percentile(f['landsat'][:], 2)
        landsat_p98 = np.percentile(f['landsat'][:], 98)

        landsat = (landsat - landsat_p2) / (landsat_p98 - landsat_p2 + 1e-8)
        aviris_gt = (aviris_gt - aviris_p2) / (aviris_p98 - aviris_p2 + 1e-8)
        landsat = np.clip(landsat, 0, 1)
        aviris_gt = np.clip(aviris_gt, 0, 1)

    print(f"  Sample {args.sample_idx}")
    print(f"  Landsat shape: {landsat.shape}")
    print(f"  AVIRIS shape: {aviris_gt.shape}")

    # Run prediction
    print(f"\nRunning 3D U-Net prediction...")
    pred = predict(landsat, model, device)

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    output_path = output_dir / f'3d_unet_sample_{args.sample_idx}.png'
    visualize_results(landsat, pred, aviris_gt, output_path, wavelengths)

    print(f"✓ Saved visualization to {output_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
