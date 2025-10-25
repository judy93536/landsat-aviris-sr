#!/usr/bin/env python3
"""
Stage 1 Inference: Generate Enhanced HSI from Landsat

This script loads a trained Stage 1 model and performs spectral super-resolution
on test patches, comparing predicted 340-band AVIRIS with ground truth.
"""

import argparse
import os
import sys
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stage1_spectral.spectral_sr_net import SpectralSRNet


def load_model(checkpoint_path, device):
    """Load trained Stage 1 model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    model = SpectralSRNet(
        in_bands=7,
        out_bands=340,
        hidden_dim=128,
        num_res_blocks=8,
        use_attention=True
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")

    return model


def load_test_patch(h5_file, patch_idx=0):
    """Load a single patch from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        aviris = f['aviris'][patch_idx]  # (256, 256, 340) -> need to transpose
        landsat = f['landsat'][patch_idx]  # (256, 256, 7) -> need to transpose

        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        aviris = aviris.transpose(2, 0, 1)  # (340, 256, 256)
        landsat = landsat.transpose(2, 0, 1)  # (7, 256, 256)

        # Get normalization stats (if available, otherwise compute from all patches)
        if 'normalization_min' in f['aviris'].attrs:
            aviris_min = f['aviris'].attrs['normalization_min']
            aviris_max = f['aviris'].attrs['normalization_max']
            landsat_min = f['landsat'].attrs['normalization_min']
            landsat_max = f['landsat'].attrs['normalization_max']
        else:
            # Compute from all patches using percentiles (same as dataloader)
            print("  Computing normalization statistics from all patches...")
            aviris_all = f['aviris'][:]
            landsat_all = f['landsat'][:]

            aviris_min = float(np.percentile(aviris_all, 1))
            aviris_max = float(np.percentile(aviris_all, 99))
            landsat_min = float(np.percentile(landsat_all, 1))
            landsat_max = float(np.percentile(landsat_all, 99))

    return {
        'aviris': aviris,
        'landsat': landsat,
        'aviris_min': aviris_min,
        'aviris_max': aviris_max,
        'landsat_min': landsat_min,
        'landsat_max': landsat_max
    }


def normalize(data, data_min, data_max):
    """Normalize to [0, 1] using stored min/max."""
    return (data - data_min) / (data_max - data_min + 1e-8)


def denormalize(data, data_min, data_max):
    """Denormalize back to original range."""
    return data * (data_max - data_min) + data_min


def compute_metrics(pred, target):
    """Compute reconstruction metrics."""
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    # Ensure (bands, height, width) format
    if pred.shape[0] != 340:
        pred = pred.transpose(1, 2, 0).transpose(2, 0, 1)
    if target.shape[0] != 340:
        target = target.transpose(1, 2, 0).transpose(2, 0, 1)

    # RMSE per band
    rmse_per_band = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))

    # Overall RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # SAM per pixel (convert to (H, W, C))
    pred_hwc = pred.transpose(1, 2, 0)  # (256, 256, 340)
    target_hwc = target.transpose(1, 2, 0)

    # Compute SAM
    pred_norm = np.linalg.norm(pred_hwc, axis=2, keepdims=True) + 1e-8
    target_norm = np.linalg.norm(target_hwc, axis=2, keepdims=True) + 1e-8
    dot_product = np.sum(pred_hwc * target_hwc, axis=2)
    sam_rad = np.arccos(np.clip(dot_product / (pred_norm[:, :, 0] * target_norm[:, :, 0]), -1, 1))
    sam_mean = np.mean(sam_rad)
    sam_deg = np.degrees(sam_mean)

    return {
        'rmse': rmse,
        'rmse_per_band': rmse_per_band,
        'sam_rad': sam_mean,
        'sam_deg': sam_deg
    }


def visualize_results(landsat, aviris_true, aviris_pred, metrics, output_path):
    """Create comprehensive visualization of results."""
    # Denormalize for visualization (if needed)

    # Get wavelengths (AVIRIS-NG range)
    wavelengths = np.linspace(380, 2500, 340)  # Approximate wavelengths in nm

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Landsat RGB composite (bands 4,3,2 = R,G,B)
    ax1 = plt.subplot(3, 4, 1)
    landsat_rgb = np.stack([
        landsat[3],  # Red
        landsat[2],  # Green
        landsat[1]   # Blue
    ], axis=-1)
    # Normalize for display
    landsat_rgb = (landsat_rgb - landsat_rgb.min()) / (landsat_rgb.max() - landsat_rgb.min() + 1e-8)
    ax1.imshow(landsat_rgb)
    ax1.set_title('Input: Landsat RGB\n(7 bands @ 30m)', fontsize=10)
    ax1.axis('off')

    # 2. True AVIRIS RGB composite (bands ~R,G,B)
    ax2 = plt.subplot(3, 4, 2)
    # AVIRIS bands: ~650nm(R), ~550nm(G), ~470nm(B)
    r_idx = np.argmin(np.abs(wavelengths - 650))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 470))

    aviris_true_rgb = np.stack([
        aviris_true[r_idx],
        aviris_true[g_idx],
        aviris_true[b_idx]
    ], axis=-1)
    aviris_true_rgb = (aviris_true_rgb - aviris_true_rgb.min()) / (aviris_true_rgb.max() - aviris_true_rgb.min() + 1e-8)
    ax2.imshow(aviris_true_rgb)
    ax2.set_title('Ground Truth: AVIRIS RGB\n(340 bands @ 4m)', fontsize=10)
    ax2.axis('off')

    # 3. Predicted AVIRIS RGB composite
    ax3 = plt.subplot(3, 4, 3)
    aviris_pred_rgb = np.stack([
        aviris_pred[r_idx],
        aviris_pred[g_idx],
        aviris_pred[b_idx]
    ], axis=-1)
    aviris_pred_rgb = (aviris_pred_rgb - aviris_pred_rgb.min()) / (aviris_pred_rgb.max() - aviris_pred_rgb.min() + 1e-8)
    ax3.imshow(aviris_pred_rgb)
    ax3.set_title('Predicted: AVIRIS RGB\n(340 bands @ 4m)', fontsize=10)
    ax3.axis('off')

    # 4. Difference map (RGB)
    ax4 = plt.subplot(3, 4, 4)
    diff_rgb = np.abs(aviris_true_rgb - aviris_pred_rgb)
    ax4.imshow(diff_rgb)
    ax4.set_title(f'Absolute Difference\nRMSE: {metrics["rmse"]:.4f}', fontsize=10)
    ax4.axis('off')

    # 5-7. Spectral curves at different pixels
    pixel_locations = [
        (128, 128, 'Center'),
        (64, 64, 'Top-Left'),
        (192, 192, 'Bottom-Right')
    ]

    for idx, (row, col, label) in enumerate(pixel_locations):
        ax = plt.subplot(3, 4, 5 + idx)
        ax.plot(wavelengths, aviris_true[:, row, col], 'b-', label='True', linewidth=2)
        ax.plot(wavelengths, aviris_pred[:, row, col], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.set_title(f'Spectral Signature ({label})\nPixel ({row}, {col})', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 8. RMSE per band
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(wavelengths, metrics['rmse_per_band'], 'k-', linewidth=1.5)
    ax8.set_xlabel('Wavelength (nm)', fontsize=9)
    ax8.set_ylabel('RMSE', fontsize=9)
    ax8.set_title('RMSE per Spectral Band', fontsize=10)
    ax8.grid(True, alpha=0.3)

    # 9. True AVIRIS band 100
    ax9 = plt.subplot(3, 4, 9)
    im9 = ax9.imshow(aviris_true[100], cmap='viridis')
    ax9.set_title(f'True AVIRIS\nBand 100 ({wavelengths[100]:.0f}nm)', fontsize=10)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    # 10. Predicted AVIRIS band 100
    ax10 = plt.subplot(3, 4, 10)
    im10 = ax10.imshow(aviris_pred[100], cmap='viridis')
    ax10.set_title(f'Predicted AVIRIS\nBand 100 ({wavelengths[100]:.0f}nm)', fontsize=10)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    # 11. Difference for band 100
    ax11 = plt.subplot(3, 4, 11)
    diff = np.abs(aviris_true[100] - aviris_pred[100])
    im11 = ax11.imshow(diff, cmap='hot')
    ax11.set_title(f'Abs Difference\nBand 100', fontsize=10)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)

    # 12. Metrics summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    metrics_text = f"""
    Performance Metrics:

    RMSE: {metrics['rmse']:.4f}
    SAM: {metrics['sam_rad']:.4f} rad
         {metrics['sam_deg']:.2f}°

    RMSE per band:
      Min: {metrics['rmse_per_band'].min():.4f}
      Max: {metrics['rmse_per_band'].max():.4f}
      Mean: {metrics['rmse_per_band'].mean():.4f}

    Model: Stage 1 Spectral SR
    Input: 7 Landsat bands
    Output: 340 AVIRIS bands
    """
    ax12.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Stage 1: Spectral Super-Resolution Results (7 → 340 bands)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Inference')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/stage1_training/checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-file', type=str,
                        default='outputs/dataset_small/ang20190624t214359_rdn_v2u1/test_patches.h5',
                        help='Path to HDF5 test data')
    parser.add_argument('--patch-idx', type=int, default=0,
                        help='Index of patch to test (default: 0)')
    parser.add_argument('--output-dir', type=str, default='outputs/inference_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load test patch
    print(f"\nLoading test patch {args.patch_idx} from: {args.data_file}")
    patch_data = load_test_patch(args.data_file, args.patch_idx)

    # Normalize
    landsat_norm = normalize(patch_data['landsat'],
                             patch_data['landsat_min'],
                             patch_data['landsat_max'])
    aviris_norm = normalize(patch_data['aviris'],
                           patch_data['aviris_min'],
                           patch_data['aviris_max'])

    # Prepare input
    landsat_tensor = torch.from_numpy(landsat_norm).float().unsqueeze(0).to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        aviris_pred_norm = model(landsat_tensor)

    # Denormalize
    aviris_pred_norm = aviris_pred_norm.squeeze(0).cpu().numpy()
    aviris_pred = denormalize(aviris_pred_norm,
                             patch_data['aviris_min'],
                             patch_data['aviris_max'])
    aviris_true = patch_data['aviris']

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(aviris_pred, aviris_true)

    print(f"\nResults:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  SAM:  {metrics['sam_rad']:.4f} rad ({metrics['sam_deg']:.2f}°)")
    print(f"  RMSE per band - Min: {metrics['rmse_per_band'].min():.4f}, "
          f"Max: {metrics['rmse_per_band'].max():.4f}, "
          f"Mean: {metrics['rmse_per_band'].mean():.4f}")

    # Visualize
    print("\nGenerating visualization...")
    output_path = os.path.join(args.output_dir, f'inference_patch_{args.patch_idx}.png')
    visualize_results(patch_data['landsat'], aviris_true, aviris_pred, metrics, output_path)

    # Save predicted AVIRIS as HDF5
    output_h5 = os.path.join(args.output_dir, f'predicted_aviris_patch_{args.patch_idx}.h5')
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('landsat_input', data=patch_data['landsat'])
        f.create_dataset('aviris_true', data=aviris_true)
        f.create_dataset('aviris_predicted', data=aviris_pred)
        f.attrs['rmse'] = metrics['rmse']
        f.attrs['sam_rad'] = metrics['sam_rad']
        f.attrs['sam_deg'] = metrics['sam_deg']
    print(f"Predicted data saved to: {output_h5}")

    print("\n" + "="*70)
    print("Inference complete!")
    print(f"Output directory: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
