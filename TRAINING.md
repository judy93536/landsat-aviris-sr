# Training Infrastructure

This document describes the training infrastructure for the two-stage hyperspectral super-resolution model.

## Overview

The model consists of two stages:

1. **Stage 1 - Spectral Super-Resolution**: Expands 7 Landsat bands to 340 AVIRIS-like hyperspectral bands
2. **Stage 2 - Spatial Super-Resolution**: Upsamples 340 bands from 30m to 4m resolution (7.5× upsampling)

## Components

### Models
- `models/stage1_spectral/spectral_sr_net.py`: Spectral SR networks
  - `SpectralSRNet`: Main model with spectral attention (4M params)
  - `SpectralUnmixingNet`: Alternative physics-based model

- `models/stage2_spatial/spatial_sr_net.py`: Spatial SR networks
  - `SpatialSRNet`: RCAN-based model for spatial upsampling
  - `LightweightSpatialSRNet`: Faster version with fewer parameters (130K params)

### Loss Functions
- `utils/losses.py`: Hyperspectral-specific loss functions
  - `SpectralAngleMapper`: Measures spectral similarity
  - `SpectralGradientLoss`: Encourages smooth spectral curves
  - `CombinedLoss`: Weighted combination of L1, SAM, and spectral gradient
  - `SpatialLoss`: For spatial SR training

### Data Loading
- `utils/dataloader.py`: HDF5 patch loading and normalization
  - `HyperspectralPairDataset`: Loads paired Landsat-AVIRIS patches
  - `create_dataloaders`: Creates train/val data loaders
  - Automatic train/val splitting
  - Robust percentile-based normalization

### Training Scripts
- `scripts/train_stage1.py`: Training script for Stage 1
  - Configurable model architecture
  - Learning rate scheduling
  - Checkpointing and early stopping
  - TensorBoard logging

## Dataset

The training data consists of 36 paired patches from 3 AVIRIS-NG scenes:
- Dataset 1: 16 patches (ang20190624t214359_rdn_v2u1)
- Dataset 2: 10 patches (ang20190624t230039_rdn_v2u1)
- Dataset 3: 10 patches (ang20190624t230448_rdn_v2u1)

Each patch:
- AVIRIS: 340 bands, 256×256 pixels @ 4m resolution
- Landsat: 7 bands, 256×256 pixels @ 30m resolution (upsampled from 34×34)

Location: `outputs/dataset_small/`

## Quick Start

### 1. Test the Training Pipeline

Verify all components work together:

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate
python scripts/test_training_pipeline.py
```

Expected output:
```
✓ Stage 1 pipeline test passed!
✓ Stage 2 pipeline test passed!
✓ All tests passed! Ready to start training.
```

### 2. Train Stage 1 (Spectral SR)

Train the spectral super-resolution model:

```bash
python scripts/train_stage1.py \
    --data-dir outputs/dataset_small \
    --output-dir outputs/stage1_training \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4
```

Options:
- `--model`: Choose between `spectral_sr` (default) or `spectral_unmixing`
- `--checkpoint`: Resume from checkpoint
- `--val-fraction`: Fraction of data for validation (default: 0.2)

### 3. Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs/stage1_training/tensorboard
```

Metrics tracked:
- Training/validation loss
- L1 reconstruction loss
- Spectral Angle Mapper (SAM)
- Spectral gradient loss
- Learning rate

### 4. Train Stage 2 (Spatial SR)

After Stage 1 converges, train the spatial super-resolution model:

```bash
# TODO: Create train_stage2.py script
```

## Model Architecture Details

### Stage 1: Spectral SR

**SpectralSRNet**:
- Input: 7 Landsat bands (256×256)
- Output: 340 AVIRIS bands (256×256)
- Architecture:
  1. Initial spectral expansion (7→340) with 2D convolutions
  2. Spectral residual blocks (1D convs along spectral dimension)
  3. Spectral attention mechanism
  4. Skip connections for stability

**Loss Function**:
- L1 loss (weight: 1.0)
- Spectral Angle Mapper (weight: 0.1)
- Spectral gradient loss (weight: 0.1)

### Stage 2: Spatial SR

**SpatialSRNet** / **LightweightSpatialSRNet**:
- Input: 340 bands at low resolution (34×34 @ 30m)
- Output: 340 bands at high resolution (255×255 @ 4m)
- Architecture:
  1. Shallow feature extraction
  2. Residual channel attention blocks (RCAB)
  3. Pixel shuffle upsampling (3 stages of 2×)
  4. Residual learning with bicubic baseline

**Loss Function**:
- L1 loss (weight: 1.0)
- Spectral Angle Mapper (weight: 0.05)

## Expected Training Time

**Stage 1** (36 patches, 100 epochs):
- CPU: ~6-8 hours
- Tesla V100 GPU: ~30-45 minutes

**Stage 2** (36 patches, 100 epochs):
- CPU: ~8-10 hours
- Tesla V100 GPU: ~45-60 minutes

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 2 or 1)
- Use `LightweightSpatialSRNet` instead of full model

### Poor Convergence
- Check TensorBoard for training curves
- Reduce learning rate (try 1e-5)
- Increase spectral loss weights

### Dimension Mismatches
- Ensure patch sizes match expected dimensions
- Stage 1: 7 bands → 340 bands (same spatial size)
- Stage 2: 34×34 → 255×255 (7.5× upsampling)

## Next Steps

1. Train Stage 1 on the 36-patch dataset
2. Evaluate spectral fidelity (SAM, RMSE by band)
3. Generate predictions for validation set
4. Create visual comparisons (RGB, spectral curves)
5. Train Stage 2 using Stage 1 outputs
6. Evaluate end-to-end performance
7. Consider processing larger datasets for more training data

## Files Created

```
models/
├── stage1_spectral/
│   └── spectral_sr_net.py          # Spectral SR models
└── stage2_spatial/
    └── spatial_sr_net.py           # Spatial SR models

utils/
├── losses.py                        # Hyperspectral loss functions
└── dataloader.py                    # HDF5 data loading

scripts/
├── train_stage1.py                  # Stage 1 training script
└── test_training_pipeline.py       # Pipeline sanity check
```
