# Post-Reboot Continuation Guide

**Date Created**: October 24, 2025
**System**: ubuntu-z8 with NVIDIA TITAN V
**Project**: Landsat-to-AVIRIS Hyperspectral Super-Resolution

---

## Current Status Before Reboot

### ✅ Completed Work

1. **Project Setup**
   - Git repository: https://github.com/judy93536/landsat-aviris-sr.git
   - Virtual environment created at `/raid/MSI_MSI_AIML/landsat-aviris-sr/venv`
   - All dependencies installed (PyTorch, h5py, spectral, matplotlib, tensorboard, etc.)
   - Project structure fully established

2. **Data Pipeline**
   - AVIRIS-NG ENVI loader implemented (`data/envi_loader.py`)
   - Landsat-8 spectral response functions (`data/landsat_srf.py`)
   - Synthetic data generator (`data/synthetic.py`)
     - Spectral integration: 340 AVIRIS bands → 7 Landsat bands
     - Spatial degradation: 4m → 30m resolution
     - Bicubic upsampling for pixel correspondence
     - Invalid pixel handling (-9999 ignore values)
     - Contamination detection
     - Thumbnail generation for quality control

3. **Training Datasets Generated**
   - **Small datasets** (COMPLETED): 36 patches from 3 scenes
     - `ang20190624t214359_rdn_v2u1`: 16 patches (1.2 GB)
     - `ang20190624t230039_rdn_v2u1`: 10 patches (759 MB)
     - `ang20190624t230448_rdn_v2u1`: 10 patches (761 MB)
     - Location: `/raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/dataset_small/`

   - **Large datasets** (IN PROGRESS): ~4-6 hours processing time
     - `ang20190623t194727_rdn_v2u1`: 749×17255 pixels
     - `ang20190715t172845_rdn_v2v2`: 657×16476 pixels
     - Expected: 200+ additional patches
     - Location: `/raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/dataset_large/`
     - Log file: `outputs/large_dataset_generation.log`

4. **Model Architectures Implemented**
   - **Stage 1 - Spectral SR** (`models/stage1_spectral/spectral_sr_net.py`)
     - `SpectralSRNet`: 7→340 bands using spectral attention (7M params)
     - `SpectralUnmixingNet`: Physics-based alternative with endmembers

   - **Stage 2 - Spatial SR** (`models/stage2_spatial/spatial_sr_net.py`)
     - `SpatialSRNet`: RCAN-based 7.5× upsampling
     - `LightweightSpatialSRNet`: Faster version (130K params)

5. **Loss Functions** (`utils/losses.py`)
   - Spectral Angle Mapper (SAM)
   - Spectral Gradient Loss
   - Combined Loss (L1 + SAM + Spectral Gradient)
   - Spatial Loss

6. **Data Loaders** (`utils/dataloader.py`)
   - HDF5 patch loading
   - Automatic train/val splitting
   - Robust percentile-based normalization
   - Works with all generated datasets

7. **Training Infrastructure**
   - Training script: `scripts/train_stage1.py`
   - TensorBoard integration
   - Checkpointing and early stopping
   - Learning rate scheduling
   - **Fixed**: Removed verbose parameter bug

8. **Testing & Validation**
   - Pipeline test script: `scripts/test_data_pipeline.py`
   - Training pipeline test: `scripts/test_training_pipeline.py`
   - ✅ All tests passed before reboot

9. **Technical Report** (`report/`)
   - SPIE-formatted LaTeX manuscript
   - 24 verified references in BibTeX format
   - Complete methodology documented
   - Placeholders for results

10. **Documentation**
    - `TRAINING.md`: Complete training guide
    - `report/README.md`: How to build LaTeX document
    - All code well-commented

---

## Issue That Required Reboot

**NVIDIA Driver Version Mismatch**
- Loaded kernel driver: 535.247.01
- nvidia-smi library: 535.274
- Result: CUDA not available to PyTorch
- Solution: Reboot to load matching driver version

---

## After Reboot: Step-by-Step Instructions

### 1. Verify GPU is Working

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# - Driver Version: 535.274 (or similar)
# - GPU: TITAN V
# - No errors
```

### 2. Verify CUDA is Available to PyTorch

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Expected output:
# CUDA available: True
# GPU: TITAN V
```

### 3. Check Large Dataset Generation Status

```bash
# Check if process is still running
ps aux | grep test_data_pipeline.py | grep -v grep

# Check progress
tail -50 /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/large_dataset_generation.log

# If completed, verify output
ls -lh /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/dataset_large/
```

**Expected Results:**
- If completed: 2 directories with `.h5` files and thumbnails
- Each dataset: ~50-100+ patches
- Total size: Several GB

### 4. Start TensorBoard (for monitoring training)

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate

# Start in background
nohup tensorboard --logdir outputs/stage1_training/tensorboard --host 0.0.0.0 --port 6006 > /dev/null 2>&1 &

# Access at: http://localhost:6006 or http://ubuntu-z8:6006
```

### 5. Start Training Stage 1

#### Option A: Train on Small Dataset (36 patches)

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate

python scripts/train_stage1.py \
    --data-dir outputs/dataset_small \
    --output-dir outputs/stage1_training \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4

# Expected time with GPU: ~30-45 minutes
# Expected time with CPU: ~6-8 hours
```

#### Option B: Train on All Data (if large datasets completed)

```bash
# First, combine datasets
mkdir -p outputs/dataset_combined
cp -r outputs/dataset_small/*/*.h5 outputs/dataset_combined/
cp -r outputs/dataset_large/*/*.h5 outputs/dataset_combined/

# Then train
python scripts/train_stage1.py \
    --data-dir outputs/dataset_combined \
    --output-dir outputs/stage1_training_full \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4

# Expected: 200+ patches
# Expected time with GPU: ~2-3 hours
```

---

## Key File Locations

### Data
- **Small datasets**: `outputs/dataset_small/*/test_patches.h5`
- **Large datasets**: `outputs/dataset_large/*/test_patches.h5`
- **Thumbnails**: `outputs/*/test_patches_thumbnails/*.png`

### Code
- **Models**: `models/stage1_spectral/`, `models/stage2_spatial/`
- **Data processing**: `data/envi_loader.py`, `data/synthetic.py`, `data/landsat_srf.py`
- **Training**: `scripts/train_stage1.py`
- **Utilities**: `utils/losses.py`, `utils/dataloader.py`

### Outputs (Created During Training)
- **Checkpoints**: `outputs/stage1_training/checkpoints/*.pth`
- **Best model**: `outputs/stage1_training/checkpoints/best_model.pth`
- **TensorBoard logs**: `outputs/stage1_training/tensorboard/`
- **Config**: `outputs/stage1_training/config.json`

### Documentation
- **Training guide**: `TRAINING.md`
- **Technical report**: `report/main.tex`
- **References**: `report/references.bib`

---

## Monitoring Training Progress

### TensorBoard (Real-time)
```bash
# Open browser: http://localhost:6006
# View:
# - Loss/train and Loss/val curves
# - Learning rate schedule
# - Individual loss components (L1, SAM, spectral_grad)
```

### Terminal Output
- Loss printed every 5 batches
- Epoch summary after each epoch
- Learning rate reductions announced
- Checkpoint saving notifications

### Log Files
```bash
# Training will print to terminal
# You can also redirect to a log file:
python scripts/train_stage1.py \
    --data-dir outputs/dataset_small \
    --output-dir outputs/stage1_training \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    2>&1 | tee outputs/stage1_training.log
```

---

## Expected Training Metrics

### Stage 1: Spectral SR (7→340 bands)

**Initial values** (epoch 0):
- L1 loss: ~0.2-0.3
- SAM loss: ~1.5-1.6 radians (~85-90°)
- Spectral gradient loss: ~0.1-0.2
- Total loss: ~0.4-0.5

**Good convergence** (after 50-100 epochs):
- L1 loss: <0.05
- SAM loss: <0.5 radians (<30°)
- Spectral gradient loss: <0.02
- Total loss: <0.1

**Validation loss** should track training loss closely (no overfitting with 36 patches).

---

## What to Do If Things Go Wrong

### GPU Still Not Working
```bash
# Check driver again
nvidia-smi

# If still broken, try reloading modules (no reboot):
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# Verify
nvidia-smi
```

### Training Crashes
```bash
# Reduce batch size
python scripts/train_stage1.py \
    --batch-size 4  # or even 2

# Or use lightweight model
# (Edit train_stage1.py to use smaller num_groups/num_blocks)
```

### Out of Memory
```bash
# Use CPU if GPU OOM
# PyTorch will automatically fall back to CPU

# Or reduce model size:
# In train_stage1.py, change:
# model = SpectralSRNet(num_groups=2, num_blocks_per_group=2)  # Smaller
```

### Large Datasets Failed
```bash
# Check log
tail -100 outputs/large_dataset_generation.log

# If failed, you can still train on 36 small patches
# Results will be preliminary but valid for testing
```

---

## Next Steps After Stage 1 Training

1. **Evaluate Stage 1 Results**
   ```bash
   # Load best model and generate predictions
   # Compare spectral curves
   # Calculate SAM, RMSE per band
   ```

2. **Train Stage 2: Spatial SR**
   - Use Stage 1 outputs as input
   - 340 bands @ 30m → 340 bands @ 4m
   - Expected: 45-60 minutes on GPU

3. **End-to-End Evaluation**
   - Full pipeline: 7 Landsat bands → 340 AVIRIS bands @ 4m resolution
   - Compare with ground truth AVIRIS
   - Generate figures for report

4. **Complete Technical Report**
   - Add results to Section 4
   - Create figures (architecture, results, comparisons)
   - Write discussion and conclusion
   - Submit to SPIE or other venue

5. **Optional: Lambda Labs**
   - Transfer code/data to cloud GPU
   - Faster iterations
   - Cost: ~$1/hour

---

## Quick Reference Commands

### Activate Environment
```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate
```

### Check Training Status
```bash
# See running processes
ps aux | grep train_stage1.py

# Monitor GPU usage
watch -n 1 nvidia-smi

# View latest checkpoint
ls -lht outputs/stage1_training/checkpoints/ | head -5
```

### Resume Training from Checkpoint
```bash
python scripts/train_stage1.py \
    --data-dir outputs/dataset_small \
    --output-dir outputs/stage1_training \
    --checkpoint outputs/stage1_training/checkpoints/checkpoint_epoch_050.pth \
    --epochs 100
```

### Test Model Inference
```bash
python scripts/test_training_pipeline.py
```

### Commit Progress to Git
```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
git add outputs/stage1_training/config.json
git commit -m "Training completed: Stage 1 spectral SR"
git push origin main
```

---

## Repository State

- **Branch**: main
- **Latest commit**: Citation fixes
- **Remote**: https://github.com/judy93536/landsat-aviris-sr.git
- **All code pushed**: ✅
- **Ready for Lambda Labs**: ✅

---

## Contact/Resources

- **GitHub Issues**: https://github.com/judy93536/landsat-aviris-sr/issues
- **SPIE LaTeX Template**: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/authors
- **RCAN Paper**: Zhang et al., ECCV 2018
- **SAM Reference**: Kruse et al., Remote Sensing of Environment 1993

---

## Summary

**You are ready to:**
1. ✅ Reboot to fix GPU driver
2. ✅ Verify CUDA works
3. ✅ Check large dataset generation
4. ✅ Start Stage 1 training (30-45 min on GPU)
5. ✅ Monitor via TensorBoard
6. ✅ Proceed to Stage 2 after convergence

**All code is working, tested, and committed to git.**

Good luck with training! 🚀
