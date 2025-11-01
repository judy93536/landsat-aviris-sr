# Lambda Labs H100 Training Guide - VNIR-5x Dataset

Complete guide for training the full U-Net model on Lambda Labs H100 GPU.

## Pre-Deployment Checklist

### 1. Local Preparation
```bash
# Merge VNIR-5x datasets
python scripts/merge_vnir_5x_datasets.py

# Upload to S3
chmod +x scripts/upload_to_s3.sh
./scripts/upload_to_s3.sh

# Push latest code to GitHub
git add .
git commit -m "Add full U-Net training for VNIR-5x dataset"
git push origin main
```

## Lambda Labs Setup

### 1. Launch H100 Instance
- Instance type: **1x H100 (80GB)**
- Region: Any available
- Expected cost: **~$2.49/hr**
- Estimated training time: **1-2 hours** for 100 epochs

### 2. SSH into Instance
```bash
ssh ubuntu@<lambda-instance-ip>
```

### 3. Clone Repository
```bash
git clone https://github.com/<your-username>/landsat-aviris-sr.git
cd landsat-aviris-sr
```

### 4. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install h5py numpy tqdm spectral scipy matplotlib

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 5. Download Dataset from S3
```bash
# Install AWS CLI (if needed)
pip install awscli

# Download merged dataset (~30GB)
mkdir -p outputs/aviris_ng_vnir_5x
aws s3 cp s3://aviris-landsat-sr-datasets/vnir_5x/merged_vnir_5x.h5 \
  outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5

# Verify download
ls -lh outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5
python -c "import h5py; f = h5py.File('outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5', 'r'); print(f'Patches: {f.attrs[\"num_patches\"]}'); print(f'Landsat shape: {f[\"landsat\"].shape}'); print(f'AVIRIS shape: {f[\"aviris\"].shape}')"
```

## Training Commands

### Option A: Full U-Net (Recommended for H100)
```bash
# Full U-Net with base_features=32
# Memory: ~20-25GB VRAM
# Batch size: 4
# Training time: ~1-2 hours for 100 epochs

python scripts/train_3d_unet.py \
  --data outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5 \
  --output-dir outputs/vnir_5x_full_unet_h100 \
  --epochs 100 \
  --batch-size 4 \
  --base-features 32 \
  --lr 5e-4 \
  --sam-weight 1.0 \
  --spectral-grad-weight 0.1 \
  --device cuda \
  --num-workers 4
```

###Option B: Larger Model (if memory allows)
```bash
# Even larger model with base_features=48
# Memory: ~35-40GB VRAM
# Batch size: 2

python scripts/train_3d_unet.py \
  --data outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5 \
  --output-dir outputs/vnir_5x_large_unet_h100 \
  --epochs 100 \
  --batch-size 2 \
  --base-features 48 \
  --lr 5e-4 \
  --sam-weight 1.0 \
  --spectral-grad-weight 0.1 \
  --device cuda \
  --num-workers 4
```

### Option C: Extended Training
```bash
# If initial results are promising, train longer
python scripts/train_3d_unet.py \
  --data outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5 \
  --output-dir outputs/vnir_5x_full_unet_extended \
  --epochs 150 \
  --batch-size 4 \
  --base-features 32 \
  --lr 5e-4 \
  --sam-weight 1.0 \
  --spectral-grad-weight 0.1 \
  --device cuda \
  --num-workers 4
```

## Monitoring Training

### In a separate terminal/tmux session:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor training logs
tail -f outputs/vnir_5x_full_unet_h100/training.log
```

### Check training progress:
```bash
# List checkpoints
ls -lh outputs/vnir_5x_full_unet_h100/checkpoints/

# Check best validation loss
python -c "import torch; ckpt = torch.load('outputs/vnir_5x_full_unet_h100/checkpoints/best_model.pth'); print(f'Best val loss: {ckpt[\"val_loss\"]:.6f}'); print(f'Epoch: {ckpt[\"epoch\"]}')"
```

## Post-Training

### 1. Visualize Results
```bash
python scripts/visualize_sr_results.py \
  --checkpoint outputs/vnir_5x_full_unet_h100/checkpoints/best_model.pth \
  --data outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5 \
  --output-dir outputs/vnir_5x_full_unet_h100_visualization \
  --n-patches 10 \
  --device cuda
```

### 2. Download Results
```bash
# On local machine:
scp -r ubuntu@<lambda-ip>:landsat-aviris-sr/outputs/vnir_5x_full_unet_h100 outputs/
scp -r ubuntu@<lambda-ip>:landsat-aviris-sr/outputs/vnir_5x_full_unet_h100_visualization outputs/
```

### 3. Upload Best Model to S3
```bash
# On Lambda instance:
aws s3 cp outputs/vnir_5x_full_unet_h100/checkpoints/best_model.pth \
  s3://aviris-landsat-sr-datasets/models/vnir_5x_full_unet_best.pth

aws s3 cp outputs/vnir_5x_full_unet_h100/config.json \
  s3://aviris-landsat-sr-datasets/models/vnir_5x_full_unet_config.json
```

## Expected Results

Based on previous experiments:

### Previous Results (Lightweight Model, SAM=1.0, 78 patches):
- Validation loss: 0.280
- Mean SAM: **9-13°** (improved from 18° with SAM=0.1)
- SAM < 5°: ~0-2%
- SAM < 10°: ~35-70%

### Target with Full U-Net + 1,000+ patches:
- Validation loss: **< 0.20** (lower is better)
- Mean SAM: **< 8°** (target: <5° for good results)
- SAM < 5°: **> 20%** (stretch goal: >50%)
- SAM < 10°: **> 80%**
- Better color accuracy and spectral fidelity

## Key Improvements Over Previous Training:
1. ✅ **13.9× more data** (1,084 patches vs 78)
2. ✅ **Larger model** (base_features=32 vs 16, ~4× more parameters)
3. ✅ **Stronger SAM penalty** (weight=1.0 to prioritize spectral shape)
4. ✅ **Simplified task** (5→120 bands, 5× spatial vs 7→359 bands, 10× spatial)
5. ✅ **H100 GPU** (2-3× faster training)

## Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size
--batch-size 2

# Or use lightweight model
# Remove --base-features 32, add --lightweight
```

### Slow Training:
```bash
# Increase num_workers
--num-workers 8

# Check data loading time vs GPU time
```

### NaN Loss:
```bash
# Reduce learning rate
--lr 1e-4

# Check data for inf/nan values
python -c "import h5py; import numpy as np; f = h5py.File('outputs/aviris_ng_vnir_5x/merged_vnir_5x.h5', 'r'); print('Landsat NaN:', np.isnan(f['landsat'][:]).any()); print('AVIRIS NaN:', np.isnan(f['aviris'][:]).any())"
```

## Cost Estimation

- H100 rate: **$2.49/hour**
- Training time: **1-2 hours**
- Data download: **~10 minutes** (30GB @ 50MB/s)
- Visualization: **~5 minutes**
- **Total cost: ~$3-5**

## Clean Up

```bash
# On Lambda instance (before terminating):
# Ensure results are backed up!
aws s3 sync outputs/vnir_5x_full_unet_h100 s3://aviris-landsat-sr-datasets/training_runs/vnir_5x_full_unet_h100/

# Terminate instance through Lambda Labs dashboard
```
