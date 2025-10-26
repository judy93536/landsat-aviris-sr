# AVIRIS Classic Training on Lambda Labs H100

**Dataset**: 1,096 patches from 9 flights (~35GB) stored in S3
**Goal**: Train Stage 1 (Spectral SR) and 3D U-Net on AVIRIS Classic data
**GPU**: H100 80GB ($2.49/hr)
**Estimated Time**: 3-4 hours (~$7-10)

---

## Prerequisites

✅ Individual flight .h5 files uploaded to S3: `s3://msi-hsi-cnn/landsat-aviris-sr/aviris_classic_flights/`
✅ Git repo: https://github.com/judy93536/landsat-aviris-sr.git
✅ Lambda Labs account with SSH key: `~/.ssh/lambda_key.pem`

---

## Step 0: Upload Data to S3 (Run on ubuntu-z8 FIRST)

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr

# Upload all flights EXCEPT the largest (f180601t01p00r06)
aws s3 sync outputs/full_dataset/flights/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/aviris_classic_flights/ \
  --exclude "f180601t01p00r06_patches.h5"

# Verify upload
aws s3 ls s3://msi-hsi-cnn/landsat-aviris-sr/aviris_classic_flights/ --human-readable --recursive
```

Expected: 9 .h5 files, ~35GB total

---

## Step 1: Launch Lambda H100 Instance

1. Go to Lambda Labs dashboard
2. Launch **H100** instance ($2.49/hr, 80GB VRAM)
3. No persistent storage needed
4. Copy SSH IP address (e.g., `150.136.145.58`)

---

## Step 2: SSH and Install Dependencies (~2 min)

```bash
# SSH into Lambda
ssh -i ~/.ssh/lambda_key.pem ubuntu@<LAMBDA_IP>

# Install Python packages
pip3 install h5py spectral matplotlib tensorboard scipy

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA H100 80GB HBM3
```

---

## Step 3: Clone Repository (~30 sec)

```bash
# Clone project
git clone https://github.com/judy93536/landsat-aviris-sr.git
cd landsat-aviris-sr

# Verify structure
ls -la scripts/
```

---

## Step 4: Download Individual Flights from S3 (~10-15 min)

```bash
# Create directory
mkdir -p outputs/aviris_flights

# Download all individual flight files (FAST from AWS network!)
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/aviris_classic_flights/ \
  outputs/aviris_flights/

# Verify download
ls -lh outputs/aviris_flights/
du -sh outputs/aviris_flights/
# Expected: 9 .h5 files, ~35GB total
```

---

## Step 5: Merge Flights into Single Dataset (~2-3 min)

```bash
# Merge all flights into one combined dataset
python scripts/merge_flights.py \
  --input-dir outputs/aviris_flights \
  --output outputs/aviris_classic_merged.h5

# Verify merged file
ls -lh outputs/aviris_classic_merged.h5
# Expected: ~35GB, 1,096 patches

# Check contents
python -c "
import h5py
with h5py.File('outputs/aviris_classic_merged.h5', 'r') as f:
    print(f'Patches: {len(f[\"aviris\"])}')
    print(f'Landsat shape: {f[\"landsat\"].shape}')
    print(f'AVIRIS shape: {f[\"aviris\"].shape}')
    print(f'Flights:', f.attrs.get('n_flights'))
"
```

---

## Step 6: Train Stage 1 - Spectral SR (~60-90 min)

```bash
# Train Stage 1: 7 → 198 bands (AVIRIS Classic)
python scripts/train_aviris_classic.py \
  --data outputs/aviris_classic_merged.h5 \
  --output-dir outputs/stage1_aviris_h100 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4 \
  2>&1 | tee stage1_aviris_h100.log

# Monitor training
tail -f stage1_aviris_h100.log

# Check results
tail -20 stage1_aviris_h100.log
ls -lh outputs/stage1_aviris_h100/checkpoints/best_model.pth
```

**Expected Performance:**
- Training time: 60-90 min on H100 (1,096 patches)
- Final SAM: < 3° (better performance with more data)
- Final RMSE: < 0.04

---

## Step 7: Train 3D U-Net - Joint Spatial-Spectral (~90-120 min)

```bash
# Train 3D U-Net: End-to-end joint learning
python scripts/train_3d_unet.py \
  --data outputs/aviris_classic_merged.h5 \
  --output-dir outputs/3d_unet_aviris_h100 \
  --epochs 150 \
  --batch-size 4 \
  --lightweight \
  --base-features 16 \
  --lr 1e-4 \
  2>&1 | tee 3d_unet_aviris_h100.log

# Monitor training
tail -f 3d_unet_aviris_h100.log

# Check results
tail -20 3d_unet_aviris_h100.log
ls -lh outputs/3d_unet_aviris_h100/checkpoints/best_model.pth
```

**Expected Performance:**
- Training time: 90-120 min on H100
- Memory usage: ~40-50GB (H100 has 80GB)
- Final RMSE: < 0.05

---

## Step 8: Upload Results to S3 (~5 min)

```bash
# Upload Stage 1 results
aws s3 sync outputs/stage1_aviris_h100/checkpoints/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/stage1_h100/

# Upload 3D U-Net results
aws s3 sync outputs/3d_unet_aviris_h100/checkpoints/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/3d_unet_h100/

# Upload training logs
aws s3 cp stage1_aviris_h100.log \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/
aws s3 cp 3d_unet_aviris_h100.log \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/

# Verify upload
aws s3 ls s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/ --recursive --human-readable
```

---

## Step 9: Download Results to ubuntu-z8

**On ubuntu-z8**:

```bash
# Download all results
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_classic/ \
  /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_aviris_results/

# Check downloads
ls -lh /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_aviris_results/
```

---

## Step 10: Terminate Lambda Instance

1. Exit SSH: `exit`
2. Go to Lambda Labs dashboard
3. **Terminate** the instance (don't just stop it)
4. Verify billing

**Total Cost**: ~3-4 hours × $2.49/hr = **~$7.50-10**

---

## Timeline Summary

| Step | Time | Cost |
|------|------|------|
| Instance setup + deps | 3 min | $0.12 |
| Download from S3 | 15 min | $0.62 |
| Merge datasets | 3 min | $0.12 |
| Train Stage 1 | 90 min | $3.74 |
| Train 3D U-Net | 120 min | $4.98 |
| Upload to S3 | 5 min | $0.21 |
| **Total** | **~4 hrs** | **~$9.79** |

---

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
# Stage 1
python scripts/train_aviris_classic.py --batch-size 16  # or 8

# 3D U-Net
python scripts/train_3d_unet.py --batch-size 2  # or 1
```

### S3 Download Slow
Check AWS credentials:
```bash
aws configure list
aws s3 ls s3://msi-hsi-cnn/
```

### Training Not Converging
Check learning rate:
```bash
python scripts/train_aviris_classic.py --lr 5e-5  # Lower LR
```

---

## After Training: Run Evaluation

Once you've downloaded the models to ubuntu-z8:

```bash
# Evaluate Stage 1 model
python scripts/evaluate_model.py \
  --checkpoint outputs/lambda_aviris_results/stage1_h100/best_model.pth \
  --data outputs/test_dataset/f180601t01p00r06_patches.h5 \
  --output-dir outputs/aviris_evaluation

# Evaluate 3D U-Net
python scripts/evaluate_3d_unet.py \
  --checkpoint outputs/lambda_aviris_results/3d_unet_h100/best_model.pth \
  --data outputs/test_dataset/f180601t01p00r06_patches.h5 \
  --output-dir outputs/3d_unet_evaluation \
  --lightweight --base-features 16
```

---

## Notes

- Lambda H100 instances are ephemeral - all data deleted on termination
- S3 monthly cost: ~$0.80/month for 35GB dataset storage
- Can re-run training anytime for ~$8-10 per run
- Results stay in S3 until you delete them
- H100 80GB is ideal for 3D U-Net with larger models
- Can increase batch sizes to utilize full 80GB VRAM

---

## Comparison: A100 vs H100

| GPU | VRAM | Price/hr | Stage 1 Time | 3D U-Net Time | Total Cost |
|-----|------|----------|--------------|----------------|------------|
| A100 40GB | 40GB | $1.29 | ~90 min | ~150 min | ~$5.16 |
| H100 80GB | 80GB | $2.49 | ~60 min | ~90 min | ~$6.23 |

**Recommendation**: H100 for production training (faster, more memory for larger batches)
