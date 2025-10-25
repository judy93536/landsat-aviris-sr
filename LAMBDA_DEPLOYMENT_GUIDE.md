# Lambda Labs Deployment Guide
## Complete Two-Stage Training Pipeline

**Last Updated**: October 25, 2025
**Dataset**: 541 patches (~41GB) stored in S3: `s3://msi-hsi-cnn/landsat-aviris-sr/`
**Estimated Lambda Time**: 2-3 hours (~$3-4)

---

## Prerequisites

✅ Datasets uploaded to S3: `s3://msi-hsi-cnn/landsat-aviris-sr/`
✅ Git repo: https://github.com/judy93536/landsat-aviris-sr.git
✅ Lambda Labs account with SSH key: `~/.ssh/lambda_key.pem`

---

## Step 1: Launch Lambda A100 Instance

1. Go to Lambda Labs dashboard
2. Launch **A100 40GB** instance ($1.29/hr)
3. No persistent storage needed
4. Copy SSH IP address (e.g., `150.136.145.58`)

---

## Step 2: SSH and Install Dependencies (~2 min)

```bash
# SSH into Lambda
ssh -i ~/.ssh/lambda_key.pem ubuntu@<LAMBDA_IP>

# Install Python packages
pip3 install h5py spectral matplotlib tensorboard

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA A100-SXM4-40GB
```

---

## Step 3: Clone Repository (~30 sec)

```bash
# Clone project
git clone https://github.com/judy93536/landsat-aviris-sr.git
cd landsat-aviris-sr

# Verify structure
ls -la
```

---

## Step 4: Download Data from S3 (~5-10 min)

```bash
# Create output directory
mkdir -p outputs

# Download datasets (FAST from AWS network!)
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/dataset_large outputs/dataset_large/
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/dataset_small outputs/dataset_small/

# Verify download
du -sh outputs/dataset_*
# Expected: ~38GB dataset_large, ~2.7GB dataset_small
```

---

## Step 5: Combine Datasets (~1 min)

```bash
# Combine all patches into one directory
mkdir -p outputs/dataset_combined

# Copy HDF5 files
find outputs/dataset_small -name "test_patches.h5" -exec cp {} outputs/dataset_combined/ \; 2>/dev/null
find outputs/dataset_large -name "test_patches.h5" -exec cp {} outputs/dataset_combined/ \; 2>/dev/null

# Rename to avoid conflicts
cd outputs/dataset_combined
COUNT=0
for f in test_patches.h5; do
    mv "$f" "patches_${COUNT}.h5" 2>/dev/null || true
    COUNT=$((COUNT+1))
done
cd ../..

# Verify
ls -lh outputs/dataset_combined/
# Expected: 5 HDF5 files (patches_0.h5 through patches_4.h5)
```

---

## Step 6: Train Stage 1 - Spectral SR (~45-60 min)

```bash
# Train Stage 1: 7 → 340 bands
python scripts/train_stage1.py \
  --data-dir outputs/dataset_combined \
  --output-dir outputs/stage1_full \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  2>&1 | tee stage1_full.log

# Check results
tail -20 stage1_full.log
ls -lh outputs/stage1_full/checkpoints/best_model.pth
```

**Expected Performance:**
- Training time: 45-60 min on A100
- Final SAM: < 5° (better than 5.89° from 36-patch training)
- Final RMSE: < 0.05

---

## Step 7: Train Stage 2 - Spatial SR (~45-60 min)

```bash
# Train Stage 2: Spatial refinement
python scripts/train_stage2.py \
  --data-dir outputs/dataset_combined \
  --stage1-checkpoint outputs/stage1_full/checkpoints/best_model.pth \
  --output-dir outputs/stage2_full \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  2>&1 | tee stage2_full.log

# Check results
tail -20 stage2_full.log
ls -lh outputs/stage2_full/checkpoints/best_model.pth
```

**Expected Performance:**
- Training time: 45-60 min on A100
- Learns spatial detail refinement
- Final loss: < 0.04

---

## Step 8: Upload Results to S3 (~2 min)

```bash
# Upload Stage 1 results
aws s3 sync outputs/stage1_full/checkpoints/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/stage1_full/

# Upload Stage 2 results
aws s3 sync outputs/stage2_full/checkpoints/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/stage2_full/

# Upload training logs
aws s3 cp stage1_full.log s3://msi-hsi-cnn/landsat-aviris-sr/results/
aws s3 cp stage2_full.log s3://msi-hsi-cnn/landsat-aviris-sr/results/

# Verify upload
aws s3 ls s3://msi-hsi-cnn/landsat-aviris-sr/results/ --recursive --human-readable
```

---

## Step 9: Download Results to ubuntu-z8

**On ubuntu-z8**:

```bash
# Download all results
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/results/ \
  /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_results/

# Check downloads
ls -lh /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_results/
```

---

## Step 10: Terminate Lambda Instance

1. Exit SSH: `exit`
2. Go to Lambda Labs dashboard
3. **Terminate** the instance (don't just stop it)
4. Verify billing

**Total Cost**: ~2-3 hours × $1.29/hr = **~$3-4**

---

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
# Stage 1
python scripts/train_stage1.py --batch-size 8  # or 4

# Stage 2
python scripts/train_stage2.py --batch-size 4  # or 2
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
python scripts/train_stage1.py --lr 5e-5  # Lower LR
```

---

## Timeline Summary

| Step | Time | Cost |
|------|------|------|
| Instance setup + deps | 3 min | $0.06 |
| Download from S3 | 10 min | $0.22 |
| Combine datasets | 1 min | $0.02 |
| Train Stage 1 | 60 min | $1.29 |
| Train Stage 2 | 60 min | $1.29 |
| Upload to S3 | 2 min | $0.04 |
| **Total** | **~2.5 hrs** | **~$3.22** |

---

## After Training: Run Inference

Once you've downloaded the models to ubuntu-z8:

```bash
# Generate enhanced HSI from test patch
python scripts/inference_stage1.py \
  --checkpoint outputs/lambda_results/stage1_full/best_model.pth \
  --data-file outputs/dataset_small/ang20190624t214359_rdn_v2u1/test_patches.h5 \
  --patch-idx 0 \
  --output-dir outputs/final_results
```

---

## Notes

- Lambda instances are ephemeral - all data deleted on termination
- S3 monthly cost: ~$1/month for 41GB dataset storage
- Can re-run training anytime for $3-4 per run
- Results stay in S3 until you delete them
