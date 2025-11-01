# AVIRIS-NG Training on Lambda Labs H100

**Dataset**: 4 high-quality AVIRIS-NG flights (~25.4GB, 331 patches) stored in S3
**Goal**: Train 3D U-Net for Landsat (7 bands) → AVIRIS-NG (359 bands) super-resolution
**GPU**: H100 80GB ($2.49/hr)
**Estimated Time**: 2-3 hours (~$5-7.50)

---

## Prerequisites

✅ 4 selected .h5 files ready to upload:
  - `ang20230627t163014_patches.h5` (12GB, 156 patches, info=2.592)
  - `ang20230627t164727_patches.h5` (7.8GB, 102 patches, info=2.614)
  - `ang20230627t234004_patches.h5` (2.6GB, 34 patches, info=2.681)
  - `ang20230627t232845_patches.h5` (3.0GB, 39 patches, info=2.668)

✅ Git repo: https://github.com/judy93536/landsat-aviris-sr.git
✅ Lambda Labs account with SSH key: `~/.ssh/lambda_key.pem`

---

## Step 0: Upload Data to S3 (Run on ubuntu-z8 FIRST)

```bash
cd /raid/MSI_MSI_AIML/landsat-aviris-sr

# Upload only the 4 selected high-quality files
aws s3 sync outputs/aviris_ng_patches/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/aviris_ng_patches/ \
  --exclude "*" \
  --include "ang20230627t163014_patches.h5" \
  --include "ang20230627t164727_patches.h5" \
  --include "ang20230627t234004_patches.h5" \
  --include "ang20230627t232845_patches.h5"

# Verify upload
aws s3 ls s3://msi-hsi-cnn/landsat-aviris-sr/aviris_ng_patches/ --human-readable --recursive
```

Expected: 4 .h5 files, ~25.4GB total

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
pip3 install h5py spectral matplotlib tensorboard scipy scikit-learn

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
ls -la models/
```

---

## Step 4: Download Selected Flights from S3 (~8-12 min)

```bash
# Create directory
mkdir -p outputs/aviris_ng_flights

# Download 4 selected flight files (FAST from AWS network!)
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/aviris_ng_patches/ \
  outputs/aviris_ng_flights/ \
  --exclude "*" \
  --include "ang20230627t163014_patches.h5" \
  --include "ang20230627t164727_patches.h5" \
  --include "ang20230627t234004_patches.h5" \
  --include "ang20230627t232845_patches.h5"

# Verify download
ls -lh outputs/aviris_ng_flights/
du -sh outputs/aviris_ng_flights/
# Expected: 4 .h5 files, ~25.4GB total
```

---

## Step 5: Merge Flights into Single Dataset (~2-3 min)

**First, check if merge script exists:**
```bash
ls scripts/merge_flights.py
```

If it exists, merge the files:
```bash
python scripts/merge_flights.py \
  --input-dir outputs/aviris_ng_flights \
  --output outputs/aviris_ng_merged.h5

# Verify merged file
ls -lh outputs/aviris_ng_merged.h5
# Expected: ~25GB, 331 patches

# Check contents
python -c "
import h5py
with h5py.File('outputs/aviris_ng_merged.h5', 'r') as f:
    print(f'Patches: {len(f[\"aviris\"])}')
    print(f'Landsat shape: {f[\"landsat\"].shape}')
    print(f'AVIRIS shape: {f[\"aviris\"].shape}')
"
```

**Alternative if merge script doesn't exist:**
Train on each file separately and combine results later (see Alternative Step 6 below).

---

## Step 6: Train 3D U-Net (~90-120 min)

**Option A: Train on merged dataset (if Step 5 worked)**

```bash
# Train 3D U-Net: End-to-end joint spatial-spectral learning
python scripts/train_3d_unet.py \
  --data outputs/aviris_ng_merged.h5 \
  --output-dir outputs/3d_unet_ng_h100 \
  --epochs 150 \
  --batch-size 8 \
  --lightweight \
  --base-features 16 \
  --lr 5e-4 \
  2>&1 | tee 3d_unet_ng_h100.log

# Monitor training in another terminal
tail -f 3d_unet_ng_h100.log

# Check results
tail -30 3d_unet_ng_h100.log
ls -lh outputs/3d_unet_ng_h100/checkpoints/best_model.pth
```

**Option B: Train on largest file first (if merge didn't work)**

```bash
# Train on largest dataset first
python scripts/train_3d_unet.py \
  --data outputs/aviris_ng_flights/ang20230627t163014_patches.h5 \
  --output-dir outputs/3d_unet_ng_163014 \
  --epochs 150 \
  --batch-size 8 \
  --lightweight \
  --base-features 16 \
  --lr 5e-4 \
  2>&1 | tee 3d_unet_ng_163014.log
```

**Training Parameters Explained:**
- `--epochs 150`: Longer training for better convergence
- `--batch-size 8`: H100 has 80GB, can handle larger batches than local GPU
- `--lightweight`: Use efficient 1.2M parameter model
- `--base-features 16`: Balance between capacity and speed
- `--lr 5e-4`: Higher LR works well based on local test

**Expected Performance:**
- Training time: 90-120 min on H100 (331 patches, 359 output bands)
- Memory usage: ~50-60GB (H100 has 80GB)
- Final validation loss: ~0.10-0.15 (combined L1 + SAM + spectral gradient)
- LR will reduce automatically if plateaus (patience=10 epochs)

**Monitor progress:**
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# Check recent training progress
tail -20 3d_unet_ng_h100.log
```

---

## Step 7: Upload Results to S3 (~3-5 min)

```bash
# Upload 3D U-Net checkpoints
aws s3 sync outputs/3d_unet_ng_h100/checkpoints/ \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_ng/3d_unet_h100/

# Upload training log
aws s3 cp 3d_unet_ng_h100.log \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_ng/

# Upload config
aws s3 cp outputs/3d_unet_ng_h100/config.json \
  s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_ng/

# Verify upload
aws s3 ls s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_ng/ --recursive --human-readable
```

---

## Step 8: Download Results to ubuntu-z8

**On ubuntu-z8**:

```bash
# Download all results
aws s3 sync s3://msi-hsi-cnn/landsat-aviris-sr/results/aviris_ng/ \
  /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_ng_results/

# Check downloads
ls -lh /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_ng_results/
ls -lh /raid/MSI_MSI_AIML/landsat-aviris-sr/outputs/lambda_ng_results/3d_unet_h100/
```

---

## Step 9: Terminate Lambda Instance

1. Exit SSH: `exit`
2. Go to Lambda Labs dashboard
3. **Terminate** the instance (don't just stop it)
4. Verify billing

**Total Cost**: ~2.5 hours × $2.49/hr = **~$6.25**

---

## Timeline Summary

| Step | Time | Cost |
|------|------|------|
| Instance setup + deps | 3 min | $0.12 |
| Download from S3 | 12 min | $0.50 |
| Merge datasets | 3 min | $0.12 |
| Train 3D U-Net | 120 min | $4.98 |
| Upload to S3 | 5 min | $0.21 |
| **Total** | **~2.5 hrs** | **~$5.93** |

---

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
python scripts/train_3d_unet.py --batch-size 4  # or 2
```

### S3 Download Slow
Check AWS credentials:
```bash
aws configure list
aws s3 ls s3://msi-hsi-cnn/
```

### Training Diverging
Lower learning rate:
```bash
python scripts/train_3d_unet.py --lr 1e-4  # or 5e-5
```

### Merge Script Doesn't Exist
Train on individual files or create merge script:
```bash
# Quick merge script
python << 'EOF'
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

input_files = [
    'outputs/aviris_ng_flights/ang20230627t163014_patches.h5',
    'outputs/aviris_ng_flights/ang20230627t164727_patches.h5',
    'outputs/aviris_ng_flights/ang20230627t234004_patches.h5',
    'outputs/aviris_ng_flights/ang20230627t232845_patches.h5'
]

# Count total patches
total_patches = 0
for f in input_files:
    with h5py.File(f, 'r') as hf:
        total_patches += len(hf['aviris'])
print(f"Total patches: {total_patches}")

# Get dimensions from first file
with h5py.File(input_files[0], 'r') as hf:
    aviris_shape = hf['aviris'][0].shape
    landsat_shape = hf['landsat'][0].shape
print(f"AVIRIS shape: {aviris_shape}")
print(f"Landsat shape: {landsat_shape}")

# Create output file
with h5py.File('outputs/aviris_ng_merged.h5', 'w') as out_f:
    # Create datasets
    aviris_ds = out_f.create_dataset(
        'aviris', shape=(total_patches, *aviris_shape),
        dtype=np.float32, compression='gzip', compression_opts=4
    )
    landsat_ds = out_f.create_dataset(
        'landsat', shape=(total_patches, *landsat_shape),
        dtype=np.float32, compression='gzip', compression_opts=4
    )

    # Copy patches
    idx = 0
    for input_file in tqdm(input_files):
        with h5py.File(input_file, 'r') as in_f:
            n = len(in_f['aviris'])
            aviris_ds[idx:idx+n] = in_f['aviris'][:]
            landsat_ds[idx:idx+n] = in_f['landsat'][:]
            idx += n

    # Store metadata
    out_f.attrs['num_patches'] = total_patches
    out_f.attrs['n_files'] = len(input_files)

print(f"Created merged file: outputs/aviris_ng_merged.h5")
EOF
```

---

## After Training: Run Evaluation

Once you've downloaded the model to ubuntu-z8:

```bash
# Activate local environment
cd /raid/MSI_MSI_AIML/landsat-aviris-sr
source venv/bin/activate

# Evaluate on one of the test files
python scripts/evaluate_model.py \
  --checkpoint outputs/lambda_ng_results/3d_unet_h100/best_model.pth \
  --data outputs/aviris_ng_patches/ang20230627t175226_patches.h5 \
  --output-dir outputs/ng_evaluation \
  --lightweight --base-features 16

# Generate visualizations
# TODO: Create visualization script for spectral signatures, SAM maps, etc.
```

---

## Key Differences from AVIRIS Classic

| Feature | AVIRIS Classic | AVIRIS-NG |
|---------|----------------|-----------|
| **Output bands** | 198 | 359 |
| **Input spatial res** | 14.8m | 2.9-4.9m |
| **Dataset size** | 9 flights, 1096 patches | 4 flights, 331 patches |
| **Training approach** | Stage 1 + Stage 2 + 3D U-Net | 3D U-Net only |
| **Memory usage** | ~40GB | ~50-60GB (more bands) |
| **Training time** | ~4 hrs total | ~2 hrs (fewer patches) |
| **Total cost** | ~$10 | ~$6 |

---

## Notes

- Lambda H100 instances are ephemeral - all data deleted on termination
- S3 monthly cost: ~$0.60/month for 25GB dataset storage
- Can re-run training anytime for ~$6 per run
- Results stay in S3 until you delete them
- H100 80GB is ideal for AVIRIS-NG with 359 bands
- Can increase batch size further (try 12 or 16) to fully utilize 80GB VRAM
- AVIRIS-NG has better spatial resolution, making it ideal for Landsat SR task

---

## Dataset Quality Metrics

Selected files ranked by information score:

1. **ang20230627t234004** - info: 2.681 (BEST) - 34 patches
2. **ang20230627t232845** - info: 2.668 - 39 patches
3. **ang20230627t164727** - info: 2.614 - 102 patches
4. **ang20230627t163014** - info: 2.592 - 156 patches

**Combined average**: 2.613 (significantly above dataset mean of 2.485)
**Cloud coverage**: Near 0% for all selected files
**Domain**: Urban (high spectral/spatial complexity)

---

## Expected Results

Based on local validation run (39 patches, 50 epochs):
- Best validation loss: **0.152** (L1 + 0.1×SAM + 0.1×spectral_grad)
- L1 component: ~0.12 (12% mean absolute error)
- SAM component: ~0.18 (10-11° spectral angle)
- Spectral gradient: ~0.03 (smooth spectra)

With 331 patches and 150 epochs, expect:
- **Better generalization** (8× more data)
- **Lower final loss** (~0.10-0.12 combined)
- **Improved spectral fidelity** (SAM < 10°)

---

## Comparison: Local TITAN V vs Lambda H100

| Metric | TITAN V 12GB | H100 80GB |
|--------|--------------|-----------|
| **Batch size** | 2 | 8-16 |
| **Epoch time** | 15 sec | 5-7 sec |
| **50 epochs** | 12 min | 4-6 min |
| **150 epochs** | 38 min | 12-18 min |
| **Cost** | Free | $0.50-0.75 |
| **Use case** | Testing | Production |

**Recommendation**: Use TITAN V for quick validation, H100 for full training runs.
