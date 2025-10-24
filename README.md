# Landsat-AVIRIS Super-Resolution

Joint spatial-spectral super-resolution of Landsat multispectral imagery to AVIRIS-3 hyperspectral quality using deep learning.

## Objective

Enhance Landsat imagery from:
- **Spatial**: 30m → 4m (~7.5x upsampling)
- **Spectral**: 7-11 bands → 224 bands

Using 3D deep learning to achieve AVIRIS-3 quality hyperspectral imagery.

## Project Structure

```
landsat-aviris-sr/
├── data/                      # Data processing modules
│   ├── landsat_srf.py        # Landsat spectral response functions
│   ├── synthetic.py          # AVIRIS→Landsat simulation pipeline
│   └── augmentation.py       # Noise models and atmospheric effects
├── models/                    # Neural network architectures
│   ├── unet3d.py             # 3D U-Net for spatial-spectral SR
│   └── losses.py             # Loss functions (RMSE + spectral)
├── scripts/                   # Utility scripts
├── notebooks/                 # Jupyter notebooks for exploration
├── config/                    # Configuration files
└── outputs/                   # Generated datasets and checkpoints
    ├── synthetic/            # Synthetic Landsat-AVIRIS pairs
    ├── real/                 # Real paired data
    └── checkpoints/          # Model weights
```

## Setup

### Local Development (V100)

```bash
# Clone repository
git clone <your-repo-url>
cd landsat-aviris-sr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Lambda Labs GPU

```bash
# SSH to Lambda Labs instance
ssh -i <key> ubuntu@<instance-ip>

# Clone and setup
git clone <your-repo-url>
cd landsat-aviris-sr
pip install -r requirements.txt
```

## Workflow

### Phase 1: Synthetic Data Generation
1. Load AVIRIS radiance data (4m, 224 bands)
2. Simulate Landsat using SRFs and spatial downsampling
3. Add realistic noise, artifacts, and atmospheric variations
4. Generate training dataset

### Phase 2: Model Training
1. Train 3D U-Net on synthetic pairs
2. Progressive training with increasing realism
3. Fine-tune on limited real paired data
4. Validate on hold-out real pairs

### Phase 3: Evaluation
- Spatial metrics: PSNR, SSIM
- Spectral metrics: SAM, RMSE
- Visual inspection of spectral signatures

## Technical Details

**Working in Radiance Space**
- Using calibrated radiance (not reflectance)
- Avoids atmospheric correction artifacts
- Consistent with sensor measurements

**Loss Function**
- Primary: RMSE (validated against IEEE papers)
- Optional: Spectral smoothness regularization

**Key References**
- Landsat SRFs: [USGS Spectral Characteristics Viewer](https://www.usgs.gov/landsat-missions/spectral-characteristics-viewer)
- Barsi et al. (2014) "The spectral response of the Landsat-8 Operational Land Imager"
- Green et al. (1998), Thompson et al. (2015) for AVIRIS-NG

## Development

Developed locally on Tesla V100 (12GB), trained on Lambda Labs high-performance GPUs.

## License

TBD
