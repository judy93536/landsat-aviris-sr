"""
Synthetic Landsat-AVIRIS Pair Generation

Simulates realistic Landsat imagery from high-resolution AVIRIS hyperspectral data
through spatial downsampling and spectral integration.

Pipeline:
1. Load AVIRIS radiance cube (4m, 224 bands)
2. Apply spatial degradation (PSF + downsampling) to 30m
3. Apply spectral integration using Landsat SRFs
4. Generate paired patches for training

References:
- Spatial degradation: Gaussian PSF followed by subsampling
- Spectral integration: Weighted integration using sensor SRFs
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import convolve2d
import h5py
from pathlib import Path
import warnings
from typing import Tuple, Optional, List

from .landsat_srf import LandsatSRF, load_aviris_wavelengths
from .envi_loader import load_aviris_ng


class SyntheticDataGenerator:
    """
    Generate synthetic Landsat-AVIRIS paired data for training.
    """

    def __init__(
        self,
        spatial_downsampling_factor=7.5,
        psf_sigma=2.0,
        landsat_sensor="LC08",
        landsat_bands=None,
    ):
        """
        Initialize synthetic data generator.

        Parameters:
        -----------
        spatial_downsampling_factor : float
            Ratio of Landsat to AVIRIS resolution (30m / 4m = 7.5)
        psf_sigma : float
            Standard deviation of Gaussian PSF for spatial degradation
        landsat_sensor : str
            Landsat sensor identifier ('LC08' for Landsat-8)
        landsat_bands : list or None
            Landsat bands to simulate. If None, use all available bands.
        """
        self.downsampling_factor = spatial_downsampling_factor
        self.psf_sigma = psf_sigma

        # Initialize Landsat SRF loader
        self.srf = LandsatSRF(sensor=landsat_sensor)

        if landsat_bands is None:
            self.landsat_bands = sorted(self.srf.get_all_bands_info().keys())
        else:
            self.landsat_bands = landsat_bands

        print(f"Synthetic generator initialized:")
        print(f"  Downsampling factor: {self.downsampling_factor}x")
        print(f"  PSF sigma: {self.psf_sigma}")
        print(f"  Landsat bands: {self.landsat_bands}")

    def spatial_degrade(self, image):
        """
        Apply spatial degradation: Gaussian blur + downsampling.

        Parameters:
        -----------
        image : np.ndarray
            High-resolution image (shape: [height, width, bands])

        Returns:
        --------
        degraded : np.ndarray
            Spatially degraded low-resolution image
        """
        height, width, n_bands = image.shape

        # Calculate output dimensions
        out_height = int(height / self.downsampling_factor)
        out_width = int(width / self.downsampling_factor)

        # Initialize output
        degraded = np.zeros((out_height, out_width, n_bands), dtype=image.dtype)

        # Apply blur and downsample for each band
        for b in range(n_bands):
            # Apply Gaussian PSF
            blurred = gaussian_filter(
                image[:, :, b],
                sigma=self.psf_sigma,
                mode='reflect'
            )

            # Downsample using scipy.ndimage.zoom with exact output shape
            zoom_factor = 1.0 / self.downsampling_factor
            downsampled = zoom(
                blurred,
                zoom_factor,
                order=1,  # Bilinear interpolation
                mode='reflect'
            )

            # Ensure exact output dimensions (crop or pad if needed due to rounding)
            actual_h, actual_w = downsampled.shape
            if actual_h > out_height:
                downsampled = downsampled[:out_height, :]
            if actual_w > out_width:
                downsampled = downsampled[:, :out_width]
            if actual_h < out_height or actual_w < out_width:
                # Pad if too small (shouldn't happen but just in case)
                padded = np.zeros((out_height, out_width), dtype=image.dtype)
                padded[:actual_h, :actual_w] = downsampled
                downsampled = padded

            degraded[:, :, b] = downsampled

        return degraded

    def spectral_integrate(self, hyperspectral_cube, wavelengths):
        """
        Integrate hyperspectral cube to Landsat bands using SRFs.

        Parameters:
        -----------
        hyperspectral_cube : np.ndarray
            AVIRIS radiance cube (shape: [height, width, n_bands])
        wavelengths : np.ndarray
            Wavelength array for AVIRIS bands (in nm)

        Returns:
        --------
        landsat_cube : np.ndarray
            Simulated Landsat imagery (shape: [height, width, n_landsat_bands])
        """
        return self.srf.integrate_cube(
            hyperspectral_cube,
            wavelengths,
            band_numbers=self.landsat_bands
        )

    def generate_synthetic_landsat(
        self,
        aviris_cube,
        aviris_wavelengths,
        return_intermediate=False
    ):
        """
        Generate synthetic Landsat from AVIRIS data.

        Correct Pipeline (spectral→spatial→upsample):
        1. Spectral integration (AVIRIS 340 bands → Landsat 7 bands at 4m)
        2. Spatial degradation (blur + downsample 7 bands from 4m to 30m)
        3. Upsample back to original dimensions for pixel alignment

        Parameters:
        -----------
        aviris_cube : np.ndarray
            AVIRIS radiance cube (shape: [height, width, n_bands])
        aviris_wavelengths : np.ndarray
            Wavelength array for AVIRIS bands (in nm)
        return_intermediate : bool
            If True, return intermediate high-res Landsat cube

        Returns:
        --------
        synthetic_landsat_upsampled : np.ndarray
            Upsampled Landsat (shape: [height, width, n_landsat_bands])
        high_res_landsat : np.ndarray (optional)
            High-res Landsat before downsampling
        """
        print(f"Generating synthetic Landsat from AVIRIS cube: {aviris_cube.shape}")

        # Step 1: Spectral integration FIRST (340 bands → 7 bands, same resolution)
        print("  Step 1: Applying spectral integration (SRF)...")
        high_res_landsat = self.spectral_integrate(
            aviris_cube,
            aviris_wavelengths
        )
        print(f"    High-res Landsat shape: {high_res_landsat.shape}")

        # Step 2: Spatial degradation (downsample only 7 bands - much faster!)
        print("  Step 2: Applying spatial degradation...")
        low_res_landsat = self.spatial_degrade(high_res_landsat)
        print(f"    Low-res Landsat shape: {low_res_landsat.shape}")

        # Step 3: Upsample back to original dimensions for pixel-wise correspondence
        print("  Step 3: Upsampling to original dimensions...")
        from scipy.ndimage import zoom

        height_hr, width_hr = aviris_cube.shape[0], aviris_cube.shape[1]
        height_lr, width_lr = low_res_landsat.shape[0], low_res_landsat.shape[1]

        # Calculate zoom factors
        zoom_h = height_hr / height_lr
        zoom_w = width_hr / width_lr

        # Upsample each band with bicubic interpolation
        synthetic_landsat_upsampled = np.zeros(
            (height_hr, width_hr, low_res_landsat.shape[2]),
            dtype=np.float32
        )

        for b in range(low_res_landsat.shape[2]):
            synthetic_landsat_upsampled[:, :, b] = zoom(
                low_res_landsat[:, :, b],
                (zoom_h, zoom_w),
                order=3,  # Bicubic interpolation
                mode='reflect'
            )

        print(f"    Upsampled Landsat shape: {synthetic_landsat_upsampled.shape}")

        if return_intermediate:
            return synthetic_landsat_upsampled, high_res_landsat
        else:
            return synthetic_landsat_upsampled

    def find_valid_region(self, aviris_cube, ignore_value=-9999):
        """
        Find the largest rectangular region with minimal invalid pixels.

        Strategy: Create a validity map across all bands, then find the
        largest rectangle where most pixels are valid.

        Parameters:
        -----------
        aviris_cube : np.ndarray
            AVIRIS cube (shape: [height, width, bands])
        ignore_value : float
            Data ignore value

        Returns:
        --------
        bounds : tuple
            (row_start, row_end, col_start, col_end) of valid region
        validity_map : np.ndarray
            2D boolean map showing valid pixels
        """
        height, width, n_bands = aviris_cube.shape

        # Create validity map: pixel is valid if it's valid in ALL bands
        # Check a subset of representative bands to speed this up
        # Check every 10th band plus first/last
        check_bands = list(range(0, n_bands, 10)) + [n_bands - 1]

        print(f"  Checking {len(check_bands)} representative bands for validity...")
        validity_map = np.ones((height, width), dtype=bool)

        for band_idx in check_bands:
            band = aviris_cube[:, :, band_idx]
            band_valid = (
                np.isfinite(band) &
                (band != ignore_value) &
                (band > -1000) &  # More lenient lower bound
                (band < 1e6)
            )
            validity_map &= band_valid

        # Find valid fraction
        valid_frac = validity_map.mean()
        print(f"  Overall valid fraction: {valid_frac*100:.2f}%")

        # Find row bounds (exclude rows with <50% valid pixels)
        row_valid_frac = validity_map.mean(axis=1)
        valid_rows = np.where(row_valid_frac > 0.5)[0]

        if len(valid_rows) == 0:
            print("  WARNING: No valid rows found!")
            return (0, height, 0, width), validity_map

        row_start = valid_rows[0]
        row_end = valid_rows[-1] + 1

        # Find column bounds (exclude columns with <50% valid pixels)
        col_valid_frac = validity_map.mean(axis=0)
        valid_cols = np.where(col_valid_frac > 0.5)[0]

        if len(valid_cols) == 0:
            print("  WARNING: No valid columns found!")
            return (row_start, row_end, 0, width), validity_map

        col_start = valid_cols[0]
        col_end = valid_cols[-1] + 1

        print(f"  Valid region: rows [{row_start}:{row_end}] ({row_end-row_start} rows), "
              f"cols [{col_start}:{col_end}] ({col_end-col_start} cols)")

        return (row_start, row_end, col_start, col_end), validity_map

    def extract_patches(
        self,
        aviris_cube,
        synthetic_landsat,
        patch_size_hr=256,
        stride=None,
        min_valid_fraction=0.95,
        ignore_value=-9999
    ):
        """
        Extract paired patches from AVIRIS and synthetic Landsat.

        Parameters:
        -----------
        aviris_cube : np.ndarray
            High-res AVIRIS cube (shape: [height_hr, width_hr, n_bands])
        synthetic_landsat : np.ndarray
            Low-res synthetic Landsat (shape: [height_lr, width_lr, n_landsat_bands])
        patch_size_hr : int
            Patch size at AVIRIS resolution
        stride : int or None
            Stride for patch extraction. If None, use patch_size_hr (no overlap)
        min_valid_fraction : float
            Minimum fraction of valid pixels required (0.0-1.0)
        ignore_value : float
            Data ignore value from ENVI header (typically -9999)

        Returns:
        --------
        patches : list of dict
            List of patch pairs: [{"aviris": patch_hr, "landsat": patch_lr}, ...]
        """
        if stride is None:
            stride = patch_size_hr

        # First, find the valid rectangular region
        print("\nFinding valid image region...")
        bounds, validity_map = self.find_valid_region(aviris_cube, ignore_value)
        row_start, row_end, col_start, col_end = bounds

        # Crop to valid region
        aviris_cropped = aviris_cube[row_start:row_end, col_start:col_end, :]

        # Crop synthetic Landsat to corresponding region
        # Note: synthetic_landsat was already upsampled back to HR size, so use HR coordinates
        landsat_cropped = synthetic_landsat[row_start:row_end, col_start:col_end, :]
        validity_cropped = validity_map[row_start:row_end, col_start:col_end]

        height_hr, width_hr, _ = aviris_cropped.shape
        patch_size_lr = int(patch_size_hr / self.downsampling_factor)

        patches = []
        total_attempts = 0

        print(f"\nExtracting {patch_size_hr}x{patch_size_hr} patches with stride={stride}...")

        # Extract patches with sliding window from cropped region
        for i in range(0, height_hr - patch_size_hr + 1, stride):
            for j in range(0, width_hr - patch_size_hr + 1, stride):
                total_attempts += 1

                # Check validity map for this patch region
                patch_mask = validity_cropped[i:i+patch_size_hr, j:j+patch_size_hr]
                valid_fraction = patch_mask.mean()

                if valid_fraction < min_valid_fraction:
                    continue  # Skip patches with insufficient valid pixels

                # Extract high-res patch
                patch_hr = aviris_cropped[i:i+patch_size_hr, j:j+patch_size_hr, :]

                # Extract corresponding Landsat patch (same coordinates since upsampled)
                patch_lr = landsat_cropped[i:i+patch_size_hr, j:j+patch_size_hr, :]

                # Comprehensive validity checks
                # 1. Check for ignore values
                has_ignore = np.any(patch_hr == ignore_value) or np.any(patch_lr == ignore_value)
                has_nan = np.any(~np.isfinite(patch_hr)) or np.any(~np.isfinite(patch_lr))

                # 2. Check for extreme values (contamination from Gaussian blur of -9999)
                # Reasonable radiance range: 0 to 10000 (conservative upper bound)
                reasonable_range_hr = np.all(patch_hr >= -1) and np.all(patch_hr < 10000)
                reasonable_range_lr = np.all(patch_lr >= -1) and np.all(patch_lr < 10000)

                # 3. Check for excessive negative pixels (>1% suggests blur contamination)
                neg_frac_hr = (patch_hr < 0).mean()
                neg_frac_lr = (patch_lr < 0).mean()

                if (not has_ignore and not has_nan and
                    reasonable_range_hr and reasonable_range_lr and
                    neg_frac_hr < 0.01 and neg_frac_lr < 0.01):
                    # Calculate LR position for reference
                    i_lr = int((row_start + i) / self.downsampling_factor)
                    j_lr = int((col_start + j) / self.downsampling_factor)

                    patches.append({
                        "aviris": patch_hr.astype(np.float32),
                        "landsat": patch_lr.astype(np.float32),
                        "position_hr": (row_start + i, col_start + j),  # Global position
                        "position_lr": (i_lr, j_lr),
                        "valid_fraction": float(valid_fraction)
                    })

        print(f"Extracted {len(patches)} valid patches from {total_attempts} attempts ({100*len(patches)/max(total_attempts,1):.1f}% success rate)")
        return patches

    def process_aviris_file(
        self,
        aviris_path,
        wavelengths=None,
        patch_size=256,
        stride=None,
        output_dir=None
    ):
        """
        Process a single AVIRIS file to generate synthetic Landsat pairs.

        Parameters:
        -----------
        aviris_path : str or Path
            Path to AVIRIS data file
        wavelengths : np.ndarray or None
            Wavelength array. If None, use default AVIRIS wavelengths.
        patch_size : int
            Patch size for extraction (at AVIRIS resolution)
        stride : int or None
            Stride for patch extraction
        output_dir : str, Path, or None
            Directory to save patches. If None, return patches without saving.

        Returns:
        --------
        patches : list of dict
            Extracted patch pairs
        """
        aviris_path = Path(aviris_path)
        print(f"\nProcessing AVIRIS file: {aviris_path.name}")

        # Load AVIRIS data
        aviris_cube, aviris_wavelengths = self.load_aviris(aviris_path)

        # Use provided wavelengths or loaded wavelengths
        if wavelengths is None:
            if aviris_wavelengths is not None:
                wavelengths = aviris_wavelengths
            else:
                # Fall back to default wavelengths
                _, _, n_bands = aviris_cube.shape
                wavelengths = load_aviris_wavelengths(n_bands=n_bands)

        # Generate synthetic Landsat
        synthetic_landsat = self.generate_synthetic_landsat(aviris_cube, wavelengths)

        # Extract patches
        patches = self.extract_patches(
            aviris_cube,
            synthetic_landsat,
            patch_size_hr=patch_size,
            stride=stride
        )

        # Save if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = aviris_path.stem
            output_file = output_dir / f"{base_name}_patches.h5"

            self.save_patches(patches, output_file)
            print(f"Saved patches to: {output_file}")

        return patches

    def load_aviris(self, filepath):
        """
        Load AVIRIS data from file.

        Supports common formats: ENVI (AVIRIS-NG), .h5, .hdf, .npy

        Parameters:
        -----------
        filepath : str or Path
            Path to AVIRIS data file

        Returns:
        --------
        cube : np.ndarray
            AVIRIS radiance cube (shape: [height, width, n_bands])
        wavelengths : np.ndarray or None
            Wavelength array if available
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        wavelengths = None

        # Check if it's an ENVI format file (AVIRIS-NG _img files)
        if '_img' in filepath.name or ext == '':
            # ENVI format (AVIRIS-NG)
            cube, wavelengths, metadata = load_aviris_ng(
                filepath,
                exclude_water_bands=True
            )

        elif ext in ['.h5', '.hdf', '.hdf5']:
            with h5py.File(filepath, 'r') as f:
                # Try common dataset names
                for key in ['radiance', 'data', 'cube', 'aviris']:
                    if key in f:
                        cube = f[key][:]
                        break
                else:
                    # Use first dataset
                    key = list(f.keys())[0]
                    cube = f[key][:]
                    warnings.warn(f"Using dataset '{key}' from HDF5 file")

        elif ext == '.npy':
            cube = np.load(filepath)

        else:
            raise NotImplementedError(f"File format {ext} not supported yet")

        # Ensure shape is [height, width, bands]
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

        print(f"Loaded AVIRIS cube: {cube.shape}")
        return cube, wavelengths

    def save_patches(self, patches, output_file):
        """
        Save patch pairs to HDF5 file.

        Parameters:
        -----------
        patches : list of dict
            Patch pairs from extract_patches()
        output_file : str or Path
            Output HDF5 file path
        """
        output_file = Path(output_file)

        with h5py.File(output_file, 'w') as f:
            # Create datasets
            n_patches = len(patches)

            # Get shapes from first patch
            aviris_shape = patches[0]['aviris'].shape
            landsat_shape = patches[0]['landsat'].shape

            # Create datasets
            aviris_ds = f.create_dataset(
                'aviris',
                shape=(n_patches,) + aviris_shape,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )

            landsat_ds = f.create_dataset(
                'landsat',
                shape=(n_patches,) + landsat_shape,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )

            # Write patches
            for i, patch in enumerate(patches):
                aviris_ds[i] = patch['aviris']
                landsat_ds[i] = patch['landsat']

            # Store metadata
            f.attrs['n_patches'] = n_patches
            f.attrs['aviris_shape'] = aviris_shape
            f.attrs['landsat_shape'] = landsat_shape
            f.attrs['downsampling_factor'] = self.downsampling_factor

        print(f"Saved {n_patches} patches to {output_file}")

    def create_thumbnails(self, patches, thumbnail_dir, wavelengths=None):
        """
        Create RGB thumbnails for each patch pair.

        Parameters:
        -----------
        patches : list of dict
            Patch pairs
        thumbnail_dir : Path
            Directory to save thumbnails
        wavelengths : np.ndarray or None
            Wavelength array for AVIRIS (for RGB band selection)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        thumbnail_dir = Path(thumbnail_dir)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Creating {len(patches)} thumbnails in {thumbnail_dir.name}/...")

        for i, patch in enumerate(patches):
            aviris_patch = patch['aviris']
            landsat_patch = patch['landsat']

            # Create side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            # AVIRIS RGB
            if wavelengths is not None and len(wavelengths) == aviris_patch.shape[2]:
                r_idx = np.abs(wavelengths - 650).argmin()
                g_idx = np.abs(wavelengths - 550).argmin()
                b_idx = np.abs(wavelengths - 450).argmin()
            else:
                # Fallback: assume bands near R, G, B positions
                n_bands = aviris_patch.shape[2]
                r_idx = int(n_bands * 0.6)  # ~60% through spectrum
                g_idx = int(n_bands * 0.4)  # ~40% through spectrum
                b_idx = int(n_bands * 0.2)  # ~20% through spectrum

            aviris_rgb = np.stack([
                aviris_patch[:, :, r_idx],
                aviris_patch[:, :, g_idx],
                aviris_patch[:, :, b_idx]
            ], axis=2)

            # Robust percentile scaling
            aviris_rgb_scaled = self._scale_rgb_for_thumbnail(aviris_rgb)

            ax1.imshow(aviris_rgb_scaled)
            ax1.set_title(f"AVIRIS {aviris_patch.shape[0]}x{aviris_patch.shape[1]}")
            ax1.axis('off')

            # Landsat RGB (bands 4, 3, 2 for natural color)
            if landsat_patch.shape[2] >= 4:
                landsat_rgb = np.stack([
                    landsat_patch[:, :, 3],  # Red
                    landsat_patch[:, :, 2],  # Green
                    landsat_patch[:, :, 1]   # Blue
                ], axis=2)

                landsat_rgb_scaled = self._scale_rgb_for_thumbnail(landsat_rgb)

                ax2.imshow(landsat_rgb_scaled)
                ax2.set_title(f"Landsat {landsat_patch.shape[0]}x{landsat_patch.shape[1]}")
                ax2.axis('off')

            plt.tight_layout()

            # Save with metadata in filename
            pos_hr = patch.get('position_hr', (0, 0))
            valid_frac = patch.get('valid_fraction', 1.0)
            filename = f"patch_{i:04d}_pos{pos_hr[0]}_{pos_hr[1]}_valid{valid_frac:.2f}.png"

            plt.savefig(thumbnail_dir / filename, dpi=100, bbox_inches='tight')
            plt.close(fig)

        print(f"  Saved {len(patches)} thumbnails")

    def _scale_rgb_for_thumbnail(self, rgb, percentile_low=2, percentile_high=98):
        """
        Scale RGB to 0-1 range using percentile clipping.

        Parameters:
        -----------
        rgb : np.ndarray
            RGB image (height, width, 3)
        percentile_low : float
            Lower percentile for clipping
        percentile_high : float
            Upper percentile for clipping

        Returns:
        --------
        scaled : np.ndarray
            Scaled RGB in [0, 1]
        """
        # Create validity mask
        valid_mask = (
            (rgb > -10) &
            (rgb < 1e6) &
            np.isfinite(rgb)
        )

        valid_data = rgb[valid_mask]

        if len(valid_data) == 0:
            return np.zeros_like(rgb)

        # Percentile scaling
        p_low = np.percentile(valid_data, percentile_low)
        p_high = np.percentile(valid_data, percentile_high)

        scaled = (rgb - p_low) / (p_high - p_low + 1e-8)
        scaled = np.clip(scaled, 0, 1)

        # Set invalid pixels to black
        scaled[~valid_mask] = 0

        return scaled


# Utility functions

def load_patch_dataset(filepath):
    """
    Load saved patch dataset from HDF5.

    Parameters:
    -----------
    filepath : str or Path
        Path to HDF5 file created by save_patches()

    Returns:
    --------
    aviris_patches : np.ndarray
        AVIRIS patches (shape: [n_patches, height, width, bands])
    landsat_patches : np.ndarray
        Landsat patches (shape: [n_patches, height, width, bands])
    metadata : dict
        Dataset metadata
    """
    with h5py.File(filepath, 'r') as f:
        aviris_patches = f['aviris'][:]
        landsat_patches = f['landsat'][:]

        metadata = dict(f.attrs)

    return aviris_patches, landsat_patches, metadata


# Example usage and testing
if __name__ == "__main__":
    print("Synthetic Data Generator Test")
    print("=" * 50)

    # Create generator
    generator = SyntheticDataGenerator(
        spatial_downsampling_factor=7.5,
        psf_sigma=2.0,
        landsat_sensor="LC08"
    )

    # Create synthetic AVIRIS data for testing
    print("\nCreating synthetic AVIRIS test data...")
    height_hr = 512
    width_hr = 512
    n_aviris_bands = 224

    # Generate random radiance-like data
    np.random.seed(42)
    aviris_test = np.random.rand(height_hr, width_hr, n_aviris_bands).astype(np.float32) * 1000

    # Add some spectral structure (simulate vegetation signature)
    for i in range(height_hr):
        for j in range(width_hr):
            # Simple vegetation-like spectrum
            spectrum = np.ones(n_aviris_bands) * 500
            spectrum[100:150] *= 0.5  # Red absorption
            spectrum[150:] *= 1.5     # NIR reflectance increase
            aviris_test[i, j, :] = spectrum + np.random.randn(n_aviris_bands) * 50

    wavelengths = load_aviris_wavelengths(n_bands=n_aviris_bands)

    # Generate synthetic Landsat
    print("\nGenerating synthetic Landsat...")
    synthetic_landsat = generator.generate_synthetic_landsat(
        aviris_test,
        wavelengths
    )

    print(f"\nResults:")
    print(f"  AVIRIS shape: {aviris_test.shape}")
    print(f"  Synthetic Landsat shape: {synthetic_landsat.shape}")
    print(f"  Expected downsampling: {height_hr / synthetic_landsat.shape[0]:.2f}x")

    # Extract patches
    print("\nExtracting patches...")
    patches = generator.extract_patches(
        aviris_test,
        synthetic_landsat,
        patch_size_hr=256,
        stride=128
    )

    print(f"\nPatch statistics:")
    print(f"  Number of patches: {len(patches)}")
    if len(patches) > 0:
        print(f"  AVIRIS patch shape: {patches[0]['aviris'].shape}")
        print(f"  Landsat patch shape: {patches[0]['landsat'].shape}")

    print("\n" + "=" * 50)
    print("Test complete!")
