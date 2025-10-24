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

            # Downsample using scipy.ndimage.zoom
            zoom_factor = 1.0 / self.downsampling_factor
            degraded[:, :, b] = zoom(
                blurred,
                zoom_factor,
                order=1,  # Bilinear interpolation
                mode='reflect'
            )

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

        Pipeline:
        1. Spatial degradation (blur + downsample)
        2. Spectral integration (AVIRIS bands → Landsat bands)

        Parameters:
        -----------
        aviris_cube : np.ndarray
            AVIRIS radiance cube (shape: [height, width, n_bands])
        aviris_wavelengths : np.ndarray
            Wavelength array for AVIRIS bands (in nm)
        return_intermediate : bool
            If True, return intermediate spatially-degraded cube

        Returns:
        --------
        synthetic_landsat : np.ndarray
            Simulated Landsat imagery (shape: [height_lr, width_lr, n_landsat_bands])
        spatially_degraded : np.ndarray (optional)
            Spatially degraded AVIRIS cube before spectral integration
        """
        print(f"Generating synthetic Landsat from AVIRIS cube: {aviris_cube.shape}")

        # Step 1: Spatial degradation
        print("  Step 1: Applying spatial degradation...")
        spatially_degraded = self.spatial_degrade(aviris_cube)
        print(f"    Output shape: {spatially_degraded.shape}")

        # Step 2: Spectral integration
        print("  Step 2: Applying spectral integration...")
        synthetic_landsat = self.spectral_integrate(
            spatially_degraded,
            aviris_wavelengths
        )
        print(f"    Output shape: {synthetic_landsat.shape}")

        if return_intermediate:
            return synthetic_landsat, spatially_degraded
        else:
            return synthetic_landsat

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

        height_hr, width_hr, _ = aviris_cube.shape
        height_lr, width_lr, _ = synthetic_landsat.shape

        # Calculate corresponding patch size in low-res
        patch_size_lr = int(patch_size_hr / self.downsampling_factor)

        # Create validity mask using a representative band (middle of NIR region, ~850nm)
        # Use first band if cube is small, otherwise use band ~100 (around 850nm for AVIRIS)
        mask_band_idx = min(100, aviris_cube.shape[2] // 2)
        mask_band = aviris_cube[:, :, mask_band_idx]

        # Valid pixels: not NaN, not ignore_value, and positive radiance
        validity_mask = (
            np.isfinite(mask_band) &
            (mask_band != ignore_value) &
            (mask_band > 0) &
            (mask_band < 1e6)  # Reasonable upper bound for radiance
        )

        patches = []
        total_attempts = 0

        # Extract patches with sliding window
        for i in range(0, height_hr - patch_size_hr + 1, stride):
            for j in range(0, width_hr - patch_size_hr + 1, stride):
                total_attempts += 1

                # Check validity mask for this patch region
                patch_mask = validity_mask[i:i+patch_size_hr, j:j+patch_size_hr]
                valid_fraction = patch_mask.mean()

                if valid_fraction < min_valid_fraction:
                    continue  # Skip patches with insufficient valid pixels

                # Extract high-res patch
                patch_hr = aviris_cube[i:i+patch_size_hr, j:j+patch_size_hr, :]

                # Calculate corresponding low-res position
                i_lr = int(i / self.downsampling_factor)
                j_lr = int(j / self.downsampling_factor)

                # Extract low-res patch
                patch_lr = synthetic_landsat[
                    i_lr:i_lr+patch_size_lr,
                    j_lr:j_lr+patch_size_lr,
                    :
                ]

                # Additional validity checks
                # Check for ignore values across all bands
                has_ignore_hr = np.any(patch_hr == ignore_value)
                has_ignore_lr = np.any(patch_lr == ignore_value)

                # Check for reasonable radiance values
                valid_radiance_hr = np.all(np.isfinite(patch_hr)) and np.all(patch_hr >= 0) and np.all(patch_hr < 1e6)
                valid_radiance_lr = np.all(np.isfinite(patch_lr)) and np.all(patch_lr >= 0) and np.all(patch_lr < 1e6)

                if not has_ignore_hr and not has_ignore_lr and valid_radiance_hr and valid_radiance_lr:
                    patches.append({
                        "aviris": patch_hr.astype(np.float32),
                        "landsat": patch_lr.astype(np.float32),
                        "position_hr": (i, j),
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
