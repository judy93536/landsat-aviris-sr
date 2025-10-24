"""
Data Augmentation for Synthetic Landsat-AVIRIS Pairs

Adds realistic noise, artifacts, and atmospheric variations to synthetic data
to close the domain gap between synthetic and real imagery.

Augmentation phases:
1. Phase 1 (Basic): Spatial transforms (rotation, flip)
2. Phase 2 (Noise): Band-correlated noise, detector artifacts
3. Phase 3 (Atmospheric): MODTRAN-based atmospheric variations

References:
- Landsat noise characteristics: Markham et al. (2014)
- Sensor artifacts: Morfitt et al. (2015)
"""

import numpy as np
from scipy.ndimage import rotate, zoom
from scipy.signal import butter, filtfilt
import warnings


class NoiseModel:
    """
    Realistic noise model for multispectral sensors.

    Implements signal-dependent noise with band-specific SNR.
    """

    def __init__(self, snr_range=(30, 100), band_correlation=0.3):
        """
        Initialize noise model.

        Parameters:
        -----------
        snr_range : tuple
            (min_snr, max_snr) in dB for random SNR per band
        band_correlation : float
            Correlation coefficient for band-correlated noise (0-1)
        """
        self.snr_range = snr_range
        self.band_correlation = band_correlation

    def add_noise(self, image, snr_db=None):
        """
        Add realistic band-correlated noise to image.

        Parameters:
        -----------
        image : np.ndarray
            Clean image (shape: [height, width, bands])
        snr_db : float, np.ndarray, or None
            Signal-to-noise ratio in dB. If None, random per band.
            Can be scalar (same for all bands) or array (per band).

        Returns:
        --------
        noisy_image : np.ndarray
            Image with added noise
        """
        height, width, n_bands = image.shape

        # Generate SNR values
        if snr_db is None:
            snr_db = np.random.uniform(
                self.snr_range[0],
                self.snr_range[1],
                size=n_bands
            )
        elif np.isscalar(snr_db):
            snr_db = np.full(n_bands, snr_db)

        # Convert SNR from dB to linear
        snr_linear = 10 ** (snr_db / 10.0)

        # Initialize output
        noisy_image = np.zeros_like(image)

        # Generate correlated noise
        # Noise has two components: correlated and uncorrelated
        correlated_noise = np.random.randn(height, width)

        for b in range(n_bands):
            signal = image[:, :, b]
            signal_power = np.mean(signal ** 2)

            # Calculate noise power from SNR
            noise_power = signal_power / snr_linear[b]
            noise_std = np.sqrt(noise_power)

            # Generate noise with correlation
            uncorrelated_noise = np.random.randn(height, width)
            combined_noise = (
                self.band_correlation * correlated_noise +
                np.sqrt(1 - self.band_correlation ** 2) * uncorrelated_noise
            )
            combined_noise *= noise_std

            # Add noise
            noisy_image[:, :, b] = signal + combined_noise

        return noisy_image.astype(image.dtype)


class DetectorArtifacts:
    """
    Simulate detector artifacts: striping, banding, dead pixels.
    """

    def __init__(self):
        pass

    def add_striping(self, image, stripe_intensity=0.02, axis=0):
        """
        Add detector striping artifacts (common in pushbroom sensors).

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])
        stripe_intensity : float
            Relative intensity of striping (0-1)
        axis : int
            Axis along which to add stripes (0=horizontal, 1=vertical)

        Returns:
        --------
        striped_image : np.ndarray
            Image with striping artifacts
        """
        height, width, n_bands = image.shape

        striped_image = image.copy()

        for b in range(n_bands):
            # Generate random stripe pattern
            if axis == 0:
                # Horizontal stripes (vary across rows)
                stripe_pattern = np.random.randn(height, 1) * stripe_intensity
                stripe_pattern = np.repeat(stripe_pattern, width, axis=1)
            else:
                # Vertical stripes (vary across columns)
                stripe_pattern = np.random.randn(1, width) * stripe_intensity
                stripe_pattern = np.repeat(stripe_pattern, height, axis=0)

            # Apply multiplicative striping
            mean_value = np.mean(striped_image[:, :, b])
            striped_image[:, :, b] *= (1 + stripe_pattern)

        return striped_image

    def add_banding(self, image, band_width=8, band_intensity=0.01):
        """
        Add detector banding artifacts.

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])
        band_width : int
            Width of bands in pixels
        band_intensity : float
            Relative intensity of banding

        Returns:
        --------
        banded_image : np.ndarray
            Image with banding artifacts
        """
        height, width, n_bands = image.shape
        banded_image = image.copy()

        for b in range(n_bands):
            # Create banding pattern
            n_bands_pattern = height // band_width
            band_values = np.random.randn(n_bands_pattern) * band_intensity

            band_pattern = np.repeat(band_values, band_width)[:height]
            band_pattern = band_pattern[:, np.newaxis].repeat(width, axis=1)

            # Apply additive banding
            banded_image[:, :, b] += band_pattern * np.mean(image[:, :, b])

        return banded_image

    def add_dead_pixels(self, image, dead_pixel_fraction=0.0001):
        """
        Add random dead pixels.

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])
        dead_pixel_fraction : float
            Fraction of pixels to mark as dead (0-1)

        Returns:
        --------
        image_with_dead : np.ndarray
            Image with dead pixels
        """
        height, width, n_bands = image.shape
        n_dead = int(height * width * dead_pixel_fraction)

        image_with_dead = image.copy()

        # Randomly select dead pixel locations
        dead_rows = np.random.randint(0, height, size=n_dead)
        dead_cols = np.random.randint(0, width, size=n_dead)

        # Set dead pixels to zero or interpolate from neighbors
        for i, j in zip(dead_rows, dead_cols):
            image_with_dead[i, j, :] = 0

        return image_with_dead


class GeometricAugmentation:
    """
    Geometric transformations for data augmentation.
    """

    @staticmethod
    def random_rotation(image, angle_range=(-180, 180)):
        """
        Apply random rotation.

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])
        angle_range : tuple
            (min_angle, max_angle) in degrees

        Returns:
        --------
        rotated : np.ndarray
            Rotated image
        """
        angle = np.random.uniform(angle_range[0], angle_range[1])

        # Rotate each band
        rotated = np.stack([
            rotate(image[:, :, b], angle, reshape=False, mode='reflect')
            for b in range(image.shape[2])
        ], axis=2)

        return rotated.astype(image.dtype)

    @staticmethod
    def random_flip(image):
        """
        Apply random horizontal/vertical flip.

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])

        Returns:
        --------
        flipped : np.ndarray
            Flipped image
        """
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0)  # Vertical flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)  # Horizontal flip

        return image.copy()

    @staticmethod
    def random_crop(image, crop_size):
        """
        Extract random crop from image.

        Parameters:
        -----------
        image : np.ndarray
            Image (shape: [height, width, bands])
        crop_size : int or tuple
            Size of crop (square if int, (height, width) if tuple)

        Returns:
        --------
        cropped : np.ndarray
            Cropped image
        """
        if isinstance(crop_size, int):
            crop_h, crop_w = crop_size, crop_size
        else:
            crop_h, crop_w = crop_size

        height, width, _ = image.shape

        if height < crop_h or width < crop_w:
            raise ValueError(f"Image {image.shape} too small for crop {(crop_h, crop_w)}")

        i = np.random.randint(0, height - crop_h + 1)
        j = np.random.randint(0, width - crop_w + 1)

        return image[i:i+crop_h, j:j+crop_w, :]


class AtmosphericAugmentation:
    """
    Atmospheric effects for closing synthetic-to-real domain gap.

    Phase 3 augmentation: Use MODTRAN or simplified atmospheric models
    to add realistic atmospheric variations.
    """

    def __init__(self):
        """
        Initialize atmospheric augmentation.

        Note: Full MODTRAN integration requires external tools.
        This provides simplified atmospheric effects.
        """
        warnings.warn(
            "AtmosphericAugmentation: Full MODTRAN integration not yet implemented. "
            "Using simplified atmospheric model."
        )

    def add_atmospheric_haze(
        self,
        image,
        wavelengths,
        visibility_km=20.0,
        water_vapor_cm=2.0
    ):
        """
        Add simplified atmospheric haze effect.

        Parameters:
        -----------
        image : np.ndarray
            Radiance image (shape: [height, width, bands])
        wavelengths : np.ndarray
            Wavelength array (in nm)
        visibility_km : float
            Atmospheric visibility in kilometers
        water_vapor_cm : float
            Column water vapor in cm

        Returns:
        --------
        hazy_image : np.ndarray
            Image with atmospheric haze
        """
        # Simplified atmospheric scattering model
        # Real implementation would use MODTRAN LUTs

        # Rayleigh scattering: λ^-4 dependence
        rayleigh_optical_depth = 0.008735 * (wavelengths / 1000) ** (-4.08)

        # Aerosol scattering: λ^-α dependence (α ~ 1.3 for typical aerosols)
        aerosol_optical_depth = (3.912 / visibility_km) * (wavelengths / 550) ** (-1.3)

        # Total optical depth
        total_tau = rayleigh_optical_depth + aerosol_optical_depth

        # Atmospheric transmission
        transmission = np.exp(-total_tau)

        # Path radiance (simplified)
        path_radiance = 0.1 * (1 - transmission) * np.mean(image, axis=(0, 1))

        # Apply atmospheric effect
        hazy_image = image * transmission[np.newaxis, np.newaxis, :] + path_radiance[np.newaxis, np.newaxis, :]

        return hazy_image.astype(image.dtype)


class AugmentationPipeline:
    """
    Complete augmentation pipeline with progressive phases.
    """

    def __init__(
        self,
        geometric=True,
        noise=True,
        artifacts=True,
        atmospheric=False
    ):
        """
        Initialize augmentation pipeline.

        Parameters:
        -----------
        geometric : bool
            Enable geometric augmentations (rotation, flip)
        noise : bool
            Enable noise augmentation
        artifacts : bool
            Enable detector artifact simulation
        atmospheric : bool
            Enable atmospheric augmentation (Phase 3)
        """
        self.geometric = geometric
        self.noise_enabled = noise
        self.artifacts_enabled = artifacts
        self.atmospheric_enabled = atmospheric

        # Initialize components
        self.noise_model = NoiseModel()
        self.detector = DetectorArtifacts()
        self.geom = GeometricAugmentation()
        self.atmos = AtmosphericAugmentation()

    def augment_pair(
        self,
        aviris_patch,
        landsat_patch,
        wavelengths=None
    ):
        """
        Apply augmentation to a paired patch.

        Parameters:
        -----------
        aviris_patch : np.ndarray
            AVIRIS patch (high-res)
        landsat_patch : np.ndarray
            Landsat patch (low-res)
        wavelengths : np.ndarray or None
            Wavelengths for atmospheric augmentation

        Returns:
        --------
        augmented_aviris : np.ndarray
        augmented_landsat : np.ndarray
        """
        aug_aviris = aviris_patch.copy()
        aug_landsat = landsat_patch.copy()

        # Phase 1: Geometric (applied consistently to both)
        if self.geometric:
            # Random rotation (same for both)
            angle = np.random.uniform(-180, 180)
            aug_aviris = rotate(aug_aviris, angle, reshape=False, mode='reflect')
            aug_landsat = rotate(aug_landsat, angle, reshape=False, mode='reflect')

            # Random flip (same for both)
            if np.random.rand() > 0.5:
                aug_aviris = np.flip(aug_aviris, axis=0)
                aug_landsat = np.flip(aug_landsat, axis=0)
            if np.random.rand() > 0.5:
                aug_aviris = np.flip(aug_aviris, axis=1)
                aug_landsat = np.flip(aug_landsat, axis=1)

        # Phase 2: Noise and artifacts
        if self.noise_enabled:
            aug_landsat = self.noise_model.add_noise(aug_landsat)

        if self.artifacts_enabled:
            if np.random.rand() > 0.5:
                aug_landsat = self.detector.add_striping(aug_landsat, stripe_intensity=0.01)
            if np.random.rand() > 0.7:
                aug_landsat = self.detector.add_banding(aug_landsat, band_intensity=0.005)

        # Phase 3: Atmospheric (requires wavelengths)
        if self.atmospheric_enabled and wavelengths is not None:
            visibility = np.random.uniform(5, 40)  # km
            water_vapor = np.random.uniform(0.5, 4.0)  # cm
            aug_aviris = self.atmos.add_atmospheric_haze(
                aug_aviris,
                wavelengths,
                visibility_km=visibility,
                water_vapor_cm=water_vapor
            )

        return aug_aviris.astype(np.float32), aug_landsat.astype(np.float32)


# Example usage and testing
if __name__ == "__main__":
    print("Data Augmentation Module Test")
    print("=" * 50)

    # Create test image
    height, width, n_bands = 256, 256, 7
    test_image = np.random.rand(height, width, n_bands).astype(np.float32) * 1000

    print(f"\nTest image shape: {test_image.shape}")
    print(f"Test image mean: {test_image.mean():.2f}")

    # Test noise model
    print("\n1. Testing noise model...")
    noise_model = NoiseModel(snr_range=(40, 60))
    noisy = noise_model.add_noise(test_image, snr_db=50)
    print(f"  Noisy image mean: {noisy.mean():.2f}")
    print(f"  Noise std: {(noisy - test_image).std():.2f}")

    # Test detector artifacts
    print("\n2. Testing detector artifacts...")
    detector = DetectorArtifacts()
    striped = detector.add_striping(test_image, stripe_intensity=0.02)
    print(f"  Striped image mean: {striped.mean():.2f}")

    banded = detector.add_banding(test_image, band_intensity=0.01)
    print(f"  Banded image mean: {banded.mean():.2f}")

    # Test geometric augmentation
    print("\n3. Testing geometric augmentation...")
    geom = GeometricAugmentation()
    rotated = geom.random_rotation(test_image, angle_range=(-45, 45))
    print(f"  Rotated image shape: {rotated.shape}")

    flipped = geom.random_flip(test_image)
    print(f"  Flipped image shape: {flipped.shape}")

    # Test augmentation pipeline
    print("\n4. Testing augmentation pipeline...")
    pipeline = AugmentationPipeline(
        geometric=True,
        noise=True,
        artifacts=True,
        atmospheric=False
    )

    # Create mock paired data
    aviris_patch = np.random.rand(256, 256, 224).astype(np.float32) * 1000
    landsat_patch = np.random.rand(34, 34, 7).astype(np.float32) * 1000

    aug_aviris, aug_landsat = pipeline.augment_pair(aviris_patch, landsat_patch)
    print(f"  Augmented AVIRIS shape: {aug_aviris.shape}")
    print(f"  Augmented Landsat shape: {aug_landsat.shape}")

    print("\n" + "=" * 50)
    print("Test complete!")
