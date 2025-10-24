"""
Landsat Spectral Response Functions (SRF) Loader

Loads and applies Landsat spectral response functions to simulate Landsat
bands from hyperspectral data (e.g., AVIRIS).

References:
- Barsi et al. (2014) "The spectral response of the Landsat-8 Operational Land Imager"
- USGS Spectral Characteristics Viewer: https://www.usgs.gov/landsat-missions/spectral-characteristics-viewer
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings


class LandsatSRF:
    """
    Landsat Spectral Response Function handler.

    Supports Landsat-8 OLI and can be extended for other sensors.
    """

    # Landsat-8 OLI band specifications (approximate center wavelength and bandwidth)
    # Based on published specifications: https://landsat.gsfc.nasa.gov/satellites/landsat-8/spacecraft-instruments/operational-land-imager/
    LANDSAT8_BANDS = {
        1: {"name": "Coastal Aerosol", "center": 443, "fwhm": 16, "range": [435, 451]},
        2: {"name": "Blue", "center": 482, "fwhm": 60, "range": [452, 512]},
        3: {"name": "Green", "center": 562, "fwhm": 57, "range": [533, 590]},
        4: {"name": "Red", "center": 655, "fwhm": 37, "range": [636, 673]},
        5: {"name": "NIR", "center": 865, "fwhm": 28, "range": [851, 879]},
        6: {"name": "SWIR-1", "center": 1609, "fwhm": 85, "range": [1566, 1651]},
        7: {"name": "SWIR-2", "center": 2201, "fwhm": 187, "range": [2107, 2294]},
    }

    def __init__(self, sensor="LC08", use_gaussian_approximation=True):
        """
        Initialize Landsat SRF loader.

        Parameters:
        -----------
        sensor : str
            Landsat sensor code ('LC08' for Landsat-8 OLI)
        use_gaussian_approximation : bool
            If True, use Gaussian approximation for SRFs.
            If False, attempt to load actual SRF data files.
        """
        self.sensor = sensor
        self.use_gaussian = use_gaussian_approximation

        if sensor == "LC08":
            self.bands = self.LANDSAT8_BANDS
        else:
            raise NotImplementedError(f"Sensor {sensor} not yet implemented")

        self.srf_cache = {}

    def get_srf(self, band_number, wavelengths):
        """
        Get spectral response function for a specific band.

        Parameters:
        -----------
        band_number : int
            Landsat band number (1-7 for OLI)
        wavelengths : np.ndarray
            Wavelength array (in nm) for which to compute SRF

        Returns:
        --------
        srf : np.ndarray
            Spectral response values (normalized to integrate to 1)
        """
        if band_number not in self.bands:
            raise ValueError(f"Band {band_number} not available for {self.sensor}")

        band_info = self.bands[band_number]

        if self.use_gaussian:
            srf = self._gaussian_srf(
                wavelengths,
                center=band_info["center"],
                fwhm=band_info["fwhm"]
            )
        else:
            # TODO: Load actual SRF from file
            warnings.warn("Actual SRF loading not implemented, using Gaussian approximation")
            srf = self._gaussian_srf(
                wavelengths,
                center=band_info["center"],
                fwhm=band_info["fwhm"]
            )

        return srf

    def _gaussian_srf(self, wavelengths, center, fwhm):
        """
        Compute Gaussian approximation of spectral response function.

        Parameters:
        -----------
        wavelengths : np.ndarray
            Wavelength array (in nm)
        center : float
            Center wavelength (in nm)
        fwhm : float
            Full-width at half-maximum (in nm)

        Returns:
        --------
        srf : np.ndarray
            Normalized spectral response function
        """
        # Convert FWHM to standard deviation
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Compute Gaussian
        srf = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

        # Normalize to integrate to 1 (for proper radiance integration)
        # Using trapezoidal rule for integration
        integral = np.trapz(srf, wavelengths)
        if integral > 0:
            srf = srf / integral

        return srf

    def integrate_spectrum(self, spectrum, wavelengths, band_number):
        """
        Integrate a hyperspectral spectrum to a single Landsat band.

        This simulates what the Landsat sensor would measure from the
        hyperspectral signal.

        Parameters:
        -----------
        spectrum : np.ndarray
            Radiance spectrum (shape: [n_wavelengths])
        wavelengths : np.ndarray
            Wavelength array corresponding to spectrum (in nm)
        band_number : int
            Target Landsat band number

        Returns:
        --------
        integrated_value : float
            Integrated radiance for the Landsat band
        """
        # Get SRF for this band
        srf = self.get_srf(band_number, wavelengths)

        # Integrate: L_band = ∫ L(λ) * SRF(λ) dλ
        # Since SRF is normalized, this gives weighted average radiance
        integrated_value = np.trapz(spectrum * srf, wavelengths)

        return integrated_value

    def integrate_cube(self, hyperspectral_cube, wavelengths, band_numbers=None):
        """
        Integrate entire hyperspectral cube to Landsat bands.

        Parameters:
        -----------
        hyperspectral_cube : np.ndarray
            Hyperspectral data cube (shape: [height, width, n_bands])
        wavelengths : np.ndarray
            Wavelength array for hyperspectral bands (in nm)
        band_numbers : list or None
            List of Landsat bands to compute. If None, compute all bands.

        Returns:
        --------
        landsat_cube : np.ndarray
            Simulated Landsat imagery (shape: [height, width, n_landsat_bands])
        """
        if band_numbers is None:
            band_numbers = sorted(self.bands.keys())

        height, width, n_bands = hyperspectral_cube.shape
        n_landsat_bands = len(band_numbers)

        # Initialize output
        landsat_cube = np.zeros((height, width, n_landsat_bands), dtype=np.float32)

        # Integrate each pixel's spectrum
        for i, band_num in enumerate(band_numbers):
            srf = self.get_srf(band_num, wavelengths)

            # Vectorized integration over spatial dimensions
            # spectrum shape: [height, width, n_bands]
            # srf shape: [n_bands]
            # Result: [height, width]
            landsat_cube[:, :, i] = np.trapz(
                hyperspectral_cube * srf[np.newaxis, np.newaxis, :],
                wavelengths,
                axis=2
            )

        return landsat_cube

    def get_band_info(self, band_number):
        """Get information about a specific Landsat band."""
        if band_number not in self.bands:
            raise ValueError(f"Band {band_number} not available")
        return self.bands[band_number]

    def get_all_bands_info(self):
        """Get information about all available bands."""
        return self.bands

    def plot_srfs(self, wavelengths=None):
        """
        Plot spectral response functions for all bands.

        Parameters:
        -----------
        wavelengths : np.ndarray or None
            Wavelength array for plotting. If None, use 350-2500 nm range.
        """
        import matplotlib.pyplot as plt

        if wavelengths is None:
            wavelengths = np.linspace(350, 2500, 2151)

        plt.figure(figsize=(12, 6))

        for band_num in sorted(self.bands.keys()):
            srf = self.get_srf(band_num, wavelengths)
            band_info = self.bands[band_num]
            plt.plot(wavelengths, srf, label=f"B{band_num}: {band_info['name']}")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Spectral Response (normalized)")
        plt.title(f"{self.sensor} Spectral Response Functions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()


def load_aviris_wavelengths(n_bands=224, spectral_range=(380, 2510)):
    """
    Generate AVIRIS wavelength array.

    AVIRIS-Classic/NG has ~224 bands covering 380-2510 nm with ~10nm spacing.

    Parameters:
    -----------
    n_bands : int
        Number of AVIRIS bands (default: 224)
    spectral_range : tuple
        (min_wavelength, max_wavelength) in nm

    Returns:
    --------
    wavelengths : np.ndarray
        Wavelength centers for each band (in nm)
    """
    return np.linspace(spectral_range[0], spectral_range[1], n_bands)


# Example usage and testing
if __name__ == "__main__":
    print("Landsat SRF Module Test")
    print("=" * 50)

    # Initialize SRF loader
    srf = LandsatSRF(sensor="LC08")

    # Display band information
    print("\nLandsat-8 OLI Band Information:")
    for band_num, info in srf.get_all_bands_info().items():
        print(f"  Band {band_num}: {info['name']:20s} "
              f"Center: {info['center']:4d} nm, "
              f"FWHM: {info['fwhm']:3d} nm")

    # Generate AVIRIS wavelengths
    aviris_wl = load_aviris_wavelengths()
    print(f"\nAVIRIS wavelengths: {len(aviris_wl)} bands")
    print(f"  Range: {aviris_wl[0]:.1f} - {aviris_wl[-1]:.1f} nm")

    # Test integration
    print("\nTesting spectral integration:")
    test_spectrum = np.ones_like(aviris_wl)  # Flat spectrum
    for band_num in [1, 2, 3, 4, 5, 6, 7]:
        integrated = srf.integrate_spectrum(test_spectrum, aviris_wl, band_num)
        print(f"  Band {band_num}: {integrated:.6f}")

    # Plot SRFs
    print("\nGenerating SRF plot...")
    fig = srf.plot_srfs(aviris_wl)
    print("  Plot created (not displayed in CLI mode)")

    print("\n" + "=" * 50)
    print("Test complete!")
