"""
AVIRIS Classic Reflectance Data Loader

Loads AVIRIS Classic surface reflectance imagery in ENVI format.
Specifically designed for reflectance data with strict quality control.

Key features:
- Filters bad bands using bbl (bad band list)
- Excludes bands with extreme correction factors
- Validates reflectance range (0-1 or scaled)
- Detects pad pixels, dropouts, and spikes
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

# Import base ENVI classes
sys.path.append(str(Path(__file__).parent))
from envi_loader import ENVIHeader, ENVIImage


class AVIRISClassicImage(ENVIImage):
    """
    AVIRIS Classic reflectance image loader with enhanced quality control.
    """

    def __init__(self, img_path):
        """
        Initialize AVIRIS Classic image loader.

        Parameters:
        -----------
        img_path : str or Path
            Path to AVIRIS Classic corrected reflectance image (*_corr_*_img)
        """
        super().__init__(img_path)

        # Get AVIRIS Classic specific metadata
        self.bbl = np.array(self.header.get('bbl', []), dtype=int)
        self.correction_factors = np.array(self.header.get('correction factors', []))
        self.gsd = self._extract_gsd()

        print(f"  GSD: {self.gsd:.2f} m")
        print(f"  Bad band list: {len(self.bbl)} bands")

    def _extract_gsd(self):
        """Extract ground sample distance from header."""
        # Try description field first
        desc = self.header.get('description', '')
        if isinstance(desc, list):
            desc = ' '.join(desc)

        # Look for "pixel size = X.X" in description
        import re
        match = re.search(r'pixel size\s*=\s*([\d.]+)', desc)
        if match:
            return float(match.group(1))

        # Try map info
        map_info = self.header.get('map info', [])
        if isinstance(map_info, list) and len(map_info) > 5:
            return float(map_info[5])

        return None

    def get_good_bands(self,
                       use_bbl=True,
                       filter_correction_factors=True,
                       cf_min=0.5,
                       cf_max=1.5):
        """
        Get indices of good quality bands using multiple criteria.

        Parameters:
        -----------
        use_bbl : bool
            Use bad band list from header (bbl field)
        filter_correction_factors : bool
            Filter bands with extreme correction factors
        cf_min : float
            Minimum acceptable correction factor
        cf_max : float
            Maximum acceptable correction factor

        Returns:
        --------
        good_bands : np.ndarray
            Array of good band indices
        """
        n_bands = self.bands
        good_mask = np.ones(n_bands, dtype=bool)

        # Apply bad band list
        if use_bbl and len(self.bbl) == n_bands:
            good_mask &= (self.bbl == 1)
            n_bbl_bad = np.sum(self.bbl == 0)
            if n_bbl_bad > 0:
                print(f"  Excluding {n_bbl_bad} bands from bbl")

        # Filter extreme correction factors
        if filter_correction_factors and len(self.correction_factors) == n_bands:
            cf_good = ((self.correction_factors >= cf_min) &
                      (self.correction_factors <= cf_max))
            n_cf_bad = np.sum(~cf_good)
            if n_cf_bad > 0:
                print(f"  Excluding {n_cf_bad} bands with extreme correction factors")
            good_mask &= cf_good

        good_indices = np.where(good_mask)[0]
        print(f"  Final: {len(good_indices)}/{n_bands} good bands")

        return good_indices


def validate_patch(patch,
                   min_reflectance=-0.05,
                   max_reflectance=1.2,
                   max_nan_fraction=0.0,
                   max_zero_fraction=0.0,
                   check_spectral_spikes=True,
                   spike_threshold=5.0):
    """
    Validate a reflectance patch for training quality.

    Strict quality control - rejects patches with ANY anomalies.

    Parameters:
    -----------
    patch : np.ndarray
        Reflectance patch (H x W x C)
    min_reflectance : float
        Minimum valid reflectance value
    max_reflectance : float
        Maximum valid reflectance value (slight overshoot allowed)
    max_nan_fraction : float
        Maximum fraction of NaN/inf pixels allowed (0.0 = none)
    max_zero_fraction : float
        Maximum fraction of zero pixels allowed (0.0 = none)
    check_spectral_spikes : bool
        Check for spectral spikes/anomalies
    spike_threshold : float
        Z-score threshold for spike detection

    Returns:
    --------
    is_valid : bool
        True if patch passes all quality checks
    reason : str
        Reason for rejection (if invalid)
    """
    h, w, c = patch.shape
    n_pixels = h * w

    # Check for NaN/inf
    nan_mask = ~np.isfinite(patch)
    nan_fraction = np.sum(nan_mask) / (n_pixels * c)
    if nan_fraction > max_nan_fraction:
        return False, f"NaN/inf pixels: {nan_fraction*100:.2f}%"

    # Check for zeros (pad pixels or dropouts)
    zero_mask = (patch == 0)
    zero_fraction = np.sum(zero_mask) / (n_pixels * c)
    if zero_fraction > max_zero_fraction:
        return False, f"Zero pixels: {zero_fraction*100:.2f}%"

    # Check reflectance range
    valid_mask = np.isfinite(patch)
    if np.any(valid_mask):
        min_val = np.min(patch[valid_mask])
        max_val = np.max(patch[valid_mask])

        if min_val < min_reflectance:
            return False, f"Reflectance too low: {min_val:.4f}"
        if max_val > max_reflectance:
            return False, f"Reflectance too high: {max_val:.4f}"

    # Check for spectral spikes
    if check_spectral_spikes:
        # Calculate spectral derivatives (band-to-band changes)
        spectral_mean = np.mean(patch, axis=(0, 1))  # (C,)
        spectral_diff = np.diff(spectral_mean)  # (C-1,)

        # Robust spike detection using median absolute deviation
        if len(spectral_diff) > 10:
            median_diff = np.median(spectral_diff)
            mad = np.median(np.abs(spectral_diff - median_diff))

            if mad > 1e-6:  # Avoid division by zero
                z_scores = np.abs(spectral_diff - median_diff) / (mad * 1.4826)

                if np.any(z_scores > spike_threshold):
                    n_spikes = np.sum(z_scores > spike_threshold)
                    return False, f"Spectral spikes detected: {n_spikes} bands"

    return True, "Valid"


def extract_patches(img: AVIRISClassicImage,
                   patch_size: int = 256,
                   stride: Optional[int] = None,
                   good_bands: Optional[np.ndarray] = None,
                   validate: bool = True,
                   max_patches: Optional[int] = None,
                   verbose: bool = True) -> Tuple[list, dict]:
    """
    Extract non-overlapping patches from AVIRIS Classic image with validation.

    Parameters:
    -----------
    img : AVIRISClassicImage
        AVIRIS Classic image object
    patch_size : int
        Size of square patches (pixels)
    stride : int or None
        Stride between patches (None = non-overlapping)
    good_bands : np.ndarray or None
        Indices of bands to include (None = all bands)
    validate : bool
        Apply strict quality validation to each patch
    max_patches : int or None
        Maximum number of patches to extract (None = all)
    verbose : bool
        Print progress

    Returns:
    --------
    patches : list of np.ndarray
        List of valid patches (patch_size x patch_size x n_bands)
    metadata : dict
        Extraction metadata (locations, statistics, etc.)
    """
    if stride is None:
        stride = patch_size

    if good_bands is None:
        good_bands = np.arange(img.bands)

    n_bands = len(good_bands)

    if verbose:
        print(f"\nExtracting {patch_size}x{patch_size} patches...")
        print(f"  Image size: {img.samples} x {img.lines}")
        print(f"  Bands: {n_bands}")
        print(f"  Stride: {stride}")

    patches = []
    locations = []
    rejected_reasons = {}

    # Calculate number of possible patches
    n_patches_x = (img.samples - patch_size) // stride + 1
    n_patches_y = (img.lines - patch_size) // stride + 1
    total_possible = n_patches_x * n_patches_y

    if verbose:
        print(f"  Possible patches: {total_possible}")

    # Load entire image (for efficiency)
    if verbose:
        print(f"  Loading image data...")
    cube = img.read_cube(bands=good_bands)

    # Extract patches
    extracted = 0
    validated = 0

    for y in range(0, img.lines - patch_size + 1, stride):
        for x in range(0, img.samples - patch_size + 1, stride):
            # Extract patch
            patch = cube[y:y+patch_size, x:x+patch_size, :]

            extracted += 1

            # Validate if requested
            if validate:
                is_valid, reason = validate_patch(patch)

                if not is_valid:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    continue

            # Accept patch
            patches.append(patch)
            locations.append((y, x))
            validated += 1

            # Check max patches limit
            if max_patches is not None and validated >= max_patches:
                break

        if max_patches is not None and validated >= max_patches:
            break

        # Progress
        if verbose and extracted % 100 == 0:
            print(f"    Extracted {extracted}/{total_possible}, "
                  f"valid: {validated}, rejected: {extracted - validated}")

    if verbose:
        print(f"\n  Final: {validated} valid patches from {extracted} candidates")
        if len(rejected_reasons) > 0:
            print(f"  Rejection reasons:")
            for reason, count in sorted(rejected_reasons.items(),
                                       key=lambda x: x[1],
                                       reverse=True):
                print(f"    - {reason}: {count}")

    metadata = {
        'patch_size': patch_size,
        'stride': stride,
        'n_bands': n_bands,
        'good_bands': good_bands,
        'n_extracted': extracted,
        'n_valid': validated,
        'locations': locations,
        'rejected_reasons': rejected_reasons,
        'image_shape': (img.lines, img.samples),
        'wavelengths': img.wavelengths[good_bands].tolist() if len(img.wavelengths) > 0 else [],
        'gsd': img.gsd,
    }

    return patches, metadata


def load_aviris_classic(aviris_path: str,
                       filter_bad_bands: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convenience function to load AVIRIS Classic reflectance data.

    Parameters:
    -----------
    aviris_path : str or Path
        Path to AVIRIS Classic corrected reflectance image (*_corr_*_img)
    filter_bad_bands : bool
        If True, exclude bad bands based on bbl and correction factors

    Returns:
    --------
    cube : np.ndarray
        Reflectance cube (shape: [lines, samples, bands])
    wavelengths : np.ndarray
        Wavelength array (in nm)
    metadata : dict
        Additional metadata
    """
    # Load image
    img = AVIRISClassicImage(aviris_path)

    # Get good bands
    if filter_bad_bands:
        good_bands = img.get_good_bands(use_bbl=True,
                                       filter_correction_factors=True)
    else:
        good_bands = None

    # Read cube
    cube = img.read_cube(bands=good_bands)

    # Get wavelengths
    wavelengths = img.get_wavelengths(bands=good_bands)

    metadata = {
        'gsd': img.gsd,
        'n_bands_original': img.bands,
        'n_bands_used': cube.shape[2],
        'good_bands': good_bands,
    }

    return cube, wavelengths, metadata


# Test the loader
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python aviris_classic_loader.py <path_to_aviris_classic_img>")
        print("\nExample:")
        print("  python aviris_classic_loader.py /iex_data/AVIRIS_Classic/f180601t01p00r06/f180601t01p00r06_rfl_v1k1/f180601t01p00r06_corr_v1k1_img")
        sys.exit(1)

    aviris_path = sys.argv[1]

    print("=" * 70)
    print("AVIRIS Classic Loader Test")
    print("=" * 70)

    # Load data
    print("\nLoading AVIRIS Classic reflectance data...")
    cube, wavelengths, metadata = load_aviris_classic(aviris_path,
                                                       filter_bad_bands=True)

    print(f"\nResults:")
    print(f"  Cube shape: {cube.shape}")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"  GSD: {metadata['gsd']:.2f} m")
    print(f"  Bands: {metadata['n_bands_used']}/{metadata['n_bands_original']}")
    print(f"  Data range: {cube.min():.4f} - {cube.max():.4f}")
    print(f"  Data mean: {cube.mean():.4f}")
    print(f"  Units: Surface reflectance (0-1)")

    # Test patch extraction
    print("\n" + "=" * 70)
    print("Testing patch extraction (small test)...")
    print("=" * 70)

    img = AVIRISClassicImage(aviris_path)
    good_bands = img.get_good_bands()

    patches, patch_metadata = extract_patches(img,
                                              patch_size=256,
                                              good_bands=good_bands,
                                              validate=True,
                                              max_patches=10,
                                              verbose=True)

    print(f"\nPatch extraction complete!")
    print(f"  Valid patches: {len(patches)}")
    if len(patches) > 0:
        print(f"  Patch shape: {patches[0].shape}")
        print(f"  Patch data range: {patches[0].min():.4f} - {patches[0].max():.4f}")

    print("\n" + "=" * 70)
    print("Test complete!")
