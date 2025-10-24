"""
ENVI File Loader for AVIRIS-NG Data

Loads AVIRIS-NG radiance imagery in ENVI format (BIL interleave).

Format details:
- Binary 32-bit little-endian floating point
- Band Interleaved by Line (BIL) format
- Units: microwatts per cm² per nm per steradian
"""

import numpy as np
from pathlib import Path
import re
from typing import Dict, Tuple, Optional


class ENVIHeader:
    """
    Parser for ENVI .hdr files.
    """

    def __init__(self, hdr_path):
        """
        Load and parse ENVI header file.

        Parameters:
        -----------
        hdr_path : str or Path
            Path to .hdr file
        """
        self.hdr_path = Path(hdr_path)
        self.params = {}

        self._parse_header()

    def _parse_header(self):
        """Parse ENVI header file."""
        with open(self.hdr_path, 'r') as f:
            content = f.read()

        # Parse key-value pairs
        # Handle both single-line and multi-line values (arrays in braces)
        pattern = r'(\w[\w\s]*?)\s*=\s*({[^}]*}|[^\n]*)'

        for match in re.finditer(pattern, content):
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Parse array values (in braces)
            if value.startswith('{') and value.endswith('}'):
                # Extract array content
                array_str = value[1:-1]
                # Split by comma and convert to appropriate type
                try:
                    # Try float first
                    array_values = [float(x.strip()) for x in array_str.split(',')]
                except ValueError:
                    # Fall back to string
                    array_values = [x.strip() for x in array_str.split(',')]
                self.params[key] = array_values
            else:
                # Single value - try to convert to number
                try:
                    # Try int first
                    self.params[key] = int(value)
                except ValueError:
                    try:
                        # Try float
                        self.params[key] = float(value)
                    except ValueError:
                        # Keep as string
                        self.params[key] = value

    def get(self, key, default=None):
        """Get parameter value."""
        return self.params.get(key, default)

    def __getitem__(self, key):
        """Get parameter value (dict-like access)."""
        return self.params[key]

    def __repr__(self):
        return f"ENVIHeader({self.hdr_path.name})"


class ENVIImage:
    """
    ENVI image file loader for AVIRIS-NG radiance data.
    """

    # ENVI data type codes
    DTYPE_MAP = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }

    INTERLEAVE_MAP = {
        'bsq': 'BSQ',  # Band Sequential
        'bil': 'BIL',  # Band Interleaved by Line
        'bip': 'BIP',  # Band Interleaved by Pixel
    }

    def __init__(self, img_path):
        """
        Initialize ENVI image loader.

        Parameters:
        -----------
        img_path : str or Path
            Path to ENVI image file (without .hdr extension)
        """
        self.img_path = Path(img_path)

        # Find header file
        hdr_path = self.img_path.with_suffix(self.img_path.suffix + '.hdr')
        if not hdr_path.exists():
            # Try without extension
            hdr_path = Path(str(self.img_path) + '.hdr')

        if not hdr_path.exists():
            raise FileNotFoundError(f"Header file not found for {self.img_path}")

        # Parse header
        self.header = ENVIHeader(hdr_path)

        # Extract key parameters
        self.samples = self.header['samples']
        self.lines = self.header['lines']
        self.bands = self.header['bands']
        self.data_type = self.header['data type']
        self.interleave = self.header['interleave'].lower()
        self.byte_order = self.header.get('byte order', 0)
        self.header_offset = self.header.get('header offset', 0)

        # Get wavelengths if available
        self.wavelengths = np.array(self.header.get('wavelength', []))
        self.fwhm = np.array(self.header.get('fwhm', []))

        # Get numpy dtype
        dtype_class = self.DTYPE_MAP.get(self.data_type)
        if dtype_class is None:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        # Create dtype instance and handle byte order
        if self.byte_order == 1:  # Big-endian
            self.dtype = np.dtype(dtype_class).newbyteorder('>')
        else:  # Little-endian
            self.dtype = np.dtype(dtype_class).newbyteorder('<')

        print(f"ENVI Image: {self.img_path.name}")
        print(f"  Dimensions: {self.samples} x {self.lines} x {self.bands}")
        print(f"  Data type: {self.dtype}")
        print(f"  Interleave: {self.interleave.upper()}")
        if len(self.wavelengths) > 0:
            print(f"  Wavelengths: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")

    def read_cube(self, bands=None, lines=None, samples=None):
        """
        Read entire image cube or subset.

        Parameters:
        -----------
        bands : slice, list, or None
            Band indices to read (None = all bands)
        lines : slice, list, or None
            Line indices to read (None = all lines)
        samples : slice, list, or None
            Sample indices to read (None = all samples)

        Returns:
        --------
        cube : np.ndarray
            Image cube (shape: [lines, samples, bands])
        """
        # Convert to slices if None
        if bands is None:
            bands = slice(None)
        if lines is None:
            lines = slice(None)
        if samples is None:
            samples = slice(None)

        # Read based on interleave format
        if self.interleave == 'bil':
            cube = self._read_bil(bands, lines, samples)
        elif self.interleave == 'bsq':
            cube = self._read_bsq(bands, lines, samples)
        elif self.interleave == 'bip':
            cube = self._read_bip(bands, lines, samples)
        else:
            raise ValueError(f"Unsupported interleave: {self.interleave}")

        return cube

    def _read_bil(self, bands, lines, samples):
        """
        Read Band Interleaved by Line (BIL) format.

        BIL format: [line 0 band 0, line 0 band 1, ..., line 1 band 0, ...]
        """
        # For simplicity, read entire file then subset
        # For large files, would want to read only requested data

        # Calculate total size
        total_elements = self.lines * self.samples * self.bands

        # Read entire file
        with open(self.img_path, 'rb') as f:
            f.seek(self.header_offset)
            data = np.fromfile(f, dtype=self.dtype, count=total_elements)

        # Reshape to [lines, bands, samples]
        data = data.reshape(self.lines, self.bands, self.samples)

        # Transpose to [lines, samples, bands]
        data = np.transpose(data, (0, 2, 1))

        # Apply subset
        return data[lines, samples, bands]

    def _read_bsq(self, bands, lines, samples):
        """
        Read Band Sequential (BSQ) format.

        BSQ format: [band 0 all lines, band 1 all lines, ...]
        """
        total_elements = self.lines * self.samples * self.bands

        with open(self.img_path, 'rb') as f:
            f.seek(self.header_offset)
            data = np.fromfile(f, dtype=self.dtype, count=total_elements)

        # Reshape to [bands, lines, samples]
        data = data.reshape(self.bands, self.lines, self.samples)

        # Transpose to [lines, samples, bands]
        data = np.transpose(data, (1, 2, 0))

        # Apply subset
        return data[lines, samples, bands]

    def _read_bip(self, bands, lines, samples):
        """
        Read Band Interleaved by Pixel (BIP) format.

        BIP format: [pixel 0 all bands, pixel 1 all bands, ...]
        """
        total_elements = self.lines * self.samples * self.bands

        with open(self.img_path, 'rb') as f:
            f.seek(self.header_offset)
            data = np.fromfile(f, dtype=self.dtype, count=total_elements)

        # Reshape to [lines, samples, bands]
        data = data.reshape(self.lines, self.samples, self.bands)

        # Apply subset
        return data[lines, samples, bands]

    def get_wavelengths(self, bands=None):
        """
        Get wavelength array for specified bands.

        Parameters:
        -----------
        bands : slice, list, or None
            Band indices (None = all bands)

        Returns:
        --------
        wavelengths : np.ndarray
            Wavelength values (in nm)
        """
        if len(self.wavelengths) == 0:
            return None

        if bands is None:
            return self.wavelengths
        else:
            return self.wavelengths[bands]

    def get_good_bands(self, exclude_water_bands=True):
        """
        Get indices of good quality bands (excluding water absorption regions).

        Parameters:
        -----------
        exclude_water_bands : bool
            If True, exclude atmospheric water absorption bands

        Returns:
        --------
        good_bands : np.ndarray
            Array of good band indices
        """
        if len(self.wavelengths) == 0:
            return np.arange(self.bands)

        good_bands = np.ones(self.bands, dtype=bool)

        if exclude_water_bands:
            # Exclude major water absorption bands
            # Water absorption regions (approximate):
            # 1350-1470 nm, 1780-1980 nm, >2400 nm

            water_regions = [
                (1350, 1470),  # Strong water absorption
                (1780, 1980),  # Strong water absorption
                (2400, 3000),  # Very strong water absorption
            ]

            for wl_min, wl_max in water_regions:
                bad_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
                good_bands &= ~bad_mask

        return np.where(good_bands)[0]

    def get_map_info(self):
        """Extract map projection information from header."""
        map_info_str = self.header.get('map info', '')
        if isinstance(map_info_str, list):
            # Already parsed as array
            return map_info_str
        elif isinstance(map_info_str, str) and map_info_str.startswith('{'):
            # Parse manually
            map_info_str = map_info_str[1:-1]  # Remove braces
            return [x.strip() for x in map_info_str.split(',')]
        return []


def load_aviris_ng(aviris_path, exclude_water_bands=True):
    """
    Convenience function to load AVIRIS-NG radiance data.

    Parameters:
    -----------
    aviris_path : str or Path
        Path to AVIRIS-NG radiance image (*_rdn_*_img)
    exclude_water_bands : bool
        If True, load only good bands (excluding water absorption)

    Returns:
    --------
    cube : np.ndarray
        Radiance cube (shape: [lines, samples, bands])
    wavelengths : np.ndarray
        Wavelength array (in nm)
    metadata : dict
        Additional metadata
    """
    # Load ENVI image
    img = ENVIImage(aviris_path)

    # Get good bands
    if exclude_water_bands:
        good_bands = img.get_good_bands(exclude_water_bands=True)
        print(f"  Using {len(good_bands)}/{img.bands} bands (excluding water absorption)")
    else:
        good_bands = None

    # Read cube
    cube = img.read_cube(bands=good_bands)

    # Get wavelengths
    wavelengths = img.get_wavelengths(bands=good_bands)

    # Get map info
    map_info = img.get_map_info()
    pixel_size = float(map_info[5]) if len(map_info) > 5 else None

    metadata = {
        'pixel_size': pixel_size,
        'map_info': map_info,
        'n_bands_original': img.bands,
        'n_bands_used': cube.shape[2],
    }

    return cube, wavelengths, metadata


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python envi_loader.py <path_to_aviris_img>")
        print("\nExample:")
        print("  python envi_loader.py /raid/AVIRIS_NG/imagery/ang20190624t230039_rdn_v2u1/ang20190624t230039_rdn_v2u1_img")
        sys.exit(1)

    aviris_path = sys.argv[1]

    print("=" * 70)
    print("ENVI Loader Test - AVIRIS-NG")
    print("=" * 70)

    # Load data
    print("\nLoading AVIRIS-NG data...")
    cube, wavelengths, metadata = load_aviris_ng(aviris_path, exclude_water_bands=True)

    print(f"\nResults:")
    print(f"  Cube shape: {cube.shape}")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"  Pixel size: {metadata['pixel_size']} m")
    print(f"  Data range: {cube.min():.2f} - {cube.max():.2f}")
    print(f"  Data mean: {cube.mean():.2f}")
    print(f"  Units: microwatts per cm² per nm per steradian")

    print("\n" + "=" * 70)
    print("Test complete!")
