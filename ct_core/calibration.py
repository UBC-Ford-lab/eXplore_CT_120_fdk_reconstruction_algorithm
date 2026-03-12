"""
HU Calibration Module for CT Reconstruction Pipeline

Provides log transformation and Hounsfield Unit calibration for CT projections.

Standard CT reconstruction requires:
1. Flat-field correction: T = (I - I_dark) / (I_bright - I_dark)
2. Log transformation: p = -log(T) to convert transmission to line integrals
3. FDK reconstruction yields linear attenuation coefficient μ (proportional)
4. HU conversion: HU = (μ - μ_water) / μ_water × 1000

Author: Falk Wiegmann, University of British Columbia
Date: January 2026
"""

import numpy as np
import xmltodict
import os
import sys
from typing import Dict, Optional, Tuple

# Import VFF reader using relative import
from .vff_io import read_vff


# =============================================================================
# Soft Clamping for Gibbs Ringing Prevention
# =============================================================================

def soft_clamp_lower(x: np.ndarray, min_val: float = 0.0, sharpness: float = 50.0) -> np.ndarray:
    """
    Soft clamping using softplus - smooth minimum without derivative discontinuity.

    Avoids Gibbs ringing artifacts from hard clamping. The softplus function
    provides a smooth approximation to max(x, min_val) that is differentiable
    everywhere, preventing the ramp filter from amplifying discontinuities.

    Args:
        x: Input array
        min_val: Soft minimum value (default 0.0)
        sharpness: Transition sharpness (50.0 = ~0.06 transition width)
                   Higher values = sharper transition, closer to hard clamp
                   Transition width ≈ 3/sharpness

    Returns:
        Soft-clamped array approaching min_val smoothly
    """
    scaled = sharpness * (x - min_val)
    # Use np.where to avoid overflow: for large positive values, return x directly
    # For values near or below min_val, compute softplus
    result = np.where(
        scaled > 20,
        x,
        min_val + np.log1p(np.exp(np.clip(scaled, -700, 20))) / sharpness
    )
    return result.astype(np.float32)


def soft_clamp_upper(x: np.ndarray, max_val: float = 1.0, sharpness: float = 50.0) -> np.ndarray:
    """
    Soft clamping to maximum value using inverted softplus.

    Complements soft_clamp_lower() for bounding values from above.
    Uses the identity: soft_max(x, M) = M - soft_min(M - x, 0)

    Args:
        x: Input array
        max_val: Soft maximum value (default 1.0)
        sharpness: Transition sharpness (50.0 = ~0.06 transition width)
                   Higher values = sharper transition, closer to hard clamp
                   Transition width ≈ 3/sharpness

    Returns:
        Soft-clamped array approaching max_val smoothly from below
    """
    # Transform to use soft_clamp_lower: min(x, max_val) = max_val - max(max_val - x, 0)
    return max_val - soft_clamp_lower(max_val - x, min_val=0.0, sharpness=sharpness)


# =============================================================================
# Physical Constants
# =============================================================================

# Literature value for linear attenuation coefficient of water at ~80 kV
# Reference: NIST XCOM database, effective energy for typical CT beam
# Units: mm⁻¹ (0.184 cm⁻¹ × 0.1 cm/mm = 0.0184 mm⁻¹)
# NOTE: Previous value was 0.00184 (10x error in unit conversion)
MU_WATER_80KV = 0.0184  # mm⁻¹

# Linear attenuation coefficient of air (effectively zero)
MU_AIR = 0.0  # mm⁻¹


def parse_calibration_from_xml(xml_path: str) -> Dict[str, float]:
    """
    Extract AirValue, WaterValue, and BoneHU from scan XML.

    XML Path: Series/Tasks/Recon/TaskParams/Advanced/

    Args:
        xml_path: Path to scan.xml file

    Returns:
        Dictionary with 'air_value', 'water_value', 'bone_hu'
    """
    with open(xml_path, 'r') as f:
        header = xmltodict.parse(f.read())

    # Navigate to Advanced calibration parameters
    # Path: Series -> Tasks -> Recon -> TaskParams -> Advanced
    try:
        recon = header['Series']['Tasks']['Recon']
        advanced = recon['TaskParams']['Advanced']

        return {
            'air_value': float(advanced.get('AirValue', 1.0)),
            'water_value': float(advanced.get('WaterValue', 1.0)),
            'bone_hu': float(advanced.get('BoneHU', 3100))
        }
    except (KeyError, TypeError) as e:
        print(f"Warning: Could not parse calibration from XML: {e}")
        print("Using default values: air_value=1.0, water_value=1.0, bone_hu=3100")
        return {
            'air_value': 1.0,
            'water_value': 1.0,
            'bone_hu': 3100
        }


def load_calibration_fields(scan_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bright and dark field calibration images from scan folder.

    These are used for proper flat-field correction:
    - bright.vff: Unattenuated beam reference (I₀) - air scan with no object
    - dark.vff: Electronic noise/offset reference

    Args:
        scan_folder: Path to original scan folder (e.g., scans/Scan_1681/)

    Returns:
        Tuple of (bright_field, dark_field) as 2D numpy arrays [height, width]

    Raises:
        FileNotFoundError: If bright.vff or dark.vff not found
    """
    bright_path = os.path.join(scan_folder, 'bright.vff')
    dark_path = os.path.join(scan_folder, 'dark.vff')

    if not os.path.exists(bright_path):
        raise FileNotFoundError(f"Bright field not found: {bright_path}")
    if not os.path.exists(dark_path):
        raise FileNotFoundError(f"Dark field not found: {dark_path}")

    # Read VFF files
    _, bright_field = read_vff(bright_path, verbose=False)
    _, dark_field = read_vff(dark_path, verbose=False)

    # Average if multiple frames (calibration images are often averaged)
    if bright_field.ndim == 3:
        bright_field = np.mean(bright_field, axis=0)
    if dark_field.ndim == 3:
        dark_field = np.mean(dark_field, axis=0)

    return bright_field.astype(np.float32), dark_field.astype(np.float32)


def flat_field_correction(
    projections: np.ndarray,
    bright_field: np.ndarray,
    dark_field: np.ndarray,
    epsilon: float = 1e-6,
    soft_clip: bool = True,
    soft_clip_sharpness: float = 50.0,
    upper_clamp: bool = True,
    upper_clamp_value: float = 1.05
) -> np.ndarray:
    """
    Apply flat-field correction to convert raw intensities to transmission.

    T = (I - I_dark) / (I_bright - I_dark)

    This normalizes detector response variations and converts raw intensities
    to transmission values (fraction of beam transmitted through object).

    Args:
        projections: Raw detector intensities [n_angles, height, width]
        bright_field: Unattenuated beam reference [height, width]
        dark_field: Electronic noise reference [height, width]
        epsilon: Minimum transmission value to prevent log(0)
        soft_clip: If True, use soft clipping (no Gibbs ringing). If False, hard clip.
        soft_clip_sharpness: Sharpness of soft clip transition (higher = sharper).
            Default 50.0 gives transition width of ~0.06, affecting T < 0.06.
            The softplus transition zone is approximately 3/sharpness around epsilon.
            Lower values (broader transition) reduce center ringing artifacts.
        upper_clamp: If True, also clamp T from above to prevent negative line integrals.
        upper_clamp_value: Maximum allowed transmission value (default 1.05).
            Values > 1.0 occur from noise in air regions. Setting to 1.05 allows
            small positive deviations while clamping large outliers that cause
            discontinuities at the T=1.0 boundary.

    Returns:
        Transmission values with soft floor at epsilon and optional soft ceiling

    Note:
        Hard clipping (soft_clip=False) creates a derivative discontinuity where
        all T < epsilon values jump to exactly epsilon. This produces line integrals
        of exactly -log(epsilon) ≈ 13.816 for epsilon=1e-6. The ramp filter amplifies
        this discontinuity into characteristic center ringing artifacts.

        Soft clipping uses softplus for a smooth transition, preventing Gibbs ringing
        while still enforcing the minimum transmission floor.

        Upper clamping prevents T > 1 values (from noise/calibration mismatch) from
        creating negative line integrals. The boundary where T crosses 1.0 can create
        coherent spatial patterns that manifest as center ringing after backprojection.
    """
    # Per-pixel I₀ normalization: corrects detector response variations
    # (beam vignetting, pixel gain) by normalising each pixel individually.
    # T(i,j) = (acq(i,j) - dark(i,j)) / (bright(i,j) - dark(i,j))
    denominator = (bright_field.astype(np.float64) - dark_field.astype(np.float64))
    denominator = np.clip(denominator, 1.0, None)  # guard dead pixels
    transmission = (projections - dark_field) / (denominator + epsilon)

    if soft_clip:
        # Soft clamp to epsilon - smooth transition prevents Gibbs ringing
        transmission = soft_clamp_lower(transmission, min_val=epsilon, sharpness=soft_clip_sharpness)
        # Also clamp from above to prevent negative line integrals
        if upper_clamp:
            transmission = soft_clamp_upper(transmission, max_val=upper_clamp_value, sharpness=soft_clip_sharpness)
    else:
        # Hard clamp (legacy behavior - causes ringing)
        if upper_clamp:
            transmission = np.clip(transmission, epsilon, upper_clamp_value)
        else:
            transmission = np.clip(transmission, epsilon, None)

    return transmission.astype(np.float32)


def log_transform_transmission(
    transmission: np.ndarray,
    clamp_mode: str = "none"
) -> np.ndarray:
    """
    Convert transmission to line integrals of attenuation.

    p = -log(T) where T is transmission (already normalized by flat-field)

    This converts transmission ratios to line integrals of the linear
    attenuation coefficient μ, as required for FDK reconstruction.

    Physical interpretation:
        T = exp(-∫μ dl) → p = -log(T) = ∫μ dl

    Args:
        transmission: Flat-field corrected transmission values [n_angles, h, w]
                     Values should be in range (0, inf), typically near 1.0
        clamp_mode: How to handle negative line integrals (from T > 1 noise)
            - "none": No clamping, preserve noise (recommended, no Gibbs ringing)
            - "soft": Softplus smooth clamp to 0 (minimal ringing)
            - "hard": np.maximum hard clamp (causes Gibbs ringing - not recommended)

    Returns:
        Line integrals of linear attenuation coefficient
        Range typically [0, ~5] for medical CT

    Note:
        Transmission values > 1.0 (from noise in air regions) produce negative
        line integrals, which are physically meaningless (negative attenuation).

        The clamping mode affects Gibbs ringing artifacts:
        - "hard" clamping creates a derivative discontinuity that the ramp filter
          amplifies into characteristic ringing artifacts at the image center.
        - "soft" clamping uses softplus for a smooth transition (minimal ringing).
        - "none" preserves the continuous signal entirely (no ringing), but allows
          small negative μ values in air regions (typically < -50 HU noise floor).
    """
    line_integral = -np.log(transmission)

    if clamp_mode == "none":
        # No clamping - allows small negative values from noise
        # Continuous signal → no Gibbs ringing
        return line_integral.astype(np.float32)
    elif clamp_mode == "soft":
        # Softplus smooth clamp - minimal ringing
        return soft_clamp_lower(line_integral, min_val=0.0, sharpness=50.0)
    elif clamp_mode == "hard":
        # Hard clamp - causes Gibbs ringing (not recommended)
        return np.maximum(line_integral, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}. Use 'none', 'soft', or 'hard'.")


def convert_to_hounsfield_units(
    mu_volume: np.ndarray,
    mu_water: float
) -> np.ndarray:
    """
    Convert linear attenuation coefficient volume to Hounsfield Units.

    HU = (μ - μ_water) / μ_water × 1000

    Standard HU values:
        - Air: -1000 HU
        - Water: 0 HU
        - Soft tissue: 20-80 HU
        - Bone: +1000 to +3000 HU

    Args:
        mu_volume: Reconstructed linear attenuation coefficient volume
        mu_water: Linear attenuation coefficient of water

    Returns:
        Volume in Hounsfield Units
    """
    hu_volume = ((mu_volume - mu_water) / mu_water) * 1000.0
    return hu_volume.astype(np.float32)


def validate_hu_calibration(
    volume: np.ndarray,
    expected_air_hu: float = -1000.0,
    expected_water_hu: float = 0.0,
    tolerance: float = 100.0
) -> Dict[str, float]:
    """
    Validate HU calibration by checking volume statistics.

    Args:
        volume: Reconstructed volume in HU
        expected_air_hu: Expected HU for air regions
        expected_water_hu: Expected HU for water regions
        tolerance: Acceptable deviation from expected values

    Returns:
        Dictionary with volume statistics
    """
    stats = {
        'min': float(np.min(volume)),
        'max': float(np.max(volume)),
        'mean': float(np.mean(volume)),
        'std': float(np.std(volume)),
        'percentile_1': float(np.percentile(volume, 1)),
        'percentile_99': float(np.percentile(volume, 99)),
    }

    # Check if air regions (low values) are close to -1000 HU
    if stats['percentile_1'] < expected_air_hu - tolerance:
        print(f"Warning: Air regions ({stats['percentile_1']:.0f} HU) below expected {expected_air_hu} HU")
    elif stats['percentile_1'] > expected_air_hu + tolerance:
        print(f"Warning: Air regions ({stats['percentile_1']:.0f} HU) above expected {expected_air_hu} HU")
    else:
        print(f"Air regions: {stats['percentile_1']:.0f} HU (expected ~{expected_air_hu:.0f} HU) - OK")

    return stats


# =============================================================================
# Phantom Insert Calibration Data
# =============================================================================

# Measured HU values from physical normalization vs true HU from phantom datasheet.
# Used to fit a polynomial correction that maps uncalibrated physical HU to
# correct HU values. Columns: (measured_HU_physical, true_HU_datasheet)
PHANTOM_CALIBRATION = np.array([
    [708.345, 1460],
    [-984.921, -940],
    [-454.535, 30],
    [-414.576, 80],
    [-374.798, 130],
    [-309.299, 220],
    [-168.703, 410],
    [118.486, 770],
])

# True HU values for each phantom insert (from manufacturer datasheet).
# Ordered by ascending true HU. Names are descriptive labels.
PHANTOM_INSERT_TRUE_HU = [
    ("air",       -940),
    ("insert_30",   30),
    ("insert_80",   80),
    ("insert_130", 130),
    ("insert_220", 220),
    ("insert_410", 410),
    ("insert_770", 770),
    ("bone",      1460),
]


# =============================================================================
# Per-method polynomial calibration coefficients
# =============================================================================
# Nested by scan type ("half_scan" or "full_scan"), then by reconstruction
# method.  Coefficients are for np.polyval: HU_true = polyval(c, HU_raw).
#
# These are fitted from phantom self-calibration (--roi-config) and stored here
# so non-phantom scans can use the correct per-method polynomial without inserts.
#
# Half-scan and full-scan acquisitions have different numbers of projections,
# leading to different backprojection normalizations and thus different
# uncalibrated value ranges.  A polynomial fitted on one scan type does NOT
# transfer to the other.
#
# To regenerate: run a phantom reconstruction with --roi-config, note the
# printed coefficients, and update this dict.
CALIBRATION_COEFFICIENTS = {
    # ---------------------------------------------------------------
    # Half-scan coefficients  (from Scan_1988, QRM phantom, 7 inserts, degree-2)
    # ---------------------------------------------------------------
    "half_scan": {
        # FDK with physical_normalization=True, hamming filter (RMS 39.2 HU)
        "fdk": [-0.0007132818094142021, 2.128889255893252, 1896.855740800915],

        # ASTRA SIRT (100 iterations, min_constraint=0) (RMS 36.2 HU)
        "astra_sirt": [-3.6050254803021966e-05, 1.3530360794967076, -92.19078961741268],

        # TIGRE OS-SART (100 iterations) (RMS 35.2 HU)
        "tigre_ossart": [-4.135028143804179e-05, 1.1690824907050639, -116.2569444229479],
    },

    # ---------------------------------------------------------------
    # Full-scan coefficients  (from Scan_1989, QRM phantom, 7 inserts, degree-2)
    # ---------------------------------------------------------------
    "full_scan": {
        # FDK with physical_normalization=True, hamming filter (RMS 31.8 HU)
        "fdk": [-0.00016957771890721148, 1.4068691691082866, 634.9725504813525],

        # ASTRA SIRT (100 iterations, min_constraint=0) (RMS 47.2 HU)
        "astra_sirt": [-5.6346843901515365e-05, 1.402740763274877, -109.83505093406858],

        # TIGRE OS-SART (100 iterations) (RMS 36.9 HU)
        "tigre_ossart": [-4.8887881440773636e-05, 1.2077501396636758, -77.80945440483856],
    },
}

# Default scan type when not specified (backward compatibility)
DEFAULT_SCAN_TYPE = "half_scan"


def get_calibration_coefficients(method: str, scan_type: str = None):
    """
    Look up stored polynomial calibration coefficients for a method.

    Args:
        method: Method key (e.g., 'fdk', 'astra_sirt', 'tigre_ossart')
        scan_type: 'half_scan' or 'full_scan'. If None, uses DEFAULT_SCAN_TYPE.

    Returns:
        numpy array of polynomial coefficients for np.polyval, or None if
        no stored coefficients exist for this method/scan_type combination.
    """
    if scan_type is None:
        scan_type = DEFAULT_SCAN_TYPE

    scan_dict = CALIBRATION_COEFFICIENTS.get(scan_type)
    if scan_dict is None:
        print(f"  WARNING: Unknown scan_type '{scan_type}'. "
              f"Available: {list(CALIBRATION_COEFFICIENTS.keys())}")
        return None

    coeffs = scan_dict.get(method)
    if coeffs is not None:
        return np.array(coeffs)
    return None


def fit_hu_calibration(calibration_data: np.ndarray, degree: int = 2):
    """
    Fit a polynomial mapping measured physical HU to true HU.

    Args:
        calibration_data: Nx2 array of (measured_HU, true_HU) pairs
        degree: Polynomial degree (default 2 = quadratic)

    Returns:
        (poly_coeffs, residuals_rms, residuals_per_point)
        poly_coeffs are for np.polyval: HU_true = polyval(coeffs, HU_measured).
    """
    measured = calibration_data[:, 0]
    expected = calibration_data[:, 1]
    coeffs = np.polyfit(measured, expected, degree)
    predicted = np.polyval(coeffs, measured)
    residuals = expected - predicted
    rms = float(np.sqrt(np.mean(residuals**2)))
    return coeffs, rms, residuals


def measure_insert_rois(volume: np.ndarray, insert_rois: list,
                        z_range: tuple) -> list:
    """
    Measure mean pixel value inside each circular insert ROI.

    Args:
        volume: 3D array (z, y, x)
        insert_rois: list of dicts with keys 'name', 'cy', 'cx', 'radius', 'true_hu'
        z_range: (z_start, z_end) slice range to average over

    Returns:
        List of dicts with 'name', 'cy', 'cx', 'radius', 'true_hu',
        'measured_mean', 'measured_std'
    """
    sl = volume[z_range[0]:z_range[1]].mean(axis=0)
    results = []
    for roi in insert_rois:
        cy, cx, r = roi['cy'], roi['cx'], roi['radius']
        yy, xx = np.ogrid[cy - r:cy + r, cx - r:cx + r]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        vals = sl[cy - r:cy + r, cx - r:cx + r][mask]
        results.append({
            **roi,
            'measured_mean': float(vals.mean()),
            'measured_std': float(vals.std()),
        })
    return results


def calibrate_volume_polynomial(volume: np.ndarray,
                                calibration_data: np.ndarray,
                                degree: int = 2,
                                clip_range: tuple = (-1024, 4095)
                                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit polynomial from calibration pairs and apply to entire volume.

    Args:
        volume: 3D array (z, y, x) in uncalibrated values
        calibration_data: Nx2 array of (measured, true_hu) pairs
        degree: polynomial degree
        clip_range: (min, max) HU clipping range

    Returns:
        (calibrated_volume, poly_coeffs, rms_residual)
    """
    coeffs, rms, _ = fit_hu_calibration(calibration_data, degree)
    calibrated = np.polyval(coeffs, volume).astype(np.float32)
    if clip_range is not None:
        calibrated = np.clip(calibrated, clip_range[0], clip_range[1])
    return calibrated, coeffs, rms


def plot_calibration_diagnostic(insert_measurements: list,
                                coeffs: np.ndarray,
                                volume_slice: np.ndarray,
                                output_path: str):
    """
    Generate a diagnostic figure showing ROI placement and polynomial fit.

    Args:
        insert_measurements: list of dicts from measure_insert_rois()
        coeffs: polynomial coefficients from fit
        volume_slice: 2D slice for background image
        output_path: where to save the figure (without extension)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) ROI overlay on phantom slice
    ax = axes[0]
    vmin = float(np.percentile(volume_slice, 1))
    vmax = float(np.percentile(volume_slice, 99))
    ax.imshow(volume_slice, cmap='gray', vmin=vmin, vmax=vmax)
    for m in insert_measurements:
        circle = Circle((m['cx'], m['cy']), m['radius'],
                         fill=False, edgecolor='cyan', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(m['cx'] + m['radius'] + 5, m['cy'],
                f"{m['name']}\n{m['measured_mean']:.0f}",
                fontsize=6, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='black', alpha=0.7))
    ax.set_title('(a) Insert ROI placement', fontweight='bold')

    # (b) Measured vs True with polynomial fit
    ax = axes[1]
    measured = np.array([m['measured_mean'] for m in insert_measurements])
    true_hu = np.array([m['true_hu'] for m in insert_measurements])

    ax.scatter(measured, true_hu, c='blue', s=60, zorder=5, label='Insert measurements')
    for m in insert_measurements:
        ax.annotate(m['name'], (m['measured_mean'], m['true_hu']),
                    textcoords='offset points', xytext=(5, 5), fontsize=7)

    x_fit = np.linspace(measured.min() - 100, measured.max() + 100, 200)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, label=f'Poly deg {len(coeffs)-1} fit')
    ax.plot(x_fit, x_fit, 'k--', linewidth=0.8, alpha=0.5, label='Identity (y=x)')

    ax.set_xlabel('Measured (uncalibrated)')
    ax.set_ylabel('True HU (datasheet)')
    ax.set_title('(b) Calibration curve', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Residuals
    ax = axes[2]
    predicted = np.polyval(coeffs, measured)
    residuals = true_hu - predicted
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    ax.bar(range(len(insert_measurements)),
           residuals, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(insert_measurements)))
    ax.set_xticklabels([m['name'] for m in insert_measurements],
                       rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Residual (True - Predicted) [HU]')
    ax.set_title(f'(c) Residuals (RMS = {rms:.1f} HU)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(output_path + ext, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Diagnostic saved: {output_path}.png/.pdf")


if __name__ == '__main__':
    # Test with Scan_1681
    from .paths import RESULTS_DIR
    xml_path = str(RESULTS_DIR / '..' / 'scans' / 'Scan_1681' / 'scan.xml')

    print("Testing HU calibration module...")
    print("=" * 60)

    # Parse calibration
    calibration = parse_calibration_from_xml(str(xml_path))
    print(f"\nCalibration values from XML:")
    print(f"  Air Value:   {calibration['air_value']}")
    print(f"  Water Value: {calibration['water_value']}")
    print(f"  Bone HU:     {calibration['bone_hu']}")
    print(f"  Literature mu_water at 80 keV: {MU_WATER_80KV} mm⁻¹")

    # Test polynomial calibration fit
    coeffs, rms, _ = fit_hu_calibration(PHANTOM_CALIBRATION)
    print(f"\nPolynomial calibration fit (degree 2):")
    print(f"  Coefficients: {coeffs}")
    print(f"  RMS residual: {rms:.1f} HU")

    print("\n" + "=" * 60)
    print("Module test complete.")
