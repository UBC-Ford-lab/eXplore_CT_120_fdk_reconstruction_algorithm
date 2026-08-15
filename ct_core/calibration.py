"""
Projection preprocessing: transmission -> line integrals.

This module owns steps 1-2 of the standard chain, which turn raw detector
counts into the line integrals a reconstructor consumes:

1. Flat-field correction: T = (I - I_dark) / (I_bright - I_dark)
2. Log transformation:    p = -log(T)

Step 3 is the reconstruction itself, which yields the linear attenuation
coefficient mu (mm^-1).

Step 4 -- the HU conversion -- deliberately does NOT live here. It is fitted
once, downstream of every backend, in ``ct_core.hu_calibration``, from two
anchors measured in the reconstructed volume's own histogram. The one-point
map that used to live in this file (``mu_to_hu``, plus a family of phantom
insert-ROI polynomial fitters) has been removed: it fitted a single degree of
freedom against an affine error, relied on a scanner constant back-fitted to
one 2022 scan that did not transfer between reconstructions, and clipped to
[-1024, 4095] before anything could measure the volume -- which pinned 19-44 %
of all voxels onto the floor and destroyed the air peak.

The name is kept because the flat/dark fields it loads are the scanner's
calibration frames.

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
#
# Scanner-specific effective μ_water for the GE eXplore CT 120 at 80 kVp:
#   MU_WATER_80KV_NO_BHC  — empirically derived by aligning SIRT (no BHC,
#       physics HU) tissue values with scanner FDK output (~160 HU target).
#       Back-calculated: 0.0219 mm⁻¹ (polychromatic beam, ~55 keV effective).
#   MU_WATER_80KV_WITH_BHC — estimated from no-BHC value by applying the
#       BHC-induced μ reduction factor (4–7.3% from cross-validation data).
#       Midpoint: 0.0219 × (1 - 0.056) ≈ 0.0207 mm⁻¹. Needs phantom validation.
MU_WATER_80KV_NO_BHC   = 0.0219  # mm⁻¹  empirical, no-BHC pipeline
MU_WATER_80KV_WITH_BHC = 0.0207  # mm⁻¹  estimated, BHC pipeline (needs phantom validation)

# Alias for external code and diagnostic prints. Points to the no-BHC value
# since that is the default pipeline. Backends auto-select the correct constant
# based on whether BHC is active; this alias is the fallback for external code
# that imports MU_WATER_80KV directly.
MU_WATER_80KV = MU_WATER_80KV_NO_BHC

# Linear attenuation coefficient of air (effectively zero)
MU_AIR = 0.0  # mm⁻¹

# Sinogram-domain BHC polynomial for the GE eXplore CT 120 at 80 kVp.
# Calibrated from water phantom (Scan_1680) using bhc_sinogram_calibration.py.
# Applied as: p_corrected = c1*p + c2*p² (no constant term).
# Scanner-specific (depends on X-ray spectrum), NOT scan-specific.
BHC_COEFFICIENTS_80KV = [0.856, 0.21]


def default_mu_water(mu_water=None, bhc_coeffs=None) -> float:
    """Resolve the effective mu_water for HU conversion.

    An explicit ``mu_water`` always wins; otherwise the pipeline-matched
    constant is selected by whether sinogram-domain BHC is active (BHC
    lowers the effective mu_water by removing beam-hardening inflation).
    One definition for every backend — FDK, ASTRA, TIGRE, and future ones.
    """
    if mu_water is not None:
        return float(mu_water)
    return MU_WATER_80KV_WITH_BHC if bhc_coeffs is not None else MU_WATER_80KV_NO_BHC
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
    soft_clip_sharpness: float = 200.0,
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
