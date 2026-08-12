"""
Shared sinogram preprocessing for iterative backends (ASTRA, TIGRE).

Pipeline: flat-field → log → BHC → ring correction.
Cone-beam weighting and ramp filtering are FDK-specific and not applied here
(iterative algorithms model the forward operator internally).
"""

import numpy as np


def apply_bhc(chunk, bhc_coeffs):
    """Apply the BHC polynomial: p_corrected = c1*p + c2*p^2 + ...

    Works on numpy arrays AND torch tensors (uses only ``*``/``+``/``**``),
    so FDK's fused GPU path and the chunked numpy path share one definition.
    Returns ``chunk`` unchanged when ``bhc_coeffs`` is None.
    """
    if bhc_coeffs is None:
        return chunk
    result = bhc_coeffs[0] * chunk
    for k in range(1, len(bhc_coeffs)):
        result = result + bhc_coeffs[k] * chunk ** (k + 1)
    return result


# Backwards-compatible private alias (pre-refactor name).
_apply_bhc = apply_bhc


def ring_artifact_correction(sinogram, median_width=51, verbose=True):
    """Sinogram-space ring artifact correction, in place.

    Removes fixed-pattern detector column offsets (the cause of concentric
    ring artifacts): the angle-mean sinogram is median-filtered along
    detector columns, and the residual fixed pattern is subtracted from
    every projection.

    Args:
        sinogram: np.ndarray (N_angles, N_b, N_a) of line integrals,
            modified IN PLACE.
        median_width: median filter width along detector columns (odd int).

    Returns:
        The same ``sinogram`` array (for chaining).
    """
    from scipy.ndimage import median_filter
    if verbose:
        print("  Ring correction: computing column profile...")
    mean_sino = sinogram.mean(axis=0)
    smoothed = median_filter(mean_sino, size=(1, median_width))
    ring_artifact = mean_sino - smoothed
    sinogram -= ring_artifact[np.newaxis, :, :]
    if verbose:
        print(f"  Ring correction applied: max correction = "
              f"{np.abs(ring_artifact).max():.6f}")
    return sinogram


def downsample_projections(arr, factor):
    """Average-pool the last two (detector) axes by an integer factor.

    Accepts a 3-D projection stack (N_angles, N_b, N_a) or a single 2-D
    field (N_b, N_a) — bright/dark fields pool with the same code path.
    Trims trailing rows/columns so both detector axes are divisible by the
    factor. Returns float32.
    """
    factor = int(factor)
    if factor <= 1:
        return arr

    if arr.ndim == 2:
        N_b, N_a = arr.shape
        N_b_new, N_a_new = (N_b // factor) * factor, (N_a // factor) * factor
        trimmed = np.asarray(arr[:N_b_new, :N_a_new], dtype=np.float32)
        return trimmed.reshape(
            N_b_new // factor, factor, N_a_new // factor, factor
        ).mean(axis=(1, 3))

    N_angles, N_b, N_a = arr.shape
    N_b_new, N_a_new = (N_b // factor) * factor, (N_a // factor) * factor
    trimmed = np.array(arr[:, :N_b_new, :N_a_new], dtype=np.float32)
    return trimmed.reshape(
        N_angles, N_b_new // factor, factor, N_a_new // factor, factor
    ).mean(axis=(2, 4))


def preprocess_sinogram(projections, bright_field, dark_field,
                        clamp_mode='none', soft_clip_transmission=True,
                        soft_clip_sharpness=50.0, upper_clamp=True,
                        upper_clamp_value=1.05, chunk_angles=20,
                        bhc_coeffs=None,
                        ring_correction=False, ring_median_width=51):
    """
    Apply flat-field correction, log transform, BHC, and ring correction.

    Processes in chunks along the angle dimension to limit peak memory.

    Args:
        projections: Raw projections, shape (N_angles, N_b, N_a)
        bright_field: Unattenuated beam reference (I_0), shape (N_b, N_a).
            If None, projections are returned as-is (assumed pre-processed).
        dark_field: Electronic noise reference, shape (N_b, N_a).
            If None, projections are returned as-is (assumed pre-processed).
        clamp_mode: Line integral clamping mode ('none', 'soft', 'hard')
        soft_clip_transmission: Use soft clipping for transmission floor
        soft_clip_sharpness: Sharpness of soft clip transition
        upper_clamp: Clamp transmission from above
        upper_clamp_value: Maximum allowed transmission value
        chunk_angles: Number of projection angles per chunk (default 20)
        bhc_coeffs: BHC polynomial coefficients [c1, c2, ...] or None.
            Applied after log transform: p_corrected = c1*p + c2*p^2 + ...
        ring_correction: Apply sinogram-space ring artifact correction
        ring_median_width: Median filter width for ring correction (odd int)

    Returns:
        np.ndarray of line integrals, shape (N_angles, N_b, N_a), float32
    """
    if bright_field is None or dark_field is None:
        print("No bright/dark fields — using projections as-is (assumed pre-processed)")
        return np.array(projections, dtype=np.float32)

    print("Preprocessing: flat-field + log", end="")
    if bhc_coeffs is not None:
        coeff_str = ", ".join(f"c{k+1}={c:.6f}" for k, c in enumerate(bhc_coeffs))
        print(f" + BHC ({coeff_str})", end="")
    if ring_correction:
        print(f" + ring correction (width={ring_median_width})", end="")
    print(" ...")

    N_angles, N_b, N_a = projections.shape

    # Pre-compute denominator once (2D, small)
    denominator = (bright_field.astype(np.float64)
                   - dark_field.astype(np.float64))
    denominator = np.clip(denominator, 1.0, None)

    epsilon = 1e-6
    sharpness = soft_clip_sharpness
    upper_val = upper_clamp_value

    # Pre-allocate output
    sinogram = np.empty((N_angles, N_b, N_a), dtype=np.float32)

    bytes_per_proj = N_b * N_a * 4
    chunk_mem = chunk_angles * bytes_per_proj * 3
    print(f"  Chunk size: {chunk_angles} angles, "
          f"~{chunk_mem / 2**30:.2f} GiB peak per chunk")

    for start in range(0, N_angles, chunk_angles):
        end = min(start + chunk_angles, N_angles)

        # Flat-field correction: T = (I - dark) / (bright - dark)
        chunk = np.array(projections[start:end], dtype=np.float32)
        chunk -= dark_field
        chunk /= (denominator + epsilon)

        # Soft-clip transmission floor (prevents log(0) and Gibbs ringing)
        if soft_clip_transmission:
            scaled = sharpness * (chunk - epsilon)
            chunk = np.where(
                scaled > 20,
                chunk,
                epsilon + np.log1p(np.exp(np.clip(scaled, -700, 20))) / sharpness,
            ).astype(np.float32)
            del scaled

            if upper_clamp:
                scaled = sharpness * (upper_val - chunk)
                chunk = np.where(
                    scaled > 20,
                    chunk,
                    upper_val - np.log1p(np.exp(np.clip(scaled, -700, 20))) / sharpness,
                ).astype(np.float32)
                del scaled
        else:
            if upper_clamp:
                np.clip(chunk, epsilon, upper_val, out=chunk)
            else:
                np.clip(chunk, epsilon, None, out=chunk)

        # Log transform: p = -log(T)
        np.log(chunk, out=chunk)
        chunk *= -1

        # Optional line-integral clamping
        if clamp_mode == "soft":
            scaled = 50.0 * chunk
            chunk = np.where(
                scaled > 20,
                chunk,
                np.log1p(np.exp(np.clip(scaled, -700, 20))) / 50.0,
            ).astype(np.float32)
            del scaled
        elif clamp_mode == "hard":
            np.maximum(chunk, 0.0, out=chunk)

        # Beam hardening correction
        chunk = apply_bhc(chunk, bhc_coeffs)

        sinogram[start:end] = chunk

    # Ring correction (needs full sinogram — applied after chunked loop)
    if ring_correction:
        ring_artifact_correction(sinogram, median_width=ring_median_width)

    print(f"  Sinogram range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")
    return sinogram
