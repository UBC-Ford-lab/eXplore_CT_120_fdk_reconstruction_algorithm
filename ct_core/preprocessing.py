"""
Shared sinogram preprocessing: flat-field correction + log transform.

Extracted from astra_iterative.py to share between ASTRA and TIGRE backends.
Both iterative backends need identical preprocessing (flat-field → transmission
→ log transform) but skip cone-beam weighting and ramp filtering (those are
FDK-specific; iterative algorithms model the forward operator internally).
"""

import numpy as np


def preprocess_sinogram(projections, bright_field, dark_field,
                        clamp_mode='none', soft_clip_transmission=True,
                        soft_clip_sharpness=50.0, upper_clamp=True,
                        upper_clamp_value=1.05, chunk_angles=20):
    """
    Apply flat-field correction and log transform to raw projections.

    Processes in chunks along the angle dimension to avoid creating
    multiple full-sized intermediate arrays (which can easily exceed
    system RAM for large detectors). Peak memory is approximately:
        output_sinogram + chunk_angles * N_b * N_a * ~3 intermediates

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
        chunk_angles: Number of projection angles per chunk (default 20).
            Smaller = less RAM, larger = slightly faster.

    Returns:
        np.ndarray of line integrals, shape (N_angles, N_b, N_a), float32
    """
    if bright_field is None or dark_field is None:
        print("No bright/dark fields — using projections as-is (assumed pre-processed)")
        return np.array(projections, dtype=np.float32)

    print("Preprocessing: flat-field correction + log transform (chunked)...")

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
    chunk_mem = chunk_angles * bytes_per_proj * 3  # ~3 intermediates per chunk
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

        sinogram[start:end] = chunk

    print(f"  Sinogram range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")
    return sinogram
