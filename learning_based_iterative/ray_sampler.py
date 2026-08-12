"""Ray generation from cone-beam CT geometry.

Geometry convention (matches FDK in `reconstruction/fdk.py`):

    * Rotation about the z-axis. At theta = 0 the source sits on -x at
      distance R_s from the isocenter; the detector center sits on +x at
      distance R_d. This is the GEHC / FDK convention — see
      `reconstruction/fdk.py:909` where U = R_s + x·cos β + y·sin β
      derives directly from source at -R_s·(cos β, sin β, 0).
    * The detector "a" axis (column index, fast axis in projections) is
      tangent to the source orbit: u_hat = (-sin theta, cos theta, 0).
    * The detector "b" axis (row index) is +z: v_hat = (0, 0, 1).
    * `central_pixel_a` / `central_pixel_b` are the detector indices the
      source-isocenter line passes through. This definition is
      sign-unambiguous: at a = central_pixel_a the pixel sits exactly on
      det_center, which lies on the source->isocentre line by construction.
      CORRECTED 2026-08-11: this docstring used to claim the GEHC acquisition
      software pre-centres the columns so the iso ray lands at the geometric
      centre (N_a-1)/2. It does not. ct_recon's FDK reversed the same belief in
      commit c6c5b57 (2026-07-13) after empirical verification, noting that the
      geometric centre "reconstructs off-isocentre objects non-round".
      dataset._build_geometry now uses the scan.xml CoR converted to the
      sinogram's downsampled index space (geometry.cor_mode: xml).

Pixel position at (theta, b, a):
    p_xyz = +R_d * (cos t, sin t, 0)
            + (a - central_pixel_a) * da * u_hat
            + (b - central_pixel_b) * db * v_hat

Ray = source -> pixel, normalized.

Why we landed here: an earlier convention (source at +R_s·source_dir,
cpa mirrored, u-offset negated) appeared to validate on Scan_1988 because
the phantom is rotationally symmetric — a 180° rotation around z leaves
concentric rings invariant. Scan_1510 (off-isocenter mouse leg) exposed
the bug as a persistent ~5 mm offset that didn't shrink with iterations
or capacity. Aligning with FDK's own convention removes the offset and
keeps the phantom recon visually identical.
"""

from __future__ import annotations

import math

import torch

from .scene import Scene


def _detector_basis(angles: torch.Tensor):
    """Per-angle source/detector axes. Returns (source_dir, u_hat, v_hat),
    each shape (..., 3). source_pos = -R_s * source_dir (FDK convention:
    source on the -ve side); det_center = +R_d * source_dir."""
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    zero = torch.zeros_like(cos_t)
    one = torch.ones_like(cos_t)

    source_dir = torch.stack([cos_t, sin_t, zero], dim=-1)
    u_hat = torch.stack([-sin_t, cos_t, zero], dim=-1)
    v_hat = torch.stack([zero, zero, one], dim=-1)
    return source_dir, u_hat, v_hat


def rays_from_indices(
    scene: Scene,
    angle_idx: torch.Tensor,
    b_idx: torch.Tensor,
    a_idx: torch.Tensor,
):
    """Construct rays from index tensors (all 1-D, length N).

    Returns
    -------
    origins : (N, 3) ray origins in mm
    directions : (N, 3) unit-length ray directions
    target : (N,) line integrals at scene.sinogram[angle_idx, b_idx, a_idx]
    """
    geom = scene.geometry
    R_s = float(geom["R_s"])
    R_d = float(geom["R_d"])
    da = float(geom["da"])
    db = float(geom["db"])
    cpa = float(geom["central_pixel_a"])
    cpb = float(geom["central_pixel_b"])

    theta = scene.angles.to(angle_idx.device)[angle_idx]
    source_dir, u_hat, v_hat = _detector_basis(theta)

    # Measured in-plane detector rotation (about the detector normal). See
    # dataset._build_geometry: Scan_1510's joint calibration gives -0.652 deg.
    # Absent / 0.0 => exact no-op, so existing configs are unchanged.
    _psi = float(geom.get("det_psi_rad", 0.0) or 0.0)
    if _psi:
        _c, _s = math.cos(_psi), math.sin(_psi)
        u_hat, v_hat = _c * u_hat + _s * v_hat, -_s * u_hat + _c * v_hat

    # FDK convention: source on the -ve side of the rotation axis,
    # detector on the +ve side. See `reconstruction/fdk.py:909` for the
    # corresponding U = R_s + x·cos β + y·sin β formula that fixes this
    # sign. An earlier muNeRF convention had source at +R_s·source_dir;
    # combined with cpa-mirror + u-axis-negation it appeared to work on
    # the Scan_1988 phantom but produced a 180°-rotated reconstruction
    # invisible under rotational symmetry. Scan_1510 (off-iso leg) made
    # the bug visible.
    origins = -R_s * source_dir
    det_center = R_d * source_dir

    # Per-pixel detector distortion. The raw detector grid is not the ideal
    # linear grid the equations above assume, so map each raw sample index to
    # its IDEAL detector coordinate before converting to a physical offset.
    # Applied here — the single point where every ray in the pipeline is
    # built — so training, rendering and metrics stay geometrically
    # consistent. Absent => exact no-op. See inr_pipeline/detector_warp.py.
    _warp = geom.get("detector_warp")
    a_i, b_i = a_idx.to(theta), b_idx.to(theta)
    if _warp is not None:
        b_i, a_i = _warp.ideal_indices(b_i, a_i,
                                       downsample=int(geom.get("sinogram_downsample", 1)))
        b_i, a_i = b_i.to(theta), a_i.to(theta)

    a_off = (a_i - cpa) * da
    b_off = (b_i - cpb) * db

    pixel_xyz = (
        det_center
        + a_off.unsqueeze(-1) * u_hat
        + b_off.unsqueeze(-1) * v_hat
    )
    directions = pixel_xyz - origins
    directions = directions / directions.norm(dim=-1, keepdim=True)

    target = scene.sinogram.to(angle_idx.device)[angle_idx, b_idx, a_idx]
    return origins, directions, target


def _sample_angle_idx(n_angles, shape, generator, device, exclude_angle):
    """Random angle indices in [0, n_angles), optionally excluding one angle.

    Held-out early stopping reserves one projection as a validation signal, so
    it must never be sampled for training. We draw from n_angles-1 buckets and
    shift indices at/above the excluded angle up by one — a uniform draw over
    the remaining angles with no rejection loop.
    """
    if exclude_angle is None or n_angles <= 1:
        return torch.randint(n_angles, shape, generator=generator, device=device)
    idx = torch.randint(n_angles - 1, shape, generator=generator, device=device)
    return idx + (idx >= int(exclude_angle)).long()


def sample_random_rays(
    scene: Scene,
    n: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    exclude_angle: int | None = None,
):
    """Sample n rays uniformly across (angle, row, col).

    The sinogram-side cost is a single vectorized advanced-index gather
    (`sino[a, b, c]` for n indices) executed in PyTorch C++; for n=1024 random
    indices into a 7 GB tensor this is well under 1 ms on CPU. Building the n
    rays themselves (basis vectors + pixel positions + normalization) is a
    handful of (n, 3)-shape ops — also negligible.
    Per-iter overhead is dominated by the renderer + MLP, not this function.

    ``exclude_angle`` (if set) is never sampled — the held-out validation angle.
    """
    device = device if device is not None else scene.sinogram.device
    n_angles = scene.n_angles
    n_b, n_a = scene.detector_shape

    angle_idx = _sample_angle_idx(n_angles, (n,), generator, device, exclude_angle)
    b_idx = torch.randint(n_b, (n,), generator=generator, device=device)
    a_idx = torch.randint(n_a, (n,), generator=generator, device=device)
    return rays_from_indices(scene, angle_idx, b_idx, a_idx)


def sample_random_rows(
    scene: Scene,
    n_rows: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    exclude_angle: int | None = None,
):
    """Sample n_rows full detector rows, each from a random (angle, b).

    Returns rays for all pixels in those rows so the caller can apply a 1-D
    ramp filter along the column axis before computing the loss.

    ``exclude_angle`` (if set) is never sampled — the held-out validation angle.

    Returns
    -------
    origins :    (n_rows * n_a, 3)
    directions : (n_rows * n_a, 3)
    target :     (n_rows, n_a)  — 2-D so the ramp filter can operate row-wise
    """
    device = device if device is not None else scene.sinogram.device
    n_angles = scene.n_angles
    n_b, n_a = scene.detector_shape

    angle_idx = _sample_angle_idx(n_angles, (n_rows,), generator, device, exclude_angle)
    b_idx = torch.randint(n_b, (n_rows,), generator=generator, device=device)

    # Expand each row: same (angle, b) for all n_a columns
    a_all = torch.arange(n_a, device=device)
    # (n_rows, n_a) index grids
    angle_exp = angle_idx.unsqueeze(1).expand(-1, n_a).reshape(-1)
    b_exp = b_idx.unsqueeze(1).expand(-1, n_a).reshape(-1)
    a_exp = a_all.unsqueeze(0).expand(n_rows, -1).reshape(-1)

    origins, directions, target_flat = rays_from_indices(scene, angle_exp, b_exp, a_exp)
    target = target_flat.reshape(n_rows, n_a)
    return origins, directions, target


def sample_projection_patch(
    scene: Scene,
    patch_h: int,
    patch_w: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    exclude_angle: int | None = None,
    num_patches: int = 1,
):
    """Sample ``num_patches`` contiguous 2-D projection patches.

    Each patch is a random (angle, b0:b0+patch_h, a0:a0+patch_w) block of ONE
    projection — a spatially coherent 2-D region, so a windowed SSIM/MS-SSIM can
    operate on it (random rays/rows are NOT coherent and can't be used for SSIM).
    Across iters the random (angle, location) covers all angles → multi-angle
    consistency denoises the structural signal.

    ``exclude_angle`` (if set) is never sampled — the held-out validation angle.

    Returns
    -------
    origins, directions : (num_patches * patch_h * patch_w, 3)
    target :              (num_patches, patch_h, patch_w)
    """
    device = device if device is not None else scene.sinogram.device
    n_angles = scene.n_angles
    n_b, n_a = scene.detector_shape
    ph = min(int(patch_h), n_b)
    pw = min(int(patch_w), n_a)

    angle_idx = _sample_angle_idx(n_angles, (num_patches,), generator, device, exclude_angle)
    b0 = torch.randint(n_b - ph + 1, (num_patches,), generator=generator, device=device)
    a0 = torch.randint(n_a - pw + 1, (num_patches,), generator=generator, device=device)

    b_off = torch.arange(ph, device=device).view(1, -1, 1)
    a_off = torch.arange(pw, device=device).view(1, 1, -1)
    bb = (b0.view(-1, 1, 1) + b_off).expand(num_patches, ph, pw)
    aa = (a0.view(-1, 1, 1) + a_off).expand(num_patches, ph, pw)
    ang = angle_idx.view(-1, 1, 1).expand(num_patches, ph, pw)

    origins, directions, target_flat = rays_from_indices(
        scene, ang.reshape(-1), bb.reshape(-1), aa.reshape(-1))
    target = target_flat.reshape(num_patches, ph, pw)
    return origins, directions, target


def rays_for_projection(
    scene: Scene,
    angle_idx: int,
    device: torch.device | str | None = None,
):
    """All rays for one projection angle. Returns flattened (N_b * N_a, ...)."""
    device = device if device is not None else scene.sinogram.device
    n_b, n_a = scene.detector_shape
    bb, aa = torch.meshgrid(
        torch.arange(n_b, device=device),
        torch.arange(n_a, device=device),
        indexing="ij",
    )
    angle = torch.full((n_b * n_a,), int(angle_idx), dtype=torch.long, device=device)
    return rays_from_indices(scene, angle, bb.reshape(-1), aa.reshape(-1))
