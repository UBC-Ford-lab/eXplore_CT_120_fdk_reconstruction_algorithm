# Script created by Falk Wiegmann in Feb 2025 to simulate a 3D cone beam CT scan reconstruction
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ..ct_core.calibration import default_mu_water, mu_to_hu
from ..ct_core.preprocessing import (air_normalize_sinogram, apply_bhc,
                                     ring_artifact_correction)

# Device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_FILTER_TYPES = ('ramp', 'shepp-logan', 'cosine', 'hamming')


def _build_filter_kernel(N_a, da, filter_cutoff, filter_type,
                         physical_normalization, device):
    """Build a windowed ramp filter kernel in the frequency domain.

    All kernels are |f| × W(f), zeroed beyond f_cutoff.

    Supported windows:
        ramp        – 1 (no window, hard cutoff at f_cutoff)
        shepp-logan – sinc(f / (2·f_cutoff))
        cosine      – cos(π·f / (2·f_cutoff))
        hamming     – 0.54 + 0.46·cos(π·f / f_cutoff)

    Returns:
        (kernel, f_cutoff) where kernel has shape (1, N_a//2+1).
    """
    freqs = torch.fft.rfftfreq(N_a, d=da).to(device)
    f_cutoff = filter_cutoff * freqs.abs().max()
    ramp = torch.abs(freqs)
    mask = (freqs.abs() <= f_cutoff).float()

    if filter_type == 'ramp':
        window = torch.ones_like(freqs)
    elif filter_type == 'shepp-logan':
        window = torch.sinc(freqs / (2 * f_cutoff))   # sinc(x) = sin(πx)/(πx)
    elif filter_type == 'cosine':
        window = torch.cos(torch.pi * freqs / (2 * f_cutoff))
    elif filter_type == 'hamming':
        window = 0.54 + 0.46 * torch.cos(torch.pi * freqs / f_cutoff)
    else:
        raise ValueError(
            f"Unknown filter_type '{filter_type}'. "
            f"Supported: {SUPPORTED_FILTER_TYPES}"
        )

    kernel = (ramp * window * mask).unsqueeze(0).clamp(min=0)  # (1, N_a//2+1)
    if not physical_normalization:
        kernel = kernel / kernel.max()
    return kernel, f_cutoff

def _interpolate_metal_pixels(sinogram_chunk, mask):
    """Replace metal-corrupted sinogram pixels by linear interpolation along detector rows.

    For each detector row at each projection angle, connected runs of masked
    (metal-affected) pixels are replaced by linearly interpolating between
    the nearest unmasked neighbours on each side.  If a run extends to the
    edge of the detector (no neighbour on one side), constant extrapolation
    from the other side is used.

    Operates in-place on sinogram_chunk to avoid doubling GPU memory.

    Args:
        sinogram_chunk: Tensor of shape (chunk_len, N_b, N_a) — line integrals.
        mask: Boolean tensor of same shape — True where metal-corrupted.
    """
    chunk_np = sinogram_chunk.cpu().numpy()
    mask_np = mask.cpu().numpy()
    n_interp_total = 0

    for i in range(chunk_np.shape[0]):          # projection angle
        for j in range(chunk_np.shape[1]):      # detector row
            row_mask = mask_np[i, j]            # (N_a,)
            if not row_mask.any():
                continue

            row = chunk_np[i, j]                # (N_a,)
            N_a = len(row)

            # Walk through connected runs of masked pixels
            k = 0
            while k < N_a:
                if not row_mask[k]:
                    k += 1
                    continue

                # Start of a masked run
                run_start = k
                while k < N_a and row_mask[k]:
                    k += 1
                run_end = k  # one past last masked pixel

                n_interp_total += run_end - run_start

                # Find boundary values
                left_val = row[run_start - 1] if run_start > 0 else None
                right_val = row[run_end] if run_end < N_a else None

                if left_val is not None and right_val is not None:
                    # Linear interpolation between boundaries
                    span = run_end - run_start + 2  # include both boundaries
                    interp = np.linspace(left_val, right_val, span)
                    row[run_start:run_end] = interp[1:-1]
                elif left_val is not None:
                    # Run extends to right edge — constant extrapolation
                    row[run_start:run_end] = left_val
                elif right_val is not None:
                    # Run extends to left edge — constant extrapolation
                    row[run_start:run_end] = right_val
                # else: entire row is masked — leave unchanged

    if n_interp_total > 0:
        print(f"    MAR: interpolated {n_interp_total} pixels "
              f"across {chunk_np.shape[0]} projections")

    # Copy corrected data back into the existing GPU tensor (no new allocation)
    sinogram_chunk.copy_(torch.from_numpy(chunk_np))


class FDKReconstructor:
    def __init__(self, projections, angles, geometry, source_locations, folder_name,
                 mu_water=None, output_hu=False,
                 bright_field=None, dark_field=None,
                 clamp_mode="none", soft_clip_transmission=True,
                 soft_clip_sharpness=50.0, upper_clamp=True, upper_clamp_value=1.05,
                 physical_normalization=False, filter_cutoff=1.0,
                 filter_type='cosine', parker_weighting=True,
                 metal_artifact_reduction=False, mar_threshold=6.0,
                 ring_correction=False, ring_median_width=51,
                 air_normalization=True,
                 bhc_coeffs=None,
                 bone_bhc=False, bone_bhc_threshold=1500, bone_bhc_hu=3100):
        """
        projections: Tensor of shape (N_angles, N_b, N_a) in float32.
        angles: Tensor of shape (N_angles,) in radians.
        geometry: dictionary with keys:
           - R_s: source-to-isocenter distance (in mm)
           - R_d: detector-to-isocenter distance (in mm)
           - da: detector pixel size in horizontal direction (mm)
           - db: detector pixel size in vertical direction (mm)
           - vol_shape: tuple (Nx, Ny, Nz) for the reconstruction volume (number of voxels)
           - vol_origin: (x, y, z) volume center in mm
           - dx: voxel size in xy (mm)
           - dz: voxel size in z (mm)
           - central_pixel_a: detector center column
           - central_pixel_b: detector center row
        source_locations: list of source locations in the form [(x1, y1, z1), (x2, y2, z2), ...]
        mu_water: float, linear attenuation coefficient of water in mm⁻¹ for HU conversion
        output_hu: bool, if True, convert output to Hounsfield Units
        bright_field: np.ndarray, unattenuated beam reference (I₀) for flat-field correction [height, width]
        dark_field: np.ndarray, electronic noise reference for flat-field correction [height, width]
        clamp_mode: str, line integral clamping mode ("none", "soft", "hard")
            - "none": No clamping, preserves noise, no Gibbs ringing (recommended)
            - "soft": Softplus smooth clamp, minimal ringing
            - "hard": np.maximum hard clamp, causes Gibbs ringing (not recommended)
        soft_clip_transmission: bool, if True, use soft clipping for transmission floor (default: True)
            - True: Smooth transition at epsilon, prevents center ringing from saturated pixels
            - False: Hard clip at epsilon (legacy behavior, causes Gibbs ringing)
        soft_clip_sharpness: float, sharpness of soft clip transition (default: 50.0)
            - Lower values = broader transition = less center ringing
            - 50.0 gives ~0.06 transition width (affects T < 0.06)
            - 1000.0 gives ~0.003 transition width (effectively hard clip)
        upper_clamp: bool, if True, also clamp transmission from above (default: True)
            - Prevents negative line integrals from T > 1 (noise in air regions)
        upper_clamp_value: float, maximum allowed transmission (default: 1.05)
            - Values slightly > 1.0 allowed to preserve noise characteristics
        physical_normalization: bool, if True, keep physical units in ramp filter (no max-normalization)
            and apply FDK 1/2 prefactor so output is true μ (mm⁻¹). HU conversion then uses
            literature μ_water directly instead of empirical percentile calibration. (default: False)
        filter_cutoff: float, ramp filter bandwidth as fraction of Nyquist (0.0–1.0, default: 1.0).
            Lower values reduce noise at the cost of spatial resolution.
        filter_type: str, ramp filter window type (default: 'cosine').
            Supported: 'ramp' (Ram-Lak), 'shepp-logan', 'cosine', 'hamming'.
        parker_weighting: bool, if True, apply Parker (short-scan) redundancy weighting
            for scans covering less than 360°. Automatically skipped for full-circle scans.
            Corrects intensity shading artifacts caused by double-counted rays in short scans.
            (default: True)
        metal_artifact_reduction: bool, if True, detect and interpolate metal-corrupted
            sinogram pixels before cone-beam weighting/ramp filtering. Reduces dark
            streak artifacts behind highly attenuating objects (metal, dense bone).
            (default: False)
        mar_threshold: float, line integral threshold for metal pixel detection.
            Pixels with p = -log(T) > threshold are considered metal-corrupted.
            Typical values: 4.0 (aggressive), 6.0 (default), 8.0 (conservative).
            (default: 6.0)
        ring_correction: bool, if True, apply sinogram-space ring artifact correction
            after log transform and before cone-beam weighting. Removes fixed-pattern
            detector column offsets that cause concentric ring artifacts.
            (default: False)
        ring_median_width: int, median filter kernel width for ring correction (must be odd).
            Controls the spatial scale of features preserved in the column profile.
            Larger values remove broader ring features but risk removing real structure.
            (default: 51)
        bhc_coeffs: list/tuple of BHC polynomial coefficients [c1, c2, ...], or None.
            Applied in sinogram domain after log transform:
            p_corrected = c1*p + c2*p^2 [+ c3*p^3 + ...]
            Corrects beam hardening (scale error + cupping).
            Calibrate from water phantom using bhc_calibration.py.
            (default: None = no BHC)
        """
        self.projections = projections # (N_angles, N_b, N_a)
        self.angles = angles.to(device)
        self.R_s = geometry["R_s"]
        self.R_d = geometry["R_d"]
        self.SDD = self.R_s + self.R_d
        self.da = geometry["da"]
        self.db = geometry["db"]
        self.vol_shape = geometry["vol_shape"] # (Nx, Ny, Nz)
        self.vol_origin = geometry["vol_origin"] # (x, y, z) in mm
        self.dx = geometry["dx"] # voxel size in mm
        self.dz = geometry["dz"] # voxel size in mm
        self.central_pixel_a = geometry["central_pixel_a"]
        self.central_pixel_b = geometry["central_pixel_b"]
        # Scale for applying the detector COR/central-slice offset in backprojection.
        # 1.0 = apply the verified-correct offset (default). 0.0 = off (legacy);
        # -1.0 = flipped sign (diagnostic). See note in backprojection().
        self.cor_offset_scale = geometry.get("cor_offset_scale", 1.0)
        # Detector in-plane rotation (radians). 0.0 is a BIT-EXACT no-op: the
        # backprojection keeps its original z-independent `a_2d` factorisation
        # and never allocates the 3-D a-coordinate. Before 2026-08-11 FDK had no
        # psi term at all, so 0.0 reproduces every historical reconstruction.
        self.det_psi_rad = float(geometry.get("det_psi_rad", 0.0) or 0.0)
        self.source_locations = source_locations
        self.folder_name = folder_name

        # HU calibration parameters
        self.mu_water = default_mu_water(mu_water, bhc_coeffs)
        self.output_hu = output_hu
        self.bright_field = bright_field
        self.dark_field = dark_field
        self.clamp_mode = clamp_mode
        self.soft_clip_transmission = soft_clip_transmission
        self.soft_clip_sharpness = soft_clip_sharpness
        self.upper_clamp = upper_clamp
        self.upper_clamp_value = upper_clamp_value
        self.physical_normalization = physical_normalization
        self.filter_cutoff = filter_cutoff
        self.filter_type = filter_type
        self.parker_weighting = parker_weighting
        self.metal_artifact_reduction = metal_artifact_reduction
        self.mar_threshold = mar_threshold
        self.ring_correction = ring_correction
        self.air_normalization = air_normalization
        self.ring_median_width = ring_median_width
        self.bhc_coeffs = bhc_coeffs
        self.bone_bhc = bone_bhc
        self.bone_bhc_threshold = bone_bhc_threshold
        self.bone_bhc_hu = bone_bhc_hu
        if self.bone_bhc and not self.physical_normalization:
            raise ValueError(
                "bone_bhc requires physical_normalization=True — "
                "V1 must be in mu (mm^-1) for bone segmentation"
            )

        # Determine detector dimensions and center indices
        self.N_angles, self.N_b, self.N_a = self.projections.shape
        self.a_center = (self.N_a - 1) / 2.0
        self.a_length = self.da * self.N_a
        self.b_center = (self.N_b - 1) / 2.0
        self.b_length = self.db * self.N_b

    def _apply_bhc(self, chunk):
        """Apply BHC polynomial to line integrals (shared ct_core definition)."""
        return apply_bhc(chunk, self.bhc_coeffs)

    def _forward_project(self, volume, mask=None):
        """Voxel-driven GPU forward projection via nearest-neighbor splatting.

        Projects nonzero voxels onto the detector using the same cone-beam
        geometry as the backprojection.  Each voxel contributes value * dx
        (line integral weight) to its nearest detector pixel via scatter_add_.

        Args:
            volume: Tensor or ndarray (Nx, Ny, Nz) — values to project.
            mask: Bool tensor/ndarray (Nx, Ny, Nz) — only project where True.
                  If None, projects all nonzero voxels.

        Returns:
            sinogram: ndarray (N_angles, N_b, N_a) — projected line integrals.
        """
        Nx, Ny, Nz = self.vol_shape
        sinogram = np.zeros((self.N_angles, self.N_b, self.N_a), dtype=np.float32)

        # Coordinate vectors (same as backprojection)
        x = (torch.arange(Nx, device=device, dtype=torch.float32) - (Nx - 1) / 2) * self.dx + self.vol_origin[0]
        y = (torch.arange(Ny, device=device, dtype=torch.float32) - (Ny - 1) / 2) * self.dx + self.vol_origin[1]
        z = (torch.arange(Nz, device=device, dtype=torch.float32) - (Nz - 1) / 2) * self.dz + self.vol_origin[2]

        # Convert inputs to numpy
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy() if volume.is_cuda else volume.numpy()
        else:
            volume_np = np.asarray(volume, dtype=np.float32)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
            else:
                mask_np = np.asarray(mask, dtype=bool)
        else:
            mask_np = volume_np != 0

        # Extract nonzero voxel indices and values
        nz_ix, nz_iy, nz_iz = np.nonzero(mask_np)
        n_voxels = len(nz_ix)
        if n_voxels == 0:
            return sinogram

        print(f"  Forward projection: {n_voxels} voxels "
              f"({n_voxels / mask_np.size * 100:.1f}%)")

        # Load nonzero voxel coordinates and values onto GPU
        voxel_x = x[torch.from_numpy(nz_ix.astype(np.int64)).to(device)]
        voxel_y = y[torch.from_numpy(nz_iy.astype(np.int64)).to(device)]
        voxel_z = z[torch.from_numpy(nz_iz.astype(np.int64)).to(device)]
        voxel_val = torch.from_numpy(
            volume_np[nz_ix, nz_iy, nz_iz].astype(np.float32)
        ).to(device) * self.dx  # value * dx -> line integral contribution
        del nz_ix, nz_iy, nz_iz

        cos_beta = torch.cos(self.angles)
        sin_beta = torch.sin(self.angles)
        sino_size = self.N_b * self.N_a

        # Chunk voxels for intermediate tensors (U, ratio, a_pix, b_pix, valid, flat_idx)
        budget = self._gpu_free_bytes()
        bytes_per_voxel_tmp = 6 * 4  # 6 intermediate float32/int64 tensors
        voxel_chunk = int(budget * 0.3 / bytes_per_voxel_tmp)
        voxel_chunk = max(10000, min(voxel_chunk, n_voxels))

        for i in range(self.N_angles):
            cb = cos_beta[i]
            sb = sin_beta[i]
            sino_flat = torch.zeros(sino_size, device=device, dtype=torch.float32)

            for vs in range(0, n_voxels, voxel_chunk):
                ve = min(vs + voxel_chunk, n_voxels)
                vx = voxel_x[vs:ve]
                vy = voxel_y[vs:ve]
                vz = voxel_z[vs:ve]
                vv = voxel_val[vs:ve]

                # Source-to-voxel distance along central ray
                U = self.R_s + vx * cb + vy * sb

                # Detector pixel coordinates (nearest-neighbor)
                ratio = self.SDD / U
                a_pix = torch.round(ratio * (-vx * sb + vy * cb) / self.da + self.N_a / 2.0).long()
                b_pix = torch.round(ratio * vz / self.db + self.N_b / 2.0).long()

                # Mask valid pixels (within detector bounds)
                valid = (a_pix >= 0) & (a_pix < self.N_a) & (b_pix >= 0) & (b_pix < self.N_b)
                if valid.any():
                    flat_idx = b_pix[valid] * self.N_a + a_pix[valid]
                    sino_flat.scatter_add_(0, flat_idx, vv[valid])

            sinogram[i] = sino_flat.view(self.N_b, self.N_a).cpu().numpy()

            if (i + 1) % 50 == 0 or i == self.N_angles - 1:
                print(f"    angle {i+1}/{self.N_angles}")

        del voxel_x, voxel_y, voxel_z, voxel_val
        torch.cuda.empty_cache()
        return sinogram

    def _apply_cone_weight_and_filter(self):
        """Apply cone-beam weighting, ramp filter, and Parker to the unfiltered sinogram.

        Reads self._unfiltered_sinogram (numpy), writes filtered result to
        self.projections (numpy).  Uses cached filter parameters from
        _preprocess_and_filter().
        """
        print("  Re-applying cone weight + ramp filter...")
        bytes_per_proj = self.N_b * self.N_a * 4
        budget = self._gpu_free_bytes()
        persistent = 2 * bytes_per_proj  # cone_weight + filter_kernel
        chunk_size = int((budget - persistent) // (6 * bytes_per_proj))
        chunk_size = max(1, min(chunk_size, self.N_angles))

        for start in range(0, self.N_angles, chunk_size):
            end = min(start + chunk_size, self.N_angles)
            chunk = torch.from_numpy(
                self._unfiltered_sinogram[start:end]
            ).to(device)

            chunk = chunk * self._cone_weight_gpu
            chunk = torch.fft.rfft(chunk, dim=2, norm='forward')
            chunk = chunk * self._filter_kernel_gpu
            chunk = torch.fft.irfft(chunk, n=self.N_a, dim=2, norm='forward')
            if self._parker_weight_gpu is not None:
                chunk = chunk * self._parker_weight_gpu[start:end].unsqueeze(1)

            self.projections[start:end] = chunk.cpu().numpy()

    def _bone_bhc_correction(self):
        """Joseph & Spital two-pass bone BHC correction.

        After pass-1 backprojection, segments bone from V1 (mu volume),
        computes correction = FP((mu_bone_mono - V1) * mask) in a single
        forward projection, and applies it to self._unfiltered_sinogram.

        Memory optimization: combines what would be two forward projections
        (FP(V1*mask) and FP(mask)) into one: FP((mu_bone_mono - V1)*mask),
        since mu_bone_mono*FP(mask) - FP(V1*mask) = FP((mu_bone_mono-V1)*mask).
        """
        mu_water = self.mu_water
        mu_thresh = mu_water * (1 + self.bone_bhc_threshold / 1000.0)
        mu_bone_mono = mu_water * (1 + self.bone_bhc_hu / 1000.0)

        print(f"  Bone segmentation threshold: {mu_thresh:.6f} mm^-1 ({self.bone_bhc_threshold} HU)")
        print(f"  Bone monochromatic mu: {mu_bone_mono:.6f} mm^-1 ({self.bone_bhc_hu} HU)")

        # Get volume as numpy
        vol = self.reconstructed_volume
        if isinstance(vol, torch.Tensor):
            vol = vol.cpu().numpy() if vol.is_cuda else vol.numpy()
        vol = np.asarray(vol, dtype=np.float32)

        # Segment bone
        bone_mask = vol > mu_thresh
        n_bone = int(bone_mask.sum())
        print(f"  Bone voxels: {n_bone} ({n_bone / bone_mask.size * 100:.2f}%)")

        if n_bone == 0:
            print("  No bone voxels found — skipping bone BHC correction")
            return

        # Single forward projection: FP((mu_bone_mono - V1) * mask)
        # = mu_bone_mono * FP(mask) - FP(V1 * mask) = correction
        print("  Forward-projecting correction volume (mu_mono - V1) * mask...")
        correction_input = mu_bone_mono - vol  # mu difference at each voxel
        sino_correction = self._forward_project(correction_input, mask=bone_mask)
        del correction_input, bone_mask

        nonzero = sino_correction != 0
        if nonzero.any():
            print(f"  Correction stats: mean={sino_correction[nonzero].mean():.6f}, "
                  f"|max|={np.abs(sino_correction).max():.6f}")

        # Apply correction to unfiltered sinogram
        self._unfiltered_sinogram += sino_correction
        del sino_correction

    def _flush_projections(self):
        """Flush projections to disk only when backed by a memmap."""
        if isinstance(self.projections, np.memmap):
            self.projections.flush()

    @staticmethod
    def _gpu_free_bytes(safety=0.85):
        """Return usable GPU memory in bytes (after safety margin).

        Calls empty_cache() to reclaim fragmented memory, then queries free VRAM.
        Returns float('inf') when running on CPU (no GPU constraint).
        """
        if not torch.cuda.is_available():
            return float('inf')
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        return int(free * safety)

    def _preload_projections(self, force_cpu=False):
        """Load all projections into a contiguous tensor, on GPU if budget allows."""
        proj_np = np.array(self.projections, dtype=np.float32)  # memmap -> contiguous RAM
        if force_cpu or device.type != 'cuda':
            return torch.from_numpy(proj_np)
        try:
            return torch.from_numpy(proj_np).to(device)
        except torch.cuda.OutOfMemoryError:
            print("  GPU OOM during projection preload — falling back to CPU")
            return torch.from_numpy(proj_np)

    def _compute_parker_weights(self):
        """Compute Parker (short-scan) redundancy weights for fan-beam geometry.

        Returns a weight matrix of shape (N_angles, N_a) that ensures each ray
        is counted exactly once, eliminating intensity shading from redundant
        measurements in short scans (π + 2γ_m < Λ < 2π).

        For full-circle scans (Λ ≥ 350°) returns None (no correction needed).
        """
        # Total angular range of the scan
        Lambda = float(self.angles[-1] - self.angles[0])

        # Skip for (near-)full circle scans: all weights would be ~1
        if Lambda >= np.deg2rad(350.0):
            return None

        epsilon = Lambda - np.pi  # scan excess over π
        if epsilon <= 0:
            # Scan is shorter than π — Parker weighting not applicable
            print(f"  Parker weighting: scan range {np.rad2deg(Lambda):.1f}° < 180°, skipping.")
            return None

        # Fan angle for each detector column
        col_idx = torch.arange(self.N_a, dtype=torch.float32)
        # Use detector center (VFF projections are already COR-centered)
        gamma = torch.arctan(((col_idx - self.N_a / 2) * self.da) / self.SDD)  # (N_a,)

        # Max half-fan angle
        gamma_m = float(torch.arctan(torch.tensor((self.N_a / 2) * self.da / self.SDD)))

        # Relative projection angle (from scan start)
        beta_rel = (self.angles - self.angles[0]).cpu().float()  # (N_angles,)

        # Broadcast: beta_rel (N_angles, 1) and gamma (1, N_a) -> (N_angles, N_a)
        beta = beta_rel.unsqueeze(1)  # (N_angles, 1)
        g = gamma.unsqueeze(0)        # (1, N_a)

        # Transition width — same d for ramp-up and ramp-down ensures
        # continuity at every boundary: sin²(π/2) = 1.
        d = epsilon - 2.0 * g          # (1, N_a)

        # Boundary between full-weight and ramp-down: β = π + 2γ
        boundary = np.pi + 2.0 * g     # (1, N_a)

        # Initialize weights to 1.0 (full weight)
        weights = torch.ones(self.N_angles, self.N_a, dtype=torch.float32)

        # Only columns with d > 0 have redundancy and need weighting
        has_redundancy = (d > 0)        # (1, N_a) broadcast to (N_angles, N_a)
        safe_d = torch.where(has_redundancy, d, torch.ones_like(d))

        # Region 1: Ramp-up at scan start — 0 ≤ β < d
        in_rampup = has_redundancy & (beta >= 0) & (beta < d)
        rampup_arg = torch.where(
            has_redundancy,
            (np.pi / 2.0) * beta / safe_d,
            torch.zeros_like(beta),
        )
        weights = torch.where(in_rampup, torch.sin(rampup_arg) ** 2, weights)

        # Region 2: Full weight — d ≤ β ≤ π + 2γ (already 1.0)

        # Region 3: Ramp-down at scan end — π + 2γ < β ≤ Λ
        in_rampdown = has_redundancy & (beta > boundary) & (beta <= Lambda)
        rampdown_arg = torch.where(
            has_redundancy,
            (np.pi / 2.0) * (Lambda - beta) / safe_d,
            torch.zeros_like(beta),
        )
        weights = torch.where(in_rampdown, torch.sin(rampdown_arg) ** 2, weights)

        # Clamp to [0, 1] for numerical safety
        weights = weights.clamp(0.0, 1.0)

        print(f"  Parker weighting: Λ={np.rad2deg(Lambda):.1f}°, "
              f"ε={np.rad2deg(epsilon):.1f}°, γ_m={np.rad2deg(gamma_m):.1f}°, "
              f"margin={np.rad2deg(epsilon - 2*gamma_m):.2f}°")
        print(f"  Weight range: [{float(weights.min()):.4f}, {float(weights.max()):.4f}], "
              f"mean={float(weights.mean()):.4f}")

        return weights

    def _preprocess_and_filter(self):
        """
        Fused preprocessing pipeline: flat-field + log + cone-weight + ramp-filter
        in a single GPU pass per chunk. Replaces sequential preprocess() → pre_weight()
        → ramp_filter() calls for ~2× fewer data transfers.

        When output_hu=True with bright/dark fields: applies full flat-field correction,
        transmission clamping, log transform, cone-beam weighting, and ramp filtering.

        When output_hu=False (or no bright/dark fields): applies only cone-beam weighting
        and ramp filtering on the raw projections.
        """
        do_preprocess = (self.output_hu and self.bright_field is not None
                         and self.dark_field is not None)

        # --- Pre-compute constants on GPU (before loop) ---

        # Cone-beam weight: w(a,b) = SDD / sqrt(SDD² + a² + b²)
        # Use detector center (VFF projections are already COR-centered)
        a_coords = (torch.arange(self.N_a, device=device) - self.N_a / 2) * self.da
        b_coords = (torch.arange(self.N_b, device=device) - self.N_b / 2) * self.db
        B, A = torch.meshgrid(b_coords, a_coords, indexing='ij')
        cone_weight = self.SDD / (torch.sqrt(self.SDD**2 + A**2 + B**2) + 1e-8)  # (N_b, N_a)

        # Ramp filter kernel (windowed)
        filter_kernel, f_cutoff = _build_filter_kernel(
            self.N_a, self.da, self.filter_cutoff, self.filter_type,
            self.physical_normalization, device,
        )
        print(f"Filter: {self.filter_type}, cutoff: {self.filter_cutoff:.2f} × f_Nyquist = {float(f_cutoff):.4f} mm⁻¹")
        if self.physical_normalization:
            print(f"Ramp filter κ (filter_kernel.max) = {float(filter_kernel.max()):.4f} mm⁻¹")
        if self.metal_artifact_reduction:
            print(f"Metal artifact reduction: enabled (threshold={self.mar_threshold:.1f})")
        if self.air_normalization:
            print("Air normalization: enabled (per-projection offset from "
                  "object-free columns)")
        if self.ring_correction:
            print(f"Ring correction: enabled (median width={self.ring_median_width})")
        if self.bhc_coeffs is not None:
            coeff_str = ", ".join(f"c{k+1}={c:.6f}" for k, c in enumerate(self.bhc_coeffs))
            print(f"Beam hardening correction: enabled ({coeff_str})")
        if self.bone_bhc:
            print(f"Bone BHC (two-pass): enabled (threshold={self.bone_bhc_threshold} HU, "
                  f"mono bone={self.bone_bhc_hu} HU)")

        # Flat-field constants (only if preprocessing)
        if do_preprocess:
            print("Fused preprocessing + weighting + filtering (GPU)...")
            dark_gpu = torch.from_numpy(self.dark_field.astype(np.float32)).to(device)
            epsilon = 1e-6
            sharpness = self.soft_clip_sharpness
            upper_val = self.upper_clamp_value

            # Per-pixel I₀: corrects detector response variations (beam vignetting, pixel gain)
            I0_gpu = torch.from_numpy(
                (self.bright_field.astype(np.float32)
                 - self.dark_field.astype(np.float32))
            ).to(device)  # (N_b, N_a)
            I0_gpu = I0_gpu.clamp(min=1.0)  # guard dead pixels
            print(f"  Per-pixel I0: mean={float(I0_gpu.mean()):.0f}, "
                  f"min={float(I0_gpu.min()):.0f}, max={float(I0_gpu.max()):.0f}")
        else:
            print("Fused weighting + filtering (GPU)...")

        # Parker (short-scan) redundancy weights — computed once, moved to GPU
        parker_weight_gpu = None
        if self.parker_weighting:
            parker_weight = self._compute_parker_weights()
            if parker_weight is not None:
                parker_weight_gpu = parker_weight.to(device)  # (N_angles, N_a)
        self._parker_applied = parker_weight_gpu is not None

        # Cache filter parameters for bone BHC second pass
        if self.bone_bhc:
            self._cone_weight_gpu = cone_weight
            self._filter_kernel_gpu = filter_kernel
            self._parker_weight_gpu = parker_weight_gpu

        # Allocate output array
        if do_preprocess:
            try:
                float_projections = np.empty(self.projections.shape, dtype=np.float32)
            except MemoryError:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                float_projections = np.memmap(
                    temp_file.name, dtype=np.float32, mode='w+',
                    shape=self.projections.shape
                )

        # --- Dynamic chunk sizing ---
        budget = self._gpu_free_bytes()
        bytes_per_proj = self.N_b * self.N_a * 4  # one float32 projection
        persistent = 2 * bytes_per_proj  # cone_weight + filter_kernel
        if do_preprocess:
            persistent += 2 * bytes_per_proj  # dark_gpu + I0_gpu
        peak_multiplier = 7 if do_preprocess else 6
        chunk_size = int((budget - persistent) // (peak_multiplier * bytes_per_proj))
        chunk_size = max(1, min(chunk_size, self.N_angles))

        if budget == float('inf'):
            print(f"  GPU memory: CPU mode — no GPU constraint")
        else:
            total_mem = torch.cuda.mem_get_info()[1]
            free_mem = budget / 0.85
            print(f"  GPU memory: {total_mem / 2**30:.2f} GiB total, "
                  f"{free_mem / 2**30:.2f} GiB free → budget {budget / 2**30:.2f} GiB")

        # When ring correction is enabled, split into two passes:
        #   Pass 1: flat-field + log + MAR → sinogram on CPU
        #   Ring correction on full sinogram (needs all angles)
        #   Pass 2: cone-beam weighting + ramp filter + Parker
        if (self.ring_correction or self.bone_bhc) and do_preprocess:
            # --- Pass 1: flat-field + log + MAR ---
            print(f"  Pass 1 (flat-field + log): chunk_size={chunk_size}")
            for start in range(0, self.N_angles, chunk_size):
                end = min(start + chunk_size, self.N_angles)
                chunk = torch.from_numpy(
                    np.array(self.projections[start:end], dtype=np.float32)
                ).to(device)

                T = (chunk - dark_gpu) / (I0_gpu + epsilon)
                if self.soft_clip_transmission:
                    T = epsilon + F.softplus(T - epsilon, beta=sharpness, threshold=20.0)
                    if self.upper_clamp:
                        T = upper_val - F.softplus(upper_val - T, beta=sharpness, threshold=20.0)
                else:
                    if self.upper_clamp:
                        T = torch.clamp(T, min=epsilon, max=upper_val)
                    else:
                        T = torch.clamp(T, min=epsilon)
                chunk = -torch.log(T)
                if self.clamp_mode == "soft":
                    chunk = F.softplus(chunk, beta=50.0, threshold=20.0)
                elif self.clamp_mode == "hard":
                    chunk = torch.clamp(chunk, min=0.0)
                del T

                if self.metal_artifact_reduction:
                    metal_mask = chunk > self.mar_threshold
                    if metal_mask.any():
                        _interpolate_metal_pixels(chunk, metal_mask)
                    del metal_mask

                chunk = self._apply_bhc(chunk)

                float_projections[start:end] = chunk.cpu().numpy()

            # Free flat-field GPU memory
            del dark_gpu, I0_gpu
            torch.cuda.empty_cache()

            # --- Air normalization on full sinogram (if enabled) ---
            # FDK does its own fused flat-field+log on the GPU rather
            # than calling preprocess_sinogram, so the correction has to
            # be applied here too — same function, same place in the
            # order (before ring correction), so every backend sees the
            # same line integrals.
            if self.air_normalization:
                air_normalize_sinogram(float_projections)

            # --- Ring correction on full sinogram (if enabled) ---
            if self.ring_correction:
                ring_artifact_correction(float_projections,
                                         median_width=self.ring_median_width)

            # Store unfiltered sinogram for bone BHC (after ring correction, before filtering)
            if self.bone_bhc:
                self._unfiltered_sinogram = float_projections.copy()

            # --- Pass 2: cone-beam weighting + ramp filter + Parker ---
            # Re-measure GPU budget (flat-field tensors freed)
            budget = self._gpu_free_bytes()
            persistent = 2 * bytes_per_proj  # cone_weight + filter_kernel
            chunk_size = int((budget - persistent) // (6 * bytes_per_proj))
            chunk_size = max(1, min(chunk_size, self.N_angles))
            print(f"  Pass 2 (weighting + filtering): chunk_size={chunk_size}")
            for start in range(0, self.N_angles, chunk_size):
                end = min(start + chunk_size, self.N_angles)
                chunk = torch.from_numpy(
                    float_projections[start:end]
                ).to(device)

                chunk = chunk * cone_weight
                chunk = torch.fft.rfft(chunk, dim=2, norm='forward')
                chunk = chunk * filter_kernel
                chunk = torch.fft.irfft(chunk, n=self.N_a, dim=2, norm='forward')
                if parker_weight_gpu is not None:
                    chunk = chunk * parker_weight_gpu[start:end].unsqueeze(1)

                float_projections[start:end] = chunk.cpu().numpy()

        else:
            # --- Single-pass pipeline (no ring correction) ---
            print(f"  Preprocessing: chunk_size={chunk_size}")
            for start in range(0, self.N_angles, chunk_size):
                end = min(start + chunk_size, self.N_angles)

                chunk = torch.from_numpy(
                    np.array(self.projections[start:end], dtype=np.float32)
                ).to(device)

                # 1. Flat-field + log (only for HU path)
                if do_preprocess:
                    T = (chunk - dark_gpu) / (I0_gpu + epsilon)
                    if self.soft_clip_transmission:
                        T = epsilon + F.softplus(T - epsilon, beta=sharpness, threshold=20.0)
                        if self.upper_clamp:
                            T = upper_val - F.softplus(upper_val - T, beta=sharpness, threshold=20.0)
                    else:
                        if self.upper_clamp:
                            T = torch.clamp(T, min=epsilon, max=upper_val)
                        else:
                            T = torch.clamp(T, min=epsilon)
                    chunk = -torch.log(T)
                    if self.clamp_mode == "soft":
                        chunk = F.softplus(chunk, beta=50.0, threshold=20.0)
                    elif self.clamp_mode == "hard":
                        chunk = torch.clamp(chunk, min=0.0)
                    del T

                # 1b. Metal artifact reduction
                if self.metal_artifact_reduction:
                    metal_mask = chunk > self.mar_threshold
                    if metal_mask.any():
                        _interpolate_metal_pixels(chunk, metal_mask)
                    del metal_mask

                # 1c. Beam hardening correction
                chunk = self._apply_bhc(chunk)

                # 2. Cone-beam weighting
                chunk = chunk * cone_weight

                # 3. Ramp filter (FFT → multiply → IFFT)
                chunk = torch.fft.rfft(chunk, dim=2, norm='forward')
                chunk = chunk * filter_kernel
                chunk = torch.fft.irfft(chunk, n=self.N_a, dim=2, norm='forward')

                # 3b. Parker (short-scan) redundancy weighting
                if parker_weight_gpu is not None:
                    chunk = chunk * parker_weight_gpu[start:end].unsqueeze(1)

                result_np = chunk.cpu().numpy()
                if do_preprocess:
                    float_projections[start:end] = result_np
                else:
                    self.projections[start:end] = result_np

        # Finalize
        if do_preprocess:
            if isinstance(float_projections, np.memmap):
                float_projections.flush()
            self.projections = float_projections
            print(f"Fused preprocessing complete. dtype: {self.projections.dtype}")
        else:
            self._flush_projections()
            print("Fused weighting + filtering complete.")

    def backprojection(self):
        Nx, Ny, Nz = self.vol_shape

        with torch.no_grad():
            # Create 1D coordinate vectors (kept on self for display_volume())
            self.x = (torch.arange(Nx, device=device, dtype=torch.float32) - (Nx - 1) / 2) * self.dx + self.vol_origin[0]
            self.y = (torch.arange(Ny, device=device, dtype=torch.float32) - (Ny - 1) / 2) * self.dx + self.vol_origin[1]
            self.z = (torch.arange(Nz, device=device, dtype=torch.float32) - (Nz - 1) / 2) * self.dz + self.vol_origin[2]

            # Allocate output volume — try GPU first, fall back to CPU for large volumes
            volume_bytes = Nx * Ny * Nz * 4  # float32
            try:
                self.reconstructed_volume = torch.zeros((Nx, Ny, Nz), device=device, dtype=torch.float32)
                vol_on_cpu = False
            except torch.cuda.OutOfMemoryError:
                print(f"  Volume ({volume_bytes / 2**30:.2f} GiB) exceeds GPU memory — using CPU-resident volume")
                torch.cuda.empty_cache()
                self.reconstructed_volume = torch.zeros((Nx, Ny, Nz), dtype=torch.float32)
                vol_on_cpu = True

            # --- Dynamic GPU memory sizing for backprojection ---
            budget = self._gpu_free_bytes()
            f32 = 4  # bytes per float32

            # Fixed costs already on GPU: (volume if on GPU) + X_2d + Y_2d + coord vectors
            xy_grids_bytes = 2 * Nx * Ny * f32  # X_2d, Y_2d
            # Per-angle 2D intermediates: U_2d, inv_U_2d, a_2d, w_2d
            angle_intermediates = 4 * Nx * Ny * f32
            if vol_on_cpu:
                fixed_cost = xy_grids_bytes + angle_intermediates
            else:
                fixed_cost = volume_bytes + xy_grids_bytes + angle_intermediates

            # Step 1: Decide projection preload (GPU vs CPU)
            proj_all_bytes = self.N_angles * self.N_b * self.N_a * f32
            remaining = budget - fixed_cost
            if proj_all_bytes < remaining * 0.4:
                force_cpu = False
                remaining -= proj_all_bytes
            else:
                force_cpu = True
                remaining -= self.N_b * self.N_a * f32  # single proj on GPU at a time

            # Step 2: Size z_chunk from remaining budget
            # Per z-slice peak: grid_buf(2) + b_3d with broadcast temps(3)
            #   + sampled(1) + chunk_contrib(1) + a_2d expand temp(1) = 8 floats/voxel
            z_per_slice = 8 * Nx * Ny * f32
            z_chunk_size = int(remaining // z_per_slice) if z_per_slice > 0 else Nz
            z_chunk_size = max(1, min(z_chunk_size, Nz))

            # Print sizing summary
            if budget == float('inf'):
                print(f"  GPU memory: CPU mode — no GPU constraint")
            else:
                total_mem = torch.cuda.mem_get_info()[1]
                free_mem = budget / 0.85
                print(f"  GPU memory: {total_mem / 2**30:.2f} GiB total, "
                      f"{free_mem / 2**30:.2f} GiB free → budget {budget / 2**30:.2f} GiB")
            proj_loc = "CPU" if force_cpu else "GPU"
            vol_loc = "CPU" if vol_on_cpu else "GPU"
            print(f"  Backprojection: proj={proj_loc}, vol={vol_loc}, z_chunk={z_chunk_size}, "
                  f"volume={volume_bytes / 2**30:.2f} GiB")

            # Preload all projections into a single contiguous tensor
            proj_tensor = self._preload_projections(force_cpu=force_cpu)
            proj_on_gpu = proj_tensor.is_cuda

            # 2D coordinate grids (z-independent) — computed once
            X_2d, Y_2d = torch.meshgrid(self.x, self.y, indexing='ij')  # (Nx, Ny)

            # Detector centering (IMPORTANT — see COR-offset note below).
            # The rotation axis projects to detector column central_pixel_a and the
            # mid-plane ray to row central_pixel_b (scan.xml CentreOfRotation /
            # CentralSlice). The stored VFF projections are NOT pre-centred, so we
            # must shift the sampled detector coordinate by the displacement of the
            # TRUE centre from the geometric centre (N-1)/2. Omitting this (the old
            # a_offset=b_offset=0) reconstructs off-isocentre objects non-round and
            # mis-registers edges (worst in the vertical/b direction, ~27 px here).
            #
            # SIGN VERIFIED EMPIRICALLY (June 2026, TRIUMF scanner) against the GEHC
            # reference recon of Scan_1955: the correct offset is
            # ((N-1)/2 - central_pixel)*pixel_size. (An earlier attempt used the
            # OPPOSITE sign, which doubled the error and was reverted — do not flip.)
            #
            # cor_offset_scale: 1.0 = apply (default, correct); 0.0 = off (legacy,
            # for back-compat / comparison); -1.0 = flipped sign (diagnostic only).
            a_offset = self.cor_offset_scale * ((self.N_a - 1) / 2.0 - self.central_pixel_a) * self.da
            b_offset = self.cor_offset_scale * ((self.N_b - 1) / 2.0 - self.central_pixel_b) * self.db
            if self.cor_offset_scale != 0.0:
                print(f"  Detector COR offset (scale={self.cor_offset_scale:+.1f}): "
                      f"a={a_offset:+.4f} mm, b={b_offset:+.4f} mm")
            if self.det_psi_rad:
                print(f"  Detector in-plane rotation: psi = "
                      f"{np.rad2deg(self.det_psi_rad):+.4f} deg "
                      f"(a-coordinate becomes z-dependent)")
            else:
                print("  Detector COR offset: OFF (legacy, COR at detector centre)")
            a_scale = 1.0 / (self.a_length / 2)
            b_scale = 1.0 / (self.b_length / 2)

            # Pre-compute cos/sin for all angles
            cos_beta = torch.cos(self.angles)  # (N_angles,)
            sin_beta = torch.sin(self.angles)  # (N_angles,)

            n_z_chunks = (Nz + z_chunk_size - 1) // z_chunk_size

            # Pre-allocate grid buffer for largest z-chunk to avoid repeated allocation
            max_z_chunk = min(z_chunk_size, Nz)
            grid_buf = torch.empty((1, Nx, Ny * max_z_chunk, 2), device=device, dtype=torch.float32)

            # Angle-outer loop: U, a, weight are z-independent → compute as 2D per angle
            for i in range(self.N_angles):
                if proj_on_gpu:
                    proj = proj_tensor[i].unsqueeze(0).unsqueeze(0)
                else:
                    proj = proj_tensor[i].to(device).unsqueeze(0).unsqueeze(0)

                cb = cos_beta[i]
                sb = sin_beta[i]

                # 2D geometry (Nx, Ny) — 30× fewer elements than 3D
                # U = R_s + x': distance from source to voxel along central ray
                U_2d = self.R_s + X_2d * cb + Y_2d * sb + 1e-8  # (Nx, Ny)

                # Detector coordinate: project voxel onto flat detector at distance SDD
                inv_U_2d = self.SDD / U_2d                        # (Nx, Ny)

                # Detector coordinates in mm, before normalisation.
                a_mm_2d = inv_U_2d * (-X_2d * sb + Y_2d * cb) + a_offset
                # Normalized a-coordinate (z-independent WHEN psi == 0)
                a_2d = a_mm_2d * a_scale  # (Nx, Ny)

                # FDK weight: (R_s/U)² — inverse-square law from source
                w_2d = (self.R_s / U_2d) ** 2  # (Nx, Ny)

                # z-inner loop: only b depends on z
                for zc in range(n_z_chunks):
                    z_start = zc * z_chunk_size
                    z_end = min(z_start + z_chunk_size, Nz)
                    n_z = z_end - z_start
                    z_vals = self.z[z_start:z_end]  # (n_z,)

                    # b-coordinate via broadcasting: (Nx, Ny, 1) * (n_z,) → (Nx, Ny, n_z)
                    b_mm_3d = inv_U_2d.unsqueeze(-1) * z_vals + b_offset
                    b_3d = b_mm_3d * b_scale
                    a_3d = None
                    if self.det_psi_rad:
                        # In-plane detector rotation. This COUPLES a to b, so the
                        # a-coordinate stops being z-independent and has to
                        # become 3-D — which is why psi != 0 costs memory and
                        # time that psi == 0 does not. Rotating the sampling
                        # point by -psi is equivalent to rotating the detector by
                        # +psi.
                        cpsi = float(np.cos(self.det_psi_rad))
                        spsi = float(np.sin(self.det_psi_rad))
                        a_r = a_mm_2d.unsqueeze(-1) * cpsi + b_mm_3d * spsi
                        b_r = -a_mm_2d.unsqueeze(-1) * spsi + b_mm_3d * cpsi
                        a_3d = a_r * a_scale
                        b_3d = b_r * b_scale

                    # Write into pre-allocated grid buffer (no allocation)
                    flat_len = Ny * n_z
                    # a_2d: (Nx, Ny) → expand to (Nx, Ny, n_z) → reshape to (Nx, Ny*n_z)
                    grid_buf[0, :, :flat_len, 0] = (
                        a_2d.unsqueeze(-1).expand(-1, -1, n_z) if a_3d is None
                        else a_3d).reshape(Nx, flat_len)
                    grid_buf[0, :, :flat_len, 1] = b_3d.reshape(Nx, flat_len)

                    sampled = F.grid_sample(proj, grid_buf[:, :, :flat_len, :], mode='bilinear', align_corners=True)
                    chunk_contrib = sampled[0, 0].view(Nx, Ny, n_z) * w_2d.unsqueeze(-1)
                    if vol_on_cpu:
                        self.reconstructed_volume[:, :, z_start:z_end] += chunk_contrib.cpu()
                    else:
                        self.reconstructed_volume[:, :, z_start:z_end] += chunk_contrib

                del proj

            del proj_tensor, grid_buf

        # Apply angular normalization (Δβ) to ensure proper scaling
        # This converts the discrete sum to a proper integral approximation:
        # f(x,y,z) = Σᵢ p_filtered(βᵢ) * (R/L)² * Δβ
        # Without this, reconstructions with different numbers of projections
        # would have different intensity scales (proportional to N_angles)
        if self.N_angles > 1:
            angle_range = float(self.angles[-1] - self.angles[0])
            # Handle angle wraparound (e.g., when angles are modulo 360 and span ~360°)
            # If computed range is very small (<1°) but we have many projections,
            # this indicates the angles wrapped around - assume full 360° scan
            if abs(angle_range) < np.pi / 180:  # Less than 1 degree
                print(f"Warning: Detected angle wraparound (range={np.rad2deg(angle_range):.4f}°). Assuming full 360° scan.")
                delta_beta = 2 * np.pi / self.N_angles
            else:
                delta_beta = angle_range / (self.N_angles - 1)
        else:
            delta_beta = 2 * np.pi  # Single projection edge case
        if self.physical_normalization:
            parker_applied = getattr(self, '_parker_applied', False)
            angle_range_deg = abs(float(self.angles[-1] - self.angles[0])) * 180 / np.pi
            is_full_scan = angle_range_deg >= 350.0

            if parker_applied:
                # Parker weights (conjugate pairs sum to 1) already ensure each
                # ray is counted exactly once → no 1/2 prefactor needed.
                self.reconstructed_volume *= delta_beta
                print(f"Applied angular normalization (Parker, no 1/2): Δβ = {float(delta_beta):.6f}")
            elif is_full_scan:
                # Full scan without Parker: each ray counted twice → 1/2 prefactor.
                # μ = (1/2) ∫₀²π [R_s²/U²] g̃(β,a) dβ
                self.reconstructed_volume *= delta_beta / 2.0
                print(f"Applied angular normalization (full scan, 1/2): Δβ/2 = {float(delta_beta)/2.0:.6f}")
            else:
                # Short scan without Parker: rays are NOT uniformly redundant,
                # so no 1/2 prefactor. Result will have shading artifacts but
                # correct overall scale.
                self.reconstructed_volume *= delta_beta
                print(f"Applied angular normalization (short scan, no Parker, no 1/2): Δβ = {float(delta_beta):.6f}")
        else:
            self.reconstructed_volume *= delta_beta
        print(f"Angular step: Δβ = {float(delta_beta):.6f} rad ({float(delta_beta) * 180 / np.pi:.4f}°)")

    def convert_to_hu(self):
        """
        Convert reconstructed volume from true μ (mm⁻¹) to Hounsfield Units.

        Uses physics-based conversion: HU = (μ - μ_water) / μ_water × 1000

        This produces approximate HU values. The final polynomial calibration
        (from phantom insert measurements) is applied in run_fdk_recon.py.
        """
        vol_np = self.reconstructed_volume.cpu().numpy() if hasattr(self.reconstructed_volume, 'cpu') else self.reconstructed_volume
        del self.reconstructed_volume
        torch.cuda.empty_cache()

        print("=" * 60)
        print("Physics-based HU Calibration (true μ output)")
        print("=" * 60)

        print(f"  μ_water pipeline: "
              f"{'BHC' if self.bhc_coeffs is not None else 'no-BHC'}")
        self.reconstructed_volume = mu_to_hu(vol_np, self.mu_water)
        print("=" * 60)

    def display_volume(self):

        torch.cuda.empty_cache()
        self.x = self.x.cpu().numpy()
        self.y = self.y.cpu().numpy()
        self.z = self.z.cpu().numpy()

        num_bytes = self.reconstructed_volume.element_size() * self.reconstructed_volume.nelement()
        num_megabytes = num_bytes / (1024 ** 2)
        print(f"Reconstructed volume uses approximately {num_megabytes:.2f} MB")

        # normalise the reconstruction values by gamma factor
        gamma = 1 # gamma correction to brighten low values
        torch.cuda.empty_cache()

        self.reconstructed_volume /= self.reconstructed_volume.max()
        self.reconstructed_volume *= 256

        for i in range(self.reconstructed_volume.shape[2]):
            z_slice = self.reconstructed_volume[:, :, i].cpu().numpy()

            if i == len(self.z)-1:
                break
            os.makedirs(self.folder_name, exist_ok=True)

            fig, ax = plt.subplots()

            # Display using standard imshow convention (no transpose)
            # z_slice is (Nx, Ny), displayed as (rows=x, cols=y) to match VFF loading convention
            ax.imshow(z_slice, cmap='gray', origin='lower',
                     extent=[self.y.min(), self.y.max(), self.x.min(), self.x.max()])
            ax.set_xlabel('Y position (mm)')
            ax.set_ylabel('X position (mm)')

            ax.set_aspect('equal', adjustable='box') # set aspect ratio to 1:1 (no squishing of pixels)

            # include source locations
            # Separate x and y arrays for scatter:
            if self.source_locations is not None:
                for source_pos in self.source_locations:
                    if source_pos[2] >= self.z[i] and source_pos[2] < self.z[i+1]:
                        ax.scatter(source_pos[1], source_pos[0], c='red', s=10)

            ax.set_title(f'Reconstruction slice {i} (z: {self.z[i]:.2f}-{self.z[i+1]:.2f} mm)')

            plt.savefig(f'{self.folder_name}/reconstruction_slice_{i:03d}_{self.z[i]:.2f}-{self.z[i+1]:.2f}.png', dpi=500, bbox_inches='tight')
            plt.close(fig)

    def reconstruct(self, display_volume=True):
        """
        Complete reconstruction pipeline.

        If output_hu is True and bright_field/dark_field are provided:
        1. Applies proper preprocessing (flat-field correction + log transform)
        2. Runs FDK reconstruction
        3. Converts to Hounsfield Units using theoretical calibration

        Otherwise, runs standard FDK on raw intensities.
        """
        # Step 1+2: Fused preprocessing + weighting + filtering (single GPU pass)
        print("\nApplying fused preprocessing + weighting + filtering...")
        self._preprocess_and_filter()
        print("Backprojecting...")
        self.backprojection()

        # Bone BHC two-pass (Joseph & Spital)
        if self.bone_bhc:
            print("\n" + "=" * 60)
            print("Bone BHC — Pass 2 (Joseph & Spital)")
            print("=" * 60)

            # Free pass-1 filtered projections to reduce peak memory
            # (will be re-created by _apply_cone_weight_and_filter)
            self.projections = None
            import gc; gc.collect()

            self._bone_bhc_correction()

            # Allocate fresh projections array for re-filtered output
            self.projections = np.empty(self._unfiltered_sinogram.shape,
                                        dtype=np.float32)
            self._apply_cone_weight_and_filter()

            # Free unfiltered sinogram, pass-1 volume, and cached filter params
            del self._unfiltered_sinogram
            del self.reconstructed_volume
            del self._cone_weight_gpu, self._filter_kernel_gpu, self._parker_weight_gpu
            gc.collect()
            torch.cuda.empty_cache()

            print("Pass 2: Backprojecting corrected sinogram...")
            self.backprojection()

        # Step 3: Optional HU conversion
        if self.output_hu:
            print("\nConverting to Hounsfield Units...")
            self.convert_to_hu()

        if display_volume == True:
            self.display_volume()
            print("Reconstruction plots created.")
