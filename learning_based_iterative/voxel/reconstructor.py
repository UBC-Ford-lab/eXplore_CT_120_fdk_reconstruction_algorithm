"""Learning-based iterative reconstruction with a dense voxel grid.

Reconstruction as optimization: the volume is a `VoxelGrid` (one free
parameter per voxel — SIRT's representation) fitted to the measured line
integrals by Adam through the differentiable renderer. This is the recipe
validated in muNeRF's ``configs/scan_1510_VOXEL_mse.yaml`` (run 9bn7j2ua),
ported to the ct_core backend contract so it is a drop-in peer of the FDK /
ASTRA / TIGRE backends:

  consume  raw-count projections (N_angles, N_b, N_a) + angles (radians, FDK
           convention) + ct_core ``build_geometry`` dict
  return   float32 (Nx, Ny, Nz) volume in HU via ``reconstructed_volume``

Recipe notes carried over from the validated config:
  * plain MSE data term (SIRT's objective; no structural terms)
  * non-negativity as a PROJECTION after each step (SIRT's clip, not softplus)
  * init near air, not water — gradient pressure raises mu only where the
    projections demand it
  * short LR warmup (500) + cosine decay; the optimum sits EARLY (~16k steps
    on Scan_1510), so long schedules mostly fit noise
  * quadrature finer than the voxel (~0.55 voxel steps) to avoid aliasing
    against a critically-sampled grid
  * one held-out projection is excluded from training and scored for
    early stopping — the same SIRT-style holdout the TIGRE backend uses

Like the classical backends, the reconstruction FOV must cover the object
(truncated projections are unmodeled); the model domain here IS the export
FOV cylinder.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ...ct_core.calibration import default_mu_water, mu_to_hu
from ...ct_core.preprocessing import preprocess_sinogram
from ..scene import Scene, model_domain_from_geometry
from ..ray_sampler import rays_from_indices, sample_random_rays
from ..renderer import render_rays
from .model import VoxelGrid


class VoxelReconstructor:
    """Voxel-grid learning-based iterative reconstruction (backend contract)."""

    def __init__(self, projections, angles, geometry,
                 iterations: int = 20000,
                 rays_per_batch: int = 16384,
                 lr: float = 1e-4,
                 samples_per_ray: int | None = None,
                 lr_warmup_iters: int = 500,
                 init_density: float = 0.001,
                 gpu_index: int = 0,
                 seed: int = 0,
                 bright_field=None, dark_field=None,
                 output_hu: bool = True,
                 mu_water: float | None = None,
                 bhc_coeffs=None,
                 ring_correction: bool = False,
                 ring_median_width: int = 51,
                 crossval: bool = True,
                 holdout_index: int | None = None,
                 eval_every: int = 250,
                 patience: int = 8,
                 holdout_rays: int = 65536,
                 log_every: int = 500,
                 log_fn=None):
        self.projections = projections
        self.angles = np.asarray(angles)
        self.geometry = dict(geometry)
        self.iterations = int(iterations)
        self.rays_per_batch = int(rays_per_batch)
        self.lr = float(lr)
        self.samples_per_ray = samples_per_ray
        self.lr_warmup_iters = int(lr_warmup_iters)
        self.init_density = float(init_density)
        self.gpu_index = int(gpu_index)
        self.seed = int(seed)
        self.bright_field = bright_field
        self.dark_field = dark_field
        self.output_hu = bool(output_hu)
        self.mu_water = mu_water
        self.bhc_coeffs = bhc_coeffs
        self.ring_correction = bool(ring_correction)
        self.ring_median_width = int(ring_median_width)
        self.crossval = bool(crossval)
        self.holdout_index = holdout_index
        self.eval_every = int(eval_every)
        self.patience = int(patience)
        self.holdout_rays = int(holdout_rays)
        self.log_every = int(log_every)
        # Optional live-metric sink: Callable[[dict, int], None] — called as
        # log_fn(metrics, step). Keeps the backend free of any wandb import;
        # the driver passes ReconLogger.log.
        self.log_fn = log_fn

        self.reconstructed_volume = None
        self.crossval_history: list[dict] = []

    # ------------------------------------------------------------------ setup

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_index}")
        print("  WARNING: CUDA not available — training the voxel grid on CPU "
              "will be orders of magnitude slower.")
        return torch.device("cpu")

    def _auto_samples_per_ray(self, domain) -> int:
        """~0.55-voxel quadrature step across the widest in-plane chord.

        The grid is critically sampled at the export voxel size; a coarser
        step aliases against it (see the VOXEL config's renderer notes).
        """
        dx = float(self.geometry["dx"])
        chord = (2.0 * float(domain.radius_xy) if domain.radius_xy is not None
                 else float((domain.aabb_max - domain.aabb_min).max()))
        return max(64, int(math.ceil(chord / (0.55 * dx))))

    def _preprocess(self) -> np.ndarray:
        return preprocess_sinogram(
            self.projections, self.bright_field, self.dark_field,
            bhc_coeffs=self.bhc_coeffs,
            ring_correction=self.ring_correction,
            ring_median_width=self.ring_median_width,
        )

    # ------------------------------------------------------------------- run

    def reconstruct(self) -> np.ndarray:
        Nx, Ny, Nz = (int(v) for v in self.geometry["vol_shape"])
        n_params = Nx * Ny * Nz
        device = self._device()

        print(f"  Voxel grid: {Nx} x {Ny} x {Nz} = {n_params/1e6:.1f} M free "
              f"parameters ({n_params * 4 / 2**30:.2f} GiB fp32, ~4x with Adam)")

        sinogram = torch.from_numpy(np.ascontiguousarray(self._preprocess()))
        angles_t = torch.as_tensor(self.angles, dtype=torch.float32)

        domain = model_domain_from_geometry(self.geometry, shape="cylinder")
        scene = Scene(sinogram=sinogram.to(device), angles=angles_t.to(device),
                      geometry=self.geometry, scan_name="", model_domain=domain)

        spp = (int(self.samples_per_ray) if self.samples_per_ray
               else self._auto_samples_per_ray(domain))
        step_mm = (2.0 * float(domain.radius_xy)) / spp if domain.radius_xy else float("nan")
        print(f"  Quadrature: {spp} samples/ray (~{step_mm:.4f} mm; voxel "
              f"{float(self.geometry['dx']):.4f} mm)")

        # Grid shape == export volume shape: (Dz, Hy, Wx) tiling the FOV AABB.
        model = VoxelGrid((Nz, Ny, Nx), init_density=self.init_density).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        gen = torch.Generator(device=device).manual_seed(self.seed)

        # ---- held-out projection (never sampled for training) --------------
        holdout = None
        if self.crossval and scene.n_angles > 1:
            holdout = (int(self.holdout_index) if self.holdout_index is not None
                       else scene.n_angles // 2)
            n_b, n_a = scene.detector_shape
            hg = torch.Generator(device=device).manual_seed(self.seed + 1)
            hb = torch.randint(n_b, (self.holdout_rays,), generator=hg, device=device)
            ha = torch.randint(n_a, (self.holdout_rays,), generator=hg, device=device)
            hidx = torch.full_like(hb, holdout)
            h_o, h_d, h_t = rays_from_indices(scene, hidx, hb, ha)
            print(f"  Cross-validation: holding out projection {holdout} "
                  f"({self.holdout_rays} rays, eval every {self.eval_every}, "
                  f"patience {self.patience})")

        def lr_at(it: int) -> float:
            if self.lr_warmup_iters and it < self.lr_warmup_iters:
                return self.lr * (it + 1) / self.lr_warmup_iters
            t = (it - self.lr_warmup_iters) / max(1, self.iterations - self.lr_warmup_iters)
            return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

        best_mse, best_it, bad_evals = float("inf"), 0, 0
        t0 = time.time()
        stop_reason = "max iterations"

        for it in range(self.iterations):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(it)

            origins, directions, target = sample_random_rays(
                scene, self.rays_per_batch, generator=gen, device=device,
                exclude_angle=holdout)
            pred = render_rays(origins, directions, model, scene,
                               num_samples=spp, stratified=True, generator=gen)
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clamp_nonneg()

            if self.log_every and (it % self.log_every == 0 or it == self.iterations - 1):
                rate = (it + 1) / max(1e-9, time.time() - t0)
                print(f"  iter {it:6d}/{self.iterations}  loss {loss.item():.3e}  "
                      f"lr {lr_at(it):.2e}  {rate:.1f} it/s", flush=True)
                if self.log_fn is not None:
                    self.log_fn({"train/loss": float(loss.item()),
                                 "train/lr": lr_at(it),
                                 "train/it_per_s": rate}, it + 1)

            if holdout is not None and (it + 1) % self.eval_every == 0:
                with torch.no_grad():
                    h_pred = render_rays(h_o, h_d, model, scene,
                                         num_samples=spp, stratified=False)
                    h_mse = float(torch.mean((h_pred - h_t) ** 2))
                self.crossval_history.append({"iter": it + 1, "holdout_mse": h_mse})
                if self.log_fn is not None:
                    self.log_fn({"crossval/holdout_mse": h_mse}, it + 1)
                if h_mse < best_mse * (1.0 - 1e-4):
                    best_mse, best_it, bad_evals = h_mse, it + 1, 0
                else:
                    bad_evals += 1
                    if bad_evals >= self.patience:
                        stop_reason = (f"holdout MSE plateau ({self.patience} evals "
                                       f"without improvement; best {best_mse:.3e} "
                                       f"at iter {best_it})")
                        break

        elapsed = time.time() - t0
        print(f"  Training finished after {it + 1} iterations "
              f"({elapsed/60:.1f} min): {stop_reason}")
        if holdout is not None and self.crossval_history:
            print(f"  Holdout MSE: {self.crossval_history[-1]['holdout_mse']:.3e} "
                  f"(best {best_mse:.3e} at iter {best_it})")

        # ---- export: the parameter grid IS the volume ----------------------
        # mu[0, 0] is (Dz, Hy, Wx) with indices increasing along +z/+y/+x;
        # transpose to the FDK convention (Nx, Ny, Nz).
        vol = model.mu.detach()[0, 0].cpu().numpy().transpose(2, 1, 0)
        vol = np.ascontiguousarray(vol, dtype=np.float32)

        if self.output_hu:
            mu_w = default_mu_water(self.mu_water, self.bhc_coeffs)
            vol = mu_to_hu(vol, mu_w)

        self.reconstructed_volume = vol
        return vol
