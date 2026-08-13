"""Machine preflight: will this reconstruction fit on this machine?

Run by every driver after the scan is loaded (all shapes known) and BEFORE
any backend allocates GPU or large host buffers. Checks three things against
per-backend requirement estimates:

  1. Is a CUDA GPU present at all? (ASTRA and TIGRE algorithms are CUDA-only
     and would crash mid-run; FDK and the voxel trainer fall back to CPU,
     which is allowed but loudly warned — orders of magnitude slower.)
  2. Does the estimated peak VRAM fit in the GPU's free memory?
  3. Does the estimated peak host RAM fit in MemAvailable?

The estimates are deliberately simple, documented formulas with a safety
margin — they exist to catch the "this cannot possibly work" and "this will
be uncomfortably tight" cases up front, not to be byte-accurate:

  fdk    VRAM: processes in z-chunks sized to free VRAM at runtime, so the
         requirement is the sinogram + one modest volume slab.
         RAM: raw projections + float sinogram + full volume.
  astra  VRAM: 2x volume + 2x sinogram (SIRT/CGLS keep forward/back-projection
         buffers — same formula the backend itself uses).
         RAM: projections (float, x2 for the ASTRA reorder copy) + volume.
  tigre  VRAM: TIGRE splits internally, so the floor is the sinogram + one
         volume split; a note says larger free VRAM = fewer splits = faster.
         RAM: ~4x volume + ~3x sinogram (measured: 12 GB RSS at a 2.9 GiB
         volume — TIGRE and the crossval path keep several copies).
  voxel  VRAM: 4x parameters (fp32 param + grad + Adam m/v) + sinogram +
         per-batch ray buffers.
         RAM: raw projections + float sinogram + exported volume.

Verdicts: ok / tight (>85% of free) / insufficient / no-gpu. `insufficient`
and `no-gpu`(when the backend requires one) abort with a clear message —
``--skip-preflight`` overrides, ``--preflight-only`` prints the report and
exits without reconstructing (for sizing jobs before submitting them).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from .utils import query_gpu_memory

GiB = float(2 ** 30)
F32 = 4
SAFETY = 1.15          # multiplied onto every estimate
TIGHT_FRAC = 0.85      # need > 85% of free => "tight"


def host_mem_available_bytes():
    """MemAvailable from /proc/meminfo, or None off-Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@dataclass
class PreflightReport:
    backend: str
    gpu_required: bool
    gpu_name: str | None
    vram_free: int | None      # None = no GPU found
    vram_needed: int           # 0 = backend adapts / CPU
    ram_free: int | None
    ram_needed: int
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.vram_free is None:
            return "no-gpu"
        if (self.vram_needed > self.vram_free
                or (self.ram_free is not None and self.ram_needed > self.ram_free)):
            return "insufficient"
        if (self.vram_needed > TIGHT_FRAC * self.vram_free
                or (self.ram_free is not None
                    and self.ram_needed > TIGHT_FRAC * self.ram_free)):
            return "tight"
        return "ok"

    @property
    def fatal(self) -> bool:
        v = self.verdict
        return v == "insufficient" or (v == "no-gpu" and self.gpu_required)


# ------------------------------------------------------------- estimators --

def _sino_bytes(n_angles: int, n_b: int, n_a: int) -> int:
    return n_angles * n_b * n_a * F32


def _vol_bytes(vol_shape) -> int:
    nx, ny, nz = (int(v) for v in vol_shape)
    return nx * ny * nz * F32


def estimate(backend: str, *, n_angles: int, n_b: int, n_a: int, vol_shape,
             rays_per_batch: int = 0, samples_per_ray: int = 0):
    """(gpu_bytes, host_bytes, gpu_required, notes) for one backend."""
    sino = _sino_bytes(n_angles, n_b, n_a)
    vol = _vol_bytes(vol_shape)
    notes: list[str] = []

    if backend == "fdk":
        gpu = int((sino + vol / 8) * SAFETY)
        host = int((2 * sino + vol) * SAFETY)
        notes.append("FDK z-chunks itself to free VRAM; runs on CPU if none "
                     "(slow).")
        return gpu, host, False, notes

    if backend == "astra":
        gpu = int(2 * (sino + vol) * SAFETY)
        host = int((2 * sino + vol) * SAFETY)
        notes.append("ASTRA CUDA algorithms cannot split the volume — the "
                     "whole 2x(vol+sino) workspace must fit at once.")
        return gpu, host, True, notes

    if backend == "tigre":
        gpu = int((sino + vol / 8) * SAFETY)
        host = int((4 * vol + 3 * sino) * SAFETY)
        notes.append("TIGRE splits the volume across GPU passes; more free "
                     "VRAM = fewer splits = faster. Host RAM is the binding "
                     "constraint (~4x volume for TIGRE + crossval copies).")
        return gpu, host, True, notes

    if backend == "voxel":
        params = vol  # one fp32 parameter per voxel
        batch = int(rays_per_batch) * max(1, int(samples_per_ray)) * F32 * 6
        gpu = int((4 * params + sino + batch) * SAFETY)
        host = int((2 * sino + vol) * SAFETY)
        notes.append("Voxel grid trains param+grad+Adam(m,v) = 4x volume on "
                     "the GPU. CPU fallback exists but is impractically slow.")
        return gpu, host, False, notes

    raise ValueError(f"unknown backend {backend!r}")


# ------------------------------------------------------------------ check --

def run_preflight(backend: str, ctx, *, gpu_index: int = 0,
                  rays_per_batch: int = 0, samples_per_ray: int = 0,
                  skip: bool = False, only: bool = False,
                  logger=None) -> PreflightReport:
    """Print the machine-fit report; abort on a fatal verdict unless skipped.

    ``only=True``: print and sys.exit(0) — a dry run for sizing a job.
    ``skip=True``: print the report but never abort.
    ``logger``: optional ReconLogger — the report is recorded on the W&B run
    (config + summary), and a fatal abort marks that run FAILED with the
    reason, so an auto-aborted job is visible in W&B rather than silent.
    """
    n_angles, n_b, n_a = (int(s) for s in ctx.projections.shape)
    gpu_b, host_b, gpu_req, notes = estimate(
        backend, n_angles=n_angles, n_b=n_b, n_a=n_a,
        vol_shape=ctx.geometry["vol_shape"],
        rays_per_batch=rays_per_batch, samples_per_ray=samples_per_ray)

    gpu = query_gpu_memory(gpu_index)
    report = PreflightReport(
        backend=backend, gpu_required=gpu_req,
        gpu_name=(gpu or {}).get("name"),
        vram_free=(gpu or {}).get("free_bytes"),
        vram_needed=gpu_b,
        ram_free=host_mem_available_bytes(),
        ram_needed=host_b,
        notes=notes,
    )

    nx, ny, nz = (int(v) for v in ctx.geometry["vol_shape"])
    print(f"\nPreflight ({backend}):")
    print(f"  Data:   {n_angles} x {n_b} x {n_a} projections -> "
          f"{nx} x {ny} x {nz} volume")
    if report.vram_free is None:
        print(f"  GPU:    NONE FOUND"
              + (" — this backend requires a CUDA GPU" if gpu_req
                 else " — will run on CPU (very slow)"))
    else:
        print(f"  GPU:    {report.gpu_name} — "
              f"need ~{gpu_b / GiB:.1f} GiB VRAM, "
              f"{report.vram_free / GiB:.1f} GiB free")
    if report.ram_free is not None:
        print(f"  RAM:    need ~{host_b / GiB:.1f} GiB, "
              f"{report.ram_free / GiB:.1f} GiB available")
    for n in notes:
        print(f"  Note:   {n}")
    print(f"  Verdict: {report.verdict.upper()}")

    if logger is not None:
        logger.log_preflight(report)

    if only:
        if logger is not None:
            logger.finish()
        sys.exit(0)
    if report.fatal and not skip:
        reason = (f"preflight {report.verdict}: "
                  f"need {report.vram_needed / GiB:.1f} GiB VRAM / "
                  f"{report.ram_needed / GiB:.1f} GiB RAM"
                  if report.verdict == "insufficient"
                  else "preflight: no CUDA GPU found and this backend "
                       "requires one")
        print("\nAborting before any large allocation. Reduce the load "
              "(--downsample, smaller --fov-xy/--fov-z, larger --voxel-xy) "
              "or run on a bigger machine. --skip-preflight overrides.")
        if logger is not None:
            logger.abort(reason)
        sys.exit(1)
    if report.fatal and skip:
        print("  (--skip-preflight: continuing anyway — expect an OOM or "
              "a crash)")
    elif report.verdict == "tight":
        print("  (tight fit — expect heavy memory pressure; consider "
              "--downsample or a smaller FOV)")
    return report


def add_preflight_args(parser) -> None:
    parser.add_argument(
        '--skip-preflight', action='store_true', default=False,
        help='Run even when the preflight machine check says the job will '
             'not fit (expect OOM).')
    parser.add_argument(
        '--preflight-only', action='store_true', default=False,
        help='Load the scan, print the machine-fit report (GPU/VRAM/RAM vs '
             'this job), and exit without reconstructing. Use to size a job '
             'before submitting it.')
