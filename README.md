# eXplore CT 120 Reconstruction

Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner. Three
algorithm families, each in its own subfolder: **fdk** (analytic), **iterative**
(ASTRA: SIRT, CGLS; TIGRE: OS-SART, SART, SIRT, MLEM), and
**learning_based_iterative** (differentiable projector + gradient descent;
currently a dense voxel grid).

## Pipeline

```
Raw projections → flat-field + log → BHC → ring correction
  → FDK: cone-weight + ramp filter + Parker + backprojection
    OR iterative: ASTRA SIRT / TIGRE OS-SART
  → [bone BHC (FDK only): segment → forward-project → re-reconstruct]
  → physics HU → two-point calibration (air→-1000, water→0) → VFF
```

BHC (water, 80 kVp) and ring correction are on by default. HU calibration measures air and water/tissue directly from the reconstructed volume (standard CT two-point formula) — self-calibrating regardless of filter or BHC settings.

## Structure

Every algorithm-independent stage lives in `ct_core` and is shared by all
backends; a reconstruction algorithm is a drop-in replacement for any other:

```
run_fdk_recon.py / run_iterative_recon.py /   thin drivers (backend flags only)
run_learned_recon.py
  └─ ct_core/pipeline.py    shared CLI args, prepare_scan() → ScanContext,
                            detector-psi calibration JSON, save_outputs()
       ├─ ct_core/scan_setup.py      scan.xml, projections, geometry, VFF export
       ├─ ct_core/preprocessing.py   flat-field+log, BHC, ring corr., downsample
       ├─ ct_core/calibration.py     mu_water constants, mu→HU conversion
       └─ ct_core/utils.py           GPU memory query

fdk/                        analytic filtered backprojection
  └─ reconstructor.py
iterative/                  classical iterative, one subfolder per toolbox
  ├─ astra/reconstructor.py     SIRT3D_CUDA, CGLS3D_CUDA
  └─ tigre/reconstructor.py     OS-SART, SART, SIRT, MLEM (+TV, PWLS, crossval)
learning_based_iterative/   reconstruction as optimization (autograd)
  ├─ scene.py                   Scene / ModelDomain containers (CANONICAL —
  ├─ ray_sampler.py             muNeRF's inr_pipeline imports these from here
  ├─ renderer.py                rather than duplicating them)
  ├─ detector_warp.py           per-pixel detector distortion → ray geometry
  └─ voxel/                     dense voxel grid (SIRT's representation)
      ├─ model.py                   VoxelGrid + grid-shape rules
      └─ reconstructor.py           Adam + MSE trainer (backend contract)
```

Backend contract: consume `ScanContext.projections` (raw counts,
`(N_angles, N_b, N_a)`), `.angles` (radians, FDK convention), `.geometry`
(dict from `build_geometry`), and return a float32 `(Nx, Ny, Nz)` volume in
HU. Anything honouring that contract plugs into the same drivers — the voxel
backend is the template for future learning-based algorithms (nerf/,
hashgrid/, gaussian_splatting/ as siblings of voxel/).

The `learning_based_iterative` package is also the canonical home of the
machinery muNeRF shares (scene containers, cone-beam ray generation, the
differentiable renderer, detector warp, the voxel grid): muNeRF's
`inr_pipeline.{dataset,ray_sampler,renderer,detector_warp,model}` re-export
from here, so there is exactly one implementation of the geometry convention.

## Learning-based iterative reconstruction

```bash
python -m reconstruction.run_learned_recon data/scans/Scan_1510 --downsample 3
```

Fits a dense voxel grid (one free parameter per voxel — SIRT's
representation) to the line integrals by Adam through the differentiable
renderer, using the recipe validated in muNeRF (plain MSE, non-negativity
projection after each step, air-start init, 500-iter LR warmup + cosine
decay, ~0.55-voxel quadrature). A held-out projection is excluded from
training and its MSE early-stops the run (`--no-crossval` disables). Needs a
CUDA GPU for realistic sizes. `--detector-warp auto` additionally applies the
per-pixel detector distortion calibration to ray geometry — a correction only
this family can express.

**Ray batch size adapts to the GPU (`--rays-per-batch auto`, the default).**
The batch is the only knob that scales with VRAM: the grid, its Adam state
and the sinogram are fixed costs, so whatever is left over should be spent on
rays per step. The driver measures free VRAM, subtracts those persistent
buffers, and sizes the batch to the remainder (the same thing FDK does for
its projection chunks) — one command fills a 16 GB card (~12 k rays) and an
80 GB card (~400-500 k rays, i.e. many more epochs over the sinogram in the
same wall time) without a hardware-specific flag. The chosen value and the
memory terms behind it are printed and logged to W&B (`rays_per_batch`,
`rays_per_batch_mode`). Because batch size changes the optimization dynamics,
**pin an integer (`--rays-per-batch 16384`) when a run must be reproducible
across different GPUs.**

**Kernel fusion (`--compile on`, default `off`).** A training step is
bandwidth-bound, not compute-bound. Measured on a P100 at 12288 rays x 1091
samples/ray (13.4 M samples/step):

| part of the step | time | |
|---|---|---|
| stratification RNG | 0.24 ms | |
| quadrature chain (ray → normalized sample coordinates) | 5.74 ms | **fusible** |
| `grid_sample` | 1.63 ms | extern kernel |
| masked Riemann sum | 0.61 ms | **fusible** |
| forward total | 8.86 ms | |
| forward + backward | 14.89 ms | |

So 72% of the forward is one elementwise chain that eager mode walks through
in separate kernels, materializing a 161 MB `(N, S, 3)` intermediate at each
of `t_samples`, `xyz_mm` and `xyz_norm`. `--compile on` hands that chain to
`torch.compile`, which fuses it into a single kernel writing only what
`grid_sample` consumes; AOTAutograd fuses the matching backward. The fusible
part is 43% of the full step, so halving it is ~1.27x, and the backward
saving comes on top.

This is fusion, **not** reduced precision. `grid_sample`, the elementwise
chain and the reduction stay float32 — the ray integral sums ~1091 terms and
projection residuals live at 1e-2, so a bf16/fp16 accumulator would put
arithmetic noise at the size of the signal. (`torch.autocast` would in any
case downcast none of this path, which contains no matmul.)

Off by default because fusion **reorders floating-point operations**: a
compiled run is comparable with other compiled runs, not with an eager
baseline. The mode is printed and travels with the run's W&B config
(`compile`). Compilation is warmed up before the timer, off the training RNG
stream, so `--compile on` and `--compile off` see the same rays and the same
optimizer trajectory. Requires Triton, i.e. compute capability >= 7.0 —
older GPUs (P100 and earlier) warn once and continue eagerly rather than
failing the run. `--compile max-autotune` benchmarks kernel variants at
compile time; minutes of extra startup, worth it only for long runs.

## Geometry self-calibration (standard, all backends)

The detector in-plane rotation (psi) is calibrated automatically for every
reconstruction: the drivers first read the scan-keyed
`data/calibration/detector_psi_<serial>_<scan>.json`; on a cache miss they
**measure psi from the scan's own projections** with the half-scan-consistency
estimator (`ct_core/geometry_selfcal.py`, the validated method ported from
muNeRF — FBP the two halves of the view range, score gradient-NCC agreement,
two-stage grid search with fail-safe guards) and write the JSON so every
later run, in any pipeline, gets a cache hit. Measurement needs a CUDA GPU
(~1–5 min, once per scan); without one, TIGRE falls back to its inline
conjugate estimator and FDK/ASTRA to psi=0, with printed notices.

Only `psi_deg` is ever applied — fitted `cpa0` intercepts are known estimator
bias and are reported as a diagnostic. `--no-geometry-autocal` disables the
whole feature (pre-2026-08-11 behaviour).

**Centre of rotation: detector geometric centre (default).** The scan.xml
`CentreOfRotation`/`CentralSlice` offset and the detector-psi rotation are
nearly degenerate corrections — at any single off-midplane height they shift
rays the same way — and applying BOTH over-corrects: a decisive experiment
(Scan_1510 off-midplane tube, 2026-08-13; midplane + z=+22 mm crops vs the
vendor reconstruction) showed psi alone matches the vendor while
psi + XML COR re-splits the tube. All backends therefore place the rotation
axis at the detector geometric centre; `--cor-mode xml` restores the legacy
scan.xml values (kept in the geometry dict under `central_pixel_*_xml`).

Standalone (pre-)calibration, e.g. on a cluster GPU node before submitting
long jobs:

```bash
python -m reconstruction.run_geometry_calibration data/scans/Scan_1510
python -m reconstruction.run_geometry_calibration data/scans/Scan_1510 --force  # re-measure
```

## Machine preflight (all backends)

Every driver checks the machine BEFORE any large allocation: is a CUDA GPU
present (ASTRA/TIGRE are CUDA-only and abort cleanly; FDK and the voxel
trainer fall back to CPU with a loud warning), does the estimated peak VRAM
fit in free GPU memory, and does the estimated host RAM fit in MemAvailable —
using documented per-backend formulas (e.g. ASTRA needs its whole
2x(volume+sinogram) workspace at once; TIGRE splits VRAM internally but keeps
~4x the volume in host RAM). Verdicts: OK / TIGHT / INSUFFICIENT / NO-GPU.

`--gpu-index` is a *logical* index (the one torch uses, relative to
`CUDA_VISIBLE_DEVICES`), and the memory query maps it to the physical device
before asking nvidia-smi — so on a shared scheduler node that hands out, say,
GPU 2, the report describes the card the job will actually run on rather than
some other job's.

```bash
# Dry run: "would this job fit here?" — prints the report and exits
python -m reconstruction.run_iterative_recon data/scans/Scan_1510 \
    --backend astra --preflight-only

# Force past an INSUFFICIENT verdict (expect OOM)
... --skip-preflight
```

## Plots & experiment logging (all backends)

Every reconstruction writes a set of PNGs next to the output volume
(`<output>_plots/`): orthogonal central slices on physical mm axes, an HU
histogram, a sinogram preview, the projection diagnostics below, and — for
backends with an eval loop (TIGRE, the learned backend) — a convergence
curve. `--no-plots` disables this.

**Projection diagnostics (all backends).** Every run evaluates its
reconstruction against the measured *evaluation projection* (the central
angle): `diag/ssim`, `diag/psnr`, `diag/mse`, a local-SSIM heatmap, and a
projection power-spectrum figure. Iterative backends (TIGRE, voxel) emit
them over the iterations from their own forward projections; single-shot
backends (FDK, ASTRA) get them once, by forward-projecting the final volume
through the canonical ray tracer. Alongside, the **noise ceiling** — the
best SSIM/PSNR any reconstruction can honestly reach, and the σ²-equivalent
MSE floor — is measured from a second independent measurement of the same
line integrals: the other acquisition phase when the scan has one (e.g.
acq-01 frames), else the neighbouring projection (conservative), and is
printed, logged per-step for chart overlay, and drawn into the heatmap's
ceiling panel and the power spectrum's noise floor. All projection
diagnostics are restricted to the detector rows whose rays stay inside the
reconstruction z-slab — outer rows integrate through matter the volume does
not contain and would score FOV truncation, not the reconstruction.
By default the evaluation projection **stays in** the reconstruction
(diagnostic); pass `--withhold-eval` to remove it from the input and turn
the diag metrics into true held-out validation. With `--wandb`, the
finished volume is additionally logged as a scrollable axial-slice
sequence (`recon_slices`).

**Data budget (all backends).** Iteration counts do not compare — an OS-SART
iteration sweeps the whole sinogram, a voxel-trainer iteration touches one
random ray batch, and FDK has no iterations at all. Every driver therefore
reports how many times each measurement (one detector pixel at one angle) was
used on average, and what share of the data was touched at all:

```
Data budget: 1.00 visits per measurement (196.2 M measurements, 100.0% used at
least once) — single backprojection pass over every measurement          [FDK]
Data budget: 100.00 visits per measurement (196.2 M measurements, 100.0% used at
least once) — each iteration uses every measurement exactly once   [ASTRA/TIGRE]
Data budget: 1.25 visits per measurement (196.2 M measurements, 71.3% used at
least once) — 20000 iterations x 12288 rays, sampled with replacement   [voxel]
```

**1.00 visits = one full pass = exactly one SIRT / OS-SART iteration**, which
is what makes the families comparable. Two subtleties the numbers capture:
the learned backend samples *with replacement*, so its coverage is
`1 - e^-visits` (one visit on average means 63% of the data seen, not 100%) —
decisive in the sub-one-visit regime, where a run provably never looked at
most of the sinogram; and when cross-validation stops early, the volume that
gets saved is the peak-SSIM one, so the budget credits `best_iter` and
reports the iterations the run burned separately. Withheld projections
(`--withhold-eval`) leave the measurement pool.

Logged to W&B under the same keys for every backend — `data/visits`,
`data/coverage`, `data/measurements`, `data/sampling` (+ `data/iterations_run`
/ `data/iterations_saved` / `data/rays_per_batch` where they apply). The
learned backend additionally streams `train/data_visits` per step, usable as
a batch-size-independent x-axis. Implementation: `ct_core/data_budget.py`.

The same figures (plus native live charts: training loss / LR per step for
the learned backend, per-eval diag metrics for TIGRE and voxel) can be
logged to **Weights & Biases**, strictly opt-in:

```bash
export WANDB_PROJECT=my-ct-project        # or pass --wandb-project
python -m reconstruction.run_learned_recon data/scans/Scan_1510 --wandb
```

**Privacy** (this is a public repository): no project, entity, API key, or
path is hardcoded anywhere — project/entity come from flags or the
`WANDB_PROJECT`/`WANDB_ENTITY` env vars, auth from `wandb login` /
`WANDB_API_KEY`. The uploaded run config is a whitelist of geometry and
algorithm numbers; the scan is identified by its folder basename only, and
the raw scan.xml header (site/hardware metadata) is never uploaded. W&B's
implicit capture channels — console (stdout echoes local paths), code, and
git metadata — are disabled. Runs land in your own W&B project under your
account's privacy settings; keep that project private if the scans are
sensitive. Logging is best-effort: any W&B failure prints a notice and the
reconstruction continues. Use `--wandb-mode offline` on air-gapped nodes and
`wandb sync` later.

## Usage

```bash
# Standard FDK (BHC + ring correction + two-point HU are all defaults)
python -m reconstruction.run_fdk_recon data/scans/Scan_1988 \
    --fov-xy 93.5 --fov-z 70

# Add bone BHC (Joseph & Spital two-pass)
python -m reconstruction.run_fdk_recon data/scans/Scan_1988 \
    --bone-bhc --fov-xy 93.5 --fov-z 70

# ROI reconstruction (mouse lung)
python -m reconstruction.run_fdk_recon data/scans/Scan_1510 --roi auto

# Iterative (ASTRA SIRT)
python -m reconstruction.run_iterative_recon data/scans/Scan_1988 \
    --backend astra --algorithm SIRT3D_CUDA --iterations 100
```

Run `--help` for full argument lists.

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bhc-coeffs c1 c2` | `0.856 0.21` | Sinogram-domain water BHC polynomial (80 kVp) |
| `--no-bhc` | | Disable BHC |
| `--bone-bhc` | off | Two-pass bone BHC (Joseph & Spital, FDK only) |
| `--bone-bhc-threshold` | 1500 | HU threshold for bone segmentation |
| `--bone-bhc-hu` | 3100 | Monochromatic bone HU (from scan.xml `BoneHU`) |
| `--ring-correction` | on | Sinogram-space ring artifact correction |
| `--roi auto` | off | ROI from SubVolumeCoordinates.xml |
| `--rays-per-batch` | `auto` | Learned backend: batch sized from free VRAM |
| `--compile` | `off` | Learned backend: fuse renderer kernels (needs sm_70+) |

## Installation

```bash
pip install -e .                # Core (FDK, requires PyTorch)
pip install astra-toolbox       # Optional: ASTRA iterative
```

## Scanner Specifics

Tailored for the **GE eXplore CT 120**: cone-beam geometry, VFF projections, `scan.xml` metadata, `bright.vff`/`dark.vff` flat-field. Algorithms are general-purpose — adapting to other scanners requires only changing geometry and file I/O.
