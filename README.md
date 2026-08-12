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

Standalone (pre-)calibration, e.g. on a cluster GPU node before submitting
long jobs:

```bash
python -m reconstruction.run_geometry_calibration data/scans/Scan_1510
python -m reconstruction.run_geometry_calibration data/scans/Scan_1510 --force  # re-measure
```

## Plots & experiment logging (all backends)

Every reconstruction writes a set of PNGs next to the output volume
(`<output>_plots/`): orthogonal central slices on physical mm axes, an HU
histogram, a sinogram preview, and — for backends with a holdout
(TIGRE crossval, the learned backend) — a convergence curve. `--no-plots`
disables this.

The same figures (plus native live charts: training loss / LR / holdout
metrics per step for the learned backend, per-eval SSIM/PSNR/MSE for TIGRE
crossval) can be logged to **Weights & Biases**, strictly opt-in:

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

## Installation

```bash
pip install -e .                # Core (FDK, requires PyTorch)
pip install astra-toolbox       # Optional: ASTRA iterative
```

## Scanner Specifics

Tailored for the **GE eXplore CT 120**: cone-beam geometry, VFF projections, `scan.xml` metadata, `bright.vff`/`dark.vff` flat-field. Algorithms are general-purpose — adapting to other scanners requires only changing geometry and file I/O.
