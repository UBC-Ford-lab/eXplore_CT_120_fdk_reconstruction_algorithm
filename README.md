# eXplore CT 120 Reconstruction

Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner. Three
algorithm families, each in its own subfolder: **fdk** (analytic), **iterative**
(ASTRA: SIRT, CGLS; TIGRE: OS-SART, SART, SIRT, MLEM), and
**learning_based_iterative** (differentiable projector + gradient descent;
currently a dense voxel grid).

## Pipeline

```
Raw projections → flat-field + log → BHC → air normalization → ring correction
  → FDK: cone-weight + ramp filter + Parker + backprojection
    OR iterative: ASTRA SIRT / TIGRE OS-SART
  → [bone BHC (FDK only): segment → forward-project → re-reconstruct]
  → physics HU → two-point calibration (air→-1000, water→0) → VFF
```

BHC (water, 80 kVp), air normalization and ring correction are on by default. HU calibration measures air and water/tissue directly from the reconstructed volume (standard CT two-point formula) — self-calibrating regardless of filter or BHC settings.

**One definition per stage.** `apply_bhc` and `soft_clamp_transmission` are
written to work on numpy arrays *and* torch tensors, so FDK's fused GPU path
and the chunked numpy path used by ASTRA/TIGRE/voxel call the same function
rather than carrying parallel implementations. FDK itself is a single
preprocessing pass followed by a single filtering pass; it previously also had
a fused "single-pass" variant, which duplicated both blocks and was the reason
a correction could be wired into one path and silently miss the other.

### Air normalization (`--air-normalization`, default on)

An object-free ray must read `p = -ln(I/I_0) = 0`. On Scan_1510 it does not:
the never-shadowed columns sit at **-0.012** on average and drift from -0.002
to -0.025 across the scan, almost perfectly linearly in frame index (residual
sd 0.0025 after a straight-line fit). That is source/detector gain drift over
the scan duration — the same ~2% the conjugate-ray audit found.

Subtracting one number per projection is the *exact* inverse, not a fudge.
Drift is multiplicative in intensity, so a true flux of `g*I_0` gives

```
p_measured = -ln(I / I_0) = p_true - ln g
```

— a constant additive offset on every ray of that frame, whatever it passed
through. Measured effect on Scan_1510 (ds3, 127 object-free columns):

| | air level | frame-to-frame spread | object band |
|---|---|---|---|
| off | -0.01191 | 0.01753 | 0.13913 |
| on | +0.00042 | 0.00046 | 0.15128 |

It is orthogonal to ring correction — one is a per-frame scalar, the other a
static per-column pattern, and ring correction's 51-px median smoothing does
not touch the former. It runs first, so the ring profile is estimated from
time-consistent frames.

**This changes absolute HU.** Every line integral rises by the removed offset
(+8.7% on the object band here), so reconstructed mu and HU rise with it.
That is the point — the old level was set by an uncorrected air offset — but
it means numbers from earlier runs are not comparable. `--no-air-normalization`
is the exact revert path.

Two limits worth knowing. A **static lateral profile** (span 0.016 on
Scan_1510) survives this and is deliberately left alone: a per-frame scalar is
provably the inverse of gain drift, whereas removing a lateral shape means
extrapolating from the margins to underneath the object, where the physical
cause is not established. And a scan whose object fills the detector has no
air reference — the correction reports that it skipped rather than inventing
an offset.

### Soft-clip sharpness (`--soft-clip-sharpness`, default 200)

Transmission is kept inside `(0, 1.05]` by softplus clamps rather than
`np.clip`, so the ramp filter never meets a derivative corner. But softplus
leaks below its knee —

```
softplus(x)/s = x/s + ln(1 + e^-x)/s
```

— and the leak only decays exponentially in `s x (distance from the knee)`.
At the historical `s = 50` this was not a rounding detail. Air transmission
sits right on the knee's shoulder (mean T = 1.014, sd 0.024; 70.8% of air
pixels above T = 1.0 and 7.65% above 1.05 at full resolution), so the leak
lands squarely on it:

| sharpness | air bias | object bias | air-norm offset error |
|---|---|---|---|
| 50 | +0.00487 | +0.000028 | **+0.00271** |
| 100 | +0.00183 | +0.000000 | — |
| **200** | **+0.00111** | 0.000000 | **+0.00000** |
| 500 | +0.00093 | 0.000000 | +0.00000 |
| hard clip | +0.00090 | 0.000000 | +0.00000 |

The hard-clip row is the irreducible part — the truncation the clamp is
actually *for*. At `s = 50`, **82% of the bias was leak, not clamping.**

It mattered because air normalization estimates its offset as a median over
air pixels. The median is immune to the truncation (only the 7.65% tail is
clipped, far from the median) but not to the leak, which shifts the whole
distribution: at `s = 50` the offset came out -0.00887 against a true -0.01158,
so the object was **under-corrected by 0.0027 in p, ~1.8% of its line
integral**. Measured on Scan_1510 at ds3, moving to 200 raised the object band
from 0.15128 to 0.15387 and the recovered drift from 0.01769 to 0.02065.

Sharpness is shared with the FLOOR clamp, which is the other reason to raise
it: at 50 that clamp starts distorting any ray above `p ~ 0.92` (a `p = 3` ray
came out 0.032 low, `p = 4` came out 0.31 low). Scan_1510 tops out at 0.79 so
it never fired, but a denser specimen or a metal implant would have crossed it
silently. At 200 the onset moves to `p ~ 2.3`.

Sharpening does **not** reintroduce ringing — measured, not assumed. Ramp-
filtered high-band power (> 0.7 x Nyquist) relative to no clamp at all:
`s=50` 0.914x, `s=200` 0.993x, `s=500` 0.998x, hard clip 0.998x. Every variant
*removes* high-frequency energy; none manufactures it, because the clamp fires
on isolated noise excursions in air rather than on a coherent edge. The old
`s = 50` was quietly low-passing 8.6% of the sinogram's high-band power.

Pass `--soft-clip-sharpness 50` to reproduce pre-2026-08-14 line integrals.

## Structure

Every algorithm-independent stage lives in `ct_core` and is shared by all
backends; a reconstruction algorithm is a drop-in replacement for any other:

```
run_fdk_recon.py / run_iterative_recon.py /   thin drivers (backend flags only)
run_learned_recon.py
  └─ ct_core/pipeline.py    shared CLI args, prepare_scan() → ScanContext,
                            detector-psi calibration JSON, save_outputs()
       ├─ ct_core/scan_setup.py      scan.xml, projections, geometry, VFF export
       ├─ ct_core/preprocessing.py   flat-field+log, transmission clamp, BHC,
       │                             air norm., ring corr., downsample
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

## Reconstruction domain vs export ROI (iterative + learned)

A detector pixel measures ∫μ ds along the **whole** ray — through the animal,
the bed, the cage. A backend that *fits* that measurement (SIRT, OS-SART, the
learned family) can only reproduce it if its volume covers everything the ray
crossed. Reconstruct a smaller box and the missing attenuation does not go
away:

```
measured        p = A_dom·x_dom + A_out·x_out
your model          A_dom·x
least squares   x = x_true + A_dom⁺(A_out·x_out)
```

That second term is "reconstruct the cage, but you may only put it inside the
box" — a smooth, low-frequency cup across the **entire** reconstruction, not a
boundary artifact. On Scan_1510 it is ≈ **96 HU of DC bias**, with individual
rays demanding 400–480 HU. Edges survive it; the HU scale does not, so it is
invisible by eye and lethal to a histogram.

**FDK is exempt.** It filters the full-width projections and then backprojects
into whatever grid you ask for, so its `--roi` never enters a forward model —
it is purely an output crop. That is why the auto domain is on by default for
the iterative and learned drivers only, and why the vendor's own volume is an
ROI-shaped crop of a full-FOV FDK.

So those two drivers separate the two boxes:

| | what it is | set by |
|---|---|---|
| **reconstruction domain** | region the forward model must cover | `--model-domain` (default `auto`) |
| **export ROI** | region written to the VFF | `--roi` (default: the whole domain) |

`--model-domain auto` measures the support from the projections themselves:
the outermost detector channel above the air noise floor, converted to mm at
isocentre, then clamped by two hardware limits — the **fan** (nothing outside
it is measurable) and the **cone** (no ray reaches beyond it, so voxels past
it can never receive a gradient; on Scan_1510's old `fov_z 120` default that
was 347 M of 576 M parameters, dead but still carrying 4× Adam state). Every
ambiguous case widens the domain, never narrows it — a domain is only ever
wrong for being too small. On Scan_1510:

```
Attenuating support measured from the projections:
  16 views sampled
  transaxial: radius 40.63 mm at isocentre (outermost shadow 39.63 mm
              + 1.0 mm margin, threshold 0.0890; detector reaches 43.76 mm)
  axial:      z in [-31.63, 31.63] mm (object runs past both ends —
              cone-limited; cone reaches +-31.63 mm)
  reconstruction grid: 1083 x 1083 x 844 = 989.9 M voxels
```

which independently reproduces the `extent_xy: 88.0 / half_extent_z: 29.0`
pinned by hand in muNeRF's `configs/scan_1510_h100_base.yaml`.

Overrides: `--model-domain off` reverts to the old behaviour (grid = `--roi`
/`--fov`), and `--model-domain 88 29` pins it in muNeRF's config units
(`EXTENT_XY HALF_Z`).

**The domain costs memory** — it is the honest size of the problem, and it is
routinely several times the ROI. Preflight sizes it before anything is
allocated and says so. A larger `--voxel-xy`/`--voxel-z` keeps the domain and
costs resolution; `--model-domain off` keeps the resolution and puts the bias
back.

**Export.** `--roi auto` writes only the scan's `SubVolumeCoordinates.xml`
box; six numbers write that box; no `--roi` writes the whole domain. When
`--roi auto` finds no XML these drivers fall back to writing the full volume
(for FDK it stays an error, since there `--roi` decides what is *computed*).
The crop happens **before** HU calibration, so the two-point fit reads air and
tissue from the voxels being shipped rather than from a domain padded out with
bed and cage — and before the plots, so every figure shows the delivered
volume. The forward-projection diagnostics run *before* the crop, on the whole
domain, because that is the projection the measurement should be compared to.

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
diagnostics — including the noise ceiling, so both sides are measured on the
same pixels — are restricted to the **covered detector window**
(`covered_detector_window`): the rows whose rays stay inside the
reconstruction z-slab, and the columns whose rays cross the in-plane FOV
cylinder. Outside it the forward projection is exactly zero by construction,
so comparing there scores FOV truncation rather than the reconstruction. The
crop is reported in the run log, e.g. `rows [0, 765) of 765 and columns
[39, 1128) of 1166`. How much it removes depends entirely on the grid: ~7% of
the detector width for a measured `--model-domain` (r = 40.6 mm on Scan_1510),
but **~58%** for an FDK run on Scan_1510's `--roi auto` grid, where the volume
does not extend under most of the fan.

That FDK figure is not `r` alone. The ROI is off-centre — origin
`(2.71, -5.53, 0.19) mm`, i.e. 6.16 mm from the axis — and an off-centre grid
sweeps relative to the source as the gantry turns, so the window is the union
over angles and uses `r + |origin_xy|` = 12.38 + 6.16 = 18.5 mm rather than
12.4 mm. Assuming a centred grid would predict a 72% crop and would wrongly
discard columns that the ROI really does pass under at some angles.

Note this fixes only the columns the model *cannot* predict. Air columns
*inside* the domain can still score badly for a different reason: SSIM's
luminance term is a ratio of local means, so where the true signal is ~0 it
is dominated by whatever residual offset flat-fielding left behind, and a
`mu >= 0` model cannot follow a measured air level that sits below zero.
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
| `--ring-correction` | on | Sinogram-space ring artifact correction (static per-COLUMN pattern) |
| `--air-normalization` | on | Per-projection air level from object-free columns (per-FRAME scalar) |
| `--soft-clip-sharpness` | 200 | Softplus transmission-clamp sharpness (50 = pre-2026-08-14) |
| `--roi auto` | off | ROI from SubVolumeCoordinates.xml (FDK: the grid; iterative/learned: the export crop) |
| `--model-domain` | `auto` | Iterative/learned: region the forward model must cover, measured from the projections (`off` / `EXTENT_XY HALF_Z`) |
| `--rays-per-batch` | `auto` | Learned backend: batch sized from free VRAM |
| `--compile` | `off` | Learned backend: fuse renderer kernels (needs sm_70+) |

## Installation

```bash
pip install -e .                # Core (FDK, requires PyTorch)
pip install astra-toolbox       # Optional: ASTRA iterative
```

## Scanner Specifics

Tailored for the **GE eXplore CT 120**: cone-beam geometry, VFF projections, `scan.xml` metadata, `bright.vff`/`dark.vff` flat-field. Algorithms are general-purpose — adapting to other scanners requires only changing geometry and file I/O.
