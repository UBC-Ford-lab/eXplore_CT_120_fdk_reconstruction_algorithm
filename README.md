# eXplore CT 120 Reconstruction

Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner. Supports FDK (analytic), ASTRA (SIRT, CGLS), and TIGRE (OS-SART) backends.

## Pipeline

```
Raw projections → flat-field + log → BHC → ring correction
  → FDK: cone-weight + ramp filter + Parker + backprojection
    OR iterative: ASTRA SIRT / TIGRE OS-SART
  → [bone BHC (FDK only): segment → forward-project → re-reconstruct]
  → physics HU → two-point calibration (air→-1000, water→0) → VFF
```

BHC (water, 80 kVp) and ring correction are on by default. HU calibration measures air and water/tissue directly from the reconstructed volume (standard CT two-point formula) — self-calibrating regardless of filter or BHC settings.

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
