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

Sinogram preprocessing (flat-field, BHC, ring correction) is shared across all backends. HU calibration measures air and water/tissue directly from the reconstructed volume (standard CT two-point formula). Self-calibrating — works regardless of BHC or filter settings.

## Usage

```bash
# FDK with water BHC
python -m reconstruction.run_fdk_recon data/scans/Scan_1988 \
    --bhc-coeffs 0.856 0.21 --fov-xy 93.5 --fov-z 70 --total-angle 193.00006

# Add bone BHC (Joseph & Spital two-pass)
python -m reconstruction.run_fdk_recon data/scans/Scan_1988 \
    --bhc-coeffs 0.856 0.21 --bone-bhc --fov-xy 93.5 --fov-z 70 --total-angle 193.00006

# ROI reconstruction (mouse lung)
python -m reconstruction.run_fdk_recon data/scans/Scan_1510 \
    --bhc-coeffs 0.856 0.21 --roi auto

# Iterative (ASTRA SIRT) with BHC
python -m reconstruction.run_iterative_recon data/scans/Scan_1988 \
    --bhc-coeffs 0.856 0.21 --backend astra --algorithm SIRT3D_CUDA --iterations 100
```

Run `--help` for full argument lists.

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bhc-coeffs c1 c2` | None | Sinogram-domain water BHC polynomial |
| `--bone-bhc` | off | Two-pass bone BHC (Joseph & Spital) |
| `--bone-bhc-threshold` | 1500 | HU threshold for bone segmentation |
| `--bone-bhc-hu` | 3100 | Monochromatic bone HU (from scan.xml `BoneHU`) |
| `--calibration-method` | `two_point` | HU calibration method |
| `--ring-correction` | on | Sinogram-space ring artifact correction |
| `--roi auto` | off | ROI from SubVolumeCoordinates.xml |

## Installation

```bash
pip install -e .                # Core (FDK, requires PyTorch)
pip install astra-toolbox       # Optional: ASTRA iterative
```

## Scanner Specifics

Tailored for the **GE eXplore CT 120**: cone-beam geometry, VFF projections, `scan.xml` metadata, `bright.vff`/`dark.vff` flat-field. Algorithms are general-purpose — adapting to other scanners requires only changing geometry and file I/O.
