# eXplore CT 120 FDK Reconstruction Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-accelerated FDK (Feldkamp-Davis-Kress) cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner, implemented in PyTorch.

## Overview

This package implements the complete reconstruction pipeline for cone-beam micro-CT data acquired on a GE eXplore CT 120 scanner. It takes raw VFF projection files and produces calibrated Hounsfield Unit (HU) volumes.

### Pipeline

1. **Flat-field correction** -- normalizes detector response using bright/dark field references
2. **Log transformation** -- converts transmission to line integrals with soft-clipping to prevent Gibbs ringing
3. **Cone-beam weighting** -- applies distance-based weighting for cone-beam geometry
4. **Ramp filtering** -- frequency-domain filtering with selectable windows (Ram-Lak, Shepp-Logan, Cosine, Hamming)
5. **Parker weighting** -- short-scan redundancy correction (automatically skipped for full 360-degree scans)
6. **Backprojection** -- voxel-driven cone-beam backprojection with GPU-accelerated bilinear interpolation
7. **HU calibration** -- physics-based conversion to Hounsfield Units with optional polynomial phantom calibration

All preprocessing, filtering, and backprojection are GPU-accelerated with automatic memory management (dynamic chunk sizing, CPU fallback for large volumes).

## Package Structure

```
reconstruction/                 # Package root (also repo root)
├── __init__.py                 # Re-exports from fdk and ct_core
├── fdk.py                      # FDKReconstructor class
├── run_recon_on_vff_file.py    # CLI reconstruction script
├── ct_core/                    # Core CT utilities
│   ├── __init__.py
│   ├── vff_io.py               # VFF file I/O and VFFDataset loader
│   ├── calibration.py          # Flat-field correction, HU calibration
│   └── tiff_converter.py       # VFF to TIFF export
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

### As a standalone package

```bash
git clone https://github.com/UBC-Ford-lab/eXplore_CT_120_fdk_reconstruction_algorithm.git
cd eXplore_CT_120_fdk_reconstruction_algorithm
pip install -e .
```

### As a git submodule

When used inside a parent project (e.g., [muPIU-Net](https://github.com/UBC-Ford-lab/muPIU-Net-microCT-sinogram-infilling-network)):

```bash
git submodule add https://github.com/UBC-Ford-lab/eXplore_CT_120_fdk_reconstruction_algorithm.git reconstruction
```

No `pip install` needed -- just ensure the parent repo root is on `sys.path` (e.g., via `pip install -e .` on the parent).

## Usage

### CLI -- reconstruct a scan

```bash
python -m reconstruction.run_recon_on_vff_file data/scans/Scan_1681
```

With custom filter settings:

```bash
python -m reconstruction.run_recon_on_vff_file data/scans/Scan_1681 \
    --filter-type hamming \
    --filter-cutoff match \
    --voxel-xy 0.075 \
    --voxel-z 0.075
```

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `data_folder` | *(required)* | Path to folder containing VFF projections and `scan.xml` |
| `--scan-folder` | auto-detected | Path to original scan folder with `bright.vff` / `dark.vff` |
| `--output` | auto-generated | Output VFF filename |
| `--total-angle` | `determined` | Total angular coverage in degrees. By default, reads `IncrementAngle` and `ViewCount` from `scan.xml` and computes the total automatically. Specify a numeric value to override. |
| `--projection-pattern` | auto-detect | Glob pattern for projection files (`proj-*.vff` or `acq*`) |
| `--filter-type` | `hamming` | Ramp filter window: `ramp`, `shepp-logan`, `cosine`, `hamming` |
| `--filter-cutoff` | `match` | Bandwidth as fraction of Nyquist, or `match` to auto-compute `da/dx` |
| `--voxel-xy` | `0.075` | Reconstruction voxel size in the xy plane (mm) |
| `--voxel-z` | `0.075` | Reconstruction voxel size in z (mm) |
| `--fov-xy` | `45` | Field of view in xy (mm). Use `94` for most phantom scanner studies. |
| `--fov-z` | `120.0` | Field of view in z (mm) |
| `--display` | off | Save reconstruction slice PNGs |
| `--bilateral-filter` | off | Apply edge-preserving bilateral filter after HU calibration |
| `--bilateral-sigma-spatial` | `1.5` | Bilateral filter spatial sigma in mm (converted to voxels internally) |
| `--bilateral-sigma-range` | `50.0` | Bilateral filter intensity sigma in HU (edge-preservation threshold) |

### Python API

```python
from reconstruction.fdk import FDKReconstructor
from reconstruction.ct_core.vff_io import VFFDataset
from reconstruction.ct_core.calibration import load_calibration_fields, MU_WATER_80KV

# Load projections
dataset = VFFDataset(data_folder, xml_file, paths_str='proj-*.vff',
                     projection_spacing=0.877)
bright, dark = load_calibration_fields(scan_folder)

# Define geometry (from scan.xml)
geometry = {
    'R_s': 260.0,              # Source-to-isocenter (mm)
    'R_d': 130.0,              # Detector-to-isocenter (mm)
    'da': 0.05,                # Detector pixel size (mm)
    'db': 0.05,
    'vol_shape': (1248, 1248, 300),
    'vol_origin': (0, 0, 0),
    'dx': 0.075,               # Voxel size xy (mm)
    'dz': 0.4,                 # Voxel size z (mm)
    'central_pixel_a': 512.0,
    'central_pixel_b': 256.0,
}

# Reconstruct
recon = FDKReconstructor(
    projections=dataset.projections,
    angles=dataset.angles_rad,
    geometry=geometry,
    source_locations=None,
    folder_name='output_path',
    output_hu=True,
    bright_field=bright,
    dark_field=dark,
    mu_water=MU_WATER_80KV,
    physical_normalization=True,
    filter_type='hamming',
    filter_cutoff=geometry['da'] / geometry['dx'],
    parker_weighting=True,
)
recon.reconstruct(display_volume=False)
# Output: VFF file at output_path.vff
```

### Reading/writing VFF files

```python
from reconstruction.ct_core.vff_io import read_vff, write_vff

header, data = read_vff('volume.vff')  # data shape: (z, y, x)
write_vff('output.vff', {'bits': 16, 'spacing': '0.075 0.075 0.4'}, data)
```

### Exporting to TIFF

```python
from reconstruction.ct_core.tiff_converter import save_vff_to_tiff

save_vff_to_tiff(data, target_directory='tiff_slices/')
```

## Scanner Specifics

This implementation is tailored for the **GE eXplore CT 120** micro-CT scanner:

- Cone-beam geometry with flat-panel detector
- VFF file format for projections and reconstructed volumes
- `scan.xml` metadata (source/detector positions, angular offsets, detector spacing)
- Bright/dark field calibration files (`bright.vff`, `dark.vff`)
- Typical scan parameters: ~410 projections over 360 degrees, 0.05 mm detector pixel pitch

The reconstruction algorithm itself (FDK) is general-purpose. Adapting to other cone-beam CT scanners requires only changing the geometry parameters and file I/O.

## Requirements

- Python >= 3.8
- PyTorch (CUDA recommended for GPU acceleration, CPU fallback supported)
- NumPy
- xmltodict
- imageio
- matplotlib

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wiegmann2026fdk,
  author = {Wiegmann, Falk},
  title = {eXplore CT 120 FDK Reconstruction Algorithm},
  year = {2026},
  url = {https://github.com/UBC-Ford-lab/eXplore_CT_120_fdk_reconstruction_algorithm}
}
```
