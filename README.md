# eXplore CT 120 FDK Reconstruction Algorithm

GPU-accelerated FDK (Feldkamp-Davis-Kress) cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner.

## Overview

This package provides:
- **`fdk.py`** — `FDKReconstructor` class with GPU-accelerated cone-beam backprojection, ramp filtering (Ram-Lak / Shepp-Logan / Cosine / Hamming), Parker short-scan weighting, and Hounsfield Unit calibration.
- **`run_recon_on_vff_file.py`** — CLI script that loads VFF projections, runs the full FDK pipeline, and saves calibrated VFF output.
- **`ct_core/`** — Core utilities for VFF file I/O, bright/dark field calibration, and TIFF export.

## Installation

```bash
pip install -e .
```

Or use as a git submodule (no install needed — just ensure the parent repo root is on `sys.path` or installed as a package):

```bash
git submodule add https://github.com/UBC-Ford-lab/eXplore_CT_120_fdk_reconstruction_algorithm.git reconstruction
```

## Usage

```python
from reconstruction.fdk import FDKReconstructor
from reconstruction.ct_core.vff_io import VFFDataset
from reconstruction.ct_core.calibration import load_calibration_fields, MU_WATER_80KV
```

## License

MIT
