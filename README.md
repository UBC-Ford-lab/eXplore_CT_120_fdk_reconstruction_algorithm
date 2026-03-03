# eXplore CT 120 Reconstruction

Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner. Supports FDK (analytic), ASTRA (SIRT, CGLS), and TIGRE (OS-SART) backends, all producing calibrated HU volumes from raw VFF projections.

## Package Structure

```
reconstruction/
├── fdk.py                      # FDKReconstructor (PyTorch, GPU-accelerated)
├── astra_iterative.py          # ASTRAReconstructor (optional, requires astra-toolbox)
├── tigre_iterative.py          # TIGREReconstructor (optional, requires TIGRE)
├── run_fdk_recon.py            # CLI: FDK reconstruction
├── run_iterative_recon.py      # CLI: iterative reconstruction (ASTRA / TIGRE)
├── ct_core/
│   ├── vff_io.py               # VFF file I/O and VFFDataset loader
│   ├── calibration.py          # Flat-field correction, HU calibration
│   ├── preprocessing.py        # Shared sinogram preprocessing
│   ├── scan_setup.py           # Shared data-loading and geometry utilities
│   └── tiff_converter.py       # VFF to TIFF export
└── pyproject.toml
```

## Installation

```bash
# Core (FDK only -- requires PyTorch)
pip install -e .

# Optional: ASTRA iterative
pip install astra-toolbox

# Optional: TIGRE iterative (must build from source)
git clone --depth=1 https://github.com/CERN/TIGRE.git /tmp/TIGRE
pip install --no-build-isolation /tmp/TIGRE
```

## CLI Usage

### FDK

```bash
python -m reconstruction.run_fdk_recon data/scans/Scan_1681

python -m reconstruction.run_fdk_recon data/scans/Scan_1681 \
    --filter-type hamming --filter-cutoff match \
    --voxel-xy 0.075 --voxel-z 0.075 --fov-xy 45 --fov-z 120
```

### Iterative (ASTRA / TIGRE)

```bash
# ASTRA SIRT, 100 iterations with non-negativity
python -m reconstruction.run_iterative_recon data/scans/Scan_1681 \
    --backend astra --algorithm SIRT3D_CUDA --iterations 100 --min-constraint 0.0

# TIGRE OS-SART, 100 iterations
python -m reconstruction.run_iterative_recon data/scans/Scan_1681 \
    --backend tigre --algorithm ossart --iterations 100
```

ASTRA requires the full sinogram on GPU; use `--downsample` for large volumes. TIGRE handles GPU memory splitting internally.

Run `--help` on either script for full argument lists.

## Scanner Specifics

Tailored for the **GE eXplore CT 120**: cone-beam geometry, VFF projections, `scan.xml` metadata, `bright.vff`/`dark.vff` calibration. The reconstruction algorithms are general-purpose -- adapting to other scanners requires only changing geometry parameters and file I/O.
