"""
CT Core Library

Core utilities for CT reconstruction project including:
- VFF file I/O (vff_io)
- HU calibration (calibration)
- TIFF conversion (tiff_converter)
"""

from . import vff_io
from . import preprocessing
from . import utils

# Convenience re-exports of commonly used functions
from .vff_io import read_vff, read_vff_header, read_vff_data, VFFDataset, write_vff
from .preprocessing import (
    preprocess_sinogram,
    apply_bhc,
    ring_artifact_correction,
    downsample_projections,
)
from .utils import query_gpu_memory

# Optional modules with heavier dependencies (xmltodict, cv2, imageio)
# Wrapped so consumers that only need vff_io aren't blocked.
try:
    from . import calibration
    from .calibration import (
        parse_calibration_from_xml,
        load_calibration_fields,
        flat_field_correction,
        log_transform_transmission,
        MU_WATER_80KV,
        MU_WATER_80KV_NO_BHC,
        MU_WATER_80KV_WITH_BHC,
        MU_AIR,
        default_mu_water,
    )
except ImportError:
    pass

try:
    from . import tiff_converter
    from .tiff_converter import save_vff_to_tiff
except ImportError:
    pass

try:
    from . import scan_setup
    from .scan_setup import (
        auto_detect_scan_folder,
        load_scan_data,
        build_geometry,
        postprocess_and_save,
    )
except ImportError:
    pass

# Shared driver-level pipeline (depends on scan_setup's heavier deps)
try:
    from . import pipeline
    from .pipeline import (
        ScanContext,
        add_common_args,
        prepare_scan,
        resolve_detector_psi,
        measure_detector_psi,
        resolve_or_measure_detector_psi,
        save_outputs,
    )
except ImportError:
    pass

# Half-scan geometry self-calibration (needs torch)
try:
    from . import geometry_selfcal
    from .geometry_selfcal import estimate_psi_halfncc
except ImportError:
    pass

__all__ = [
    # Modules
    'vff_io',
    'calibration',
    'tiff_converter',
    'scan_setup',
    # VFF I/O
    'read_vff',
    'read_vff_header',
    'read_vff_data',
    'VFFDataset',
    'write_vff',
    # Calibration
    'parse_calibration_from_xml',
    'load_calibration_fields',
    'flat_field_correction',
    'log_transform_transmission',
    'MU_WATER_80KV',
    'MU_WATER_80KV_NO_BHC',
    'MU_WATER_80KV_WITH_BHC',
    'MU_AIR',
    # TIFF
    'save_vff_to_tiff',
    # Preprocessing
    'preprocessing',
    'preprocess_sinogram',
    'apply_bhc',
    'ring_artifact_correction',
    'downsample_projections',
    # HU helpers
    'default_mu_water',
    # System utilities
    'utils',
    'query_gpu_memory',
    # Scan setup utilities
    'auto_detect_scan_folder',
    'load_scan_data',
    'build_geometry',
    'postprocess_and_save',
    # Shared driver pipeline
    'pipeline',
    'ScanContext',
    'add_common_args',
    'prepare_scan',
    'resolve_detector_psi',
    'measure_detector_psi',
    'resolve_or_measure_detector_psi',
    'save_outputs',
    # Geometry self-calibration
    'geometry_selfcal',
    'estimate_psi_halfncc',
]
