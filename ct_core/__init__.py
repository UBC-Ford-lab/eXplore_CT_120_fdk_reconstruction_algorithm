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
from . import errors
from .errors import (
    ReconstructionError,
    ScanDataError,
    ConfigError,
    PreflightAbort,
    cli_main,
)

# Convenience re-exports of commonly used functions
from .vff_io import read_vff, read_vff_header, read_vff_data, VFFDataset, write_vff
from .preprocessing import (
    preprocess_sinogram,
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
        MU_AIR,
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

# Stopping rules, shared by every iterative and learning-based backend.
from . import early_stop
from .early_stop import (
    STOP_METRICS,
    EarlyStopper,
    HoldoutScorer,
    LCurve,
    PlateauLRReducer,
    StoppingRules,
    metrics_dict,
    plot_convergence,
    resolve_holdout_index,
    solution_norm,
    write_metrics,
)

__all__ = [
    # Errors (raised by the library; drivers turn them into exit codes)
    'errors',
    'ReconstructionError',
    'ScanDataError',
    'ConfigError',
    'PreflightAbort',
    'cli_main',
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
    'MU_AIR',
    # TIFF
    'save_vff_to_tiff',
    # Preprocessing
    'preprocessing',
    'preprocess_sinogram',
    'ring_artifact_correction',
    'downsample_projections',
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
    # Stopping rules
    'early_stop',
    'STOP_METRICS',
    'EarlyStopper',
    'HoldoutScorer',
    'LCurve',
    'PlateauLRReducer',
    'StoppingRules',
    'metrics_dict',
    'plot_convergence',
    'resolve_holdout_index',
    'solution_norm',
    'write_metrics',
]
