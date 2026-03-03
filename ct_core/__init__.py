"""
CT Core Library

Core utilities for CT reconstruction project including:
- VFF file I/O (vff_io)
- HU calibration (calibration)
- TIFF conversion (tiff_converter)
"""

from . import vff_io
from . import calibration
from . import tiff_converter
from . import scan_setup
from . import preprocessing

# Convenience re-exports of commonly used functions
from .vff_io import read_vff, read_vff_header, read_vff_data, VFFDataset, write_vff
from .calibration import (
    parse_calibration_from_xml,
    load_calibration_fields,
    flat_field_correction,
    log_transform_transmission,
    convert_to_hounsfield_units,
    MU_WATER_80KV,
    MU_AIR,
    PHANTOM_CALIBRATION,
    fit_hu_calibration,
)
from .tiff_converter import save_vff_to_tiff
from .preprocessing import preprocess_sinogram
from .scan_setup import (
    auto_detect_scan_folder,
    load_scan_data,
    build_geometry,
    postprocess_and_save,
)

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
    'convert_to_hounsfield_units',
    'MU_WATER_80KV',
    'MU_AIR',
    'PHANTOM_CALIBRATION',
    'fit_hu_calibration',
    # TIFF
    'save_vff_to_tiff',
    # Preprocessing
    'preprocessing',
    'preprocess_sinogram',
    # Scan setup utilities
    'auto_detect_scan_folder',
    'load_scan_data',
    'build_geometry',
    'postprocess_and_save',
]
