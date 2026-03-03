"""
Reconstruction package — cone-beam CT reconstruction with ct_core utilities.

Supports FDK (analytic) and optionally ASTRA-based iterative reconstruction.
Re-exports key items from ct_core and fdk for convenience.
"""

from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES

try:
    from .astra_iterative import ASTRAReconstructor, SUPPORTED_ALGORITHMS as SUPPORTED_ITERATIVE_ALGORITHMS
except ImportError:
    pass

try:
    from .tigre_iterative import TIGREReconstructor, SUPPORTED_TIGRE_ALGORITHMS
except ImportError:
    pass

from .ct_core import (
    vff_io,
    calibration,
    tiff_converter,
    read_vff,
    read_vff_header,
    read_vff_data,
    VFFDataset,
    write_vff,
    parse_calibration_from_xml,
    load_calibration_fields,
    flat_field_correction,
    log_transform_transmission,
    convert_to_hounsfield_units,
    MU_WATER_80KV,
    MU_AIR,
    PHANTOM_CALIBRATION,
    fit_hu_calibration,
    save_vff_to_tiff,
)
