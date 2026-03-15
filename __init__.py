"""
Reconstruction package — cone-beam CT reconstruction with ct_core utilities.

Supports FDK (analytic) and optionally ASTRA-based iterative reconstruction.
Re-exports key items from ct_core and fdk for convenience.
"""

# FDK requires calibration (xmltodict) — wrap so lightweight consumers
# (e.g. create_sinogram_dataset.py → vff_io only) aren't blocked.
try:
    from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES
except ImportError:
    pass

try:
    from .astra_iterative import ASTRAReconstructor, SUPPORTED_ALGORITHMS as SUPPORTED_ITERATIVE_ALGORITHMS
except ImportError:
    pass

try:
    from .tigre_iterative import TIGREReconstructor, SUPPORTED_TIGRE_ALGORITHMS
except ImportError:
    pass

# Always-available core
from .ct_core import vff_io
from .ct_core.vff_io import read_vff, read_vff_header, read_vff_data, VFFDataset, write_vff

# Optional re-exports (depend on xmltodict, imageio, cv2)
try:
    from .ct_core import calibration
    from .ct_core.calibration import (
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
except ImportError:
    pass

try:
    from .ct_core import tiff_converter
    from .ct_core.tiff_converter import save_vff_to_tiff
except ImportError:
    pass
