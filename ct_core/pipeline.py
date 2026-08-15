"""Shared driver-level plumbing for the reconstruction entry points.

Every reconstruction algorithm in this package — FDK (analytic), ASTRA/TIGRE
(classical iterative), and whatever comes next (learning-based iterative) —
consumes the same inputs and produces the same output:

    scan folder ─→ prepare_scan() ─→ ScanContext ─→ <algorithm> ─→ volume
                                                        │
                            save_outputs() ←────────────┘

The algorithm-independent stages live here, once:

  * ``add_common_args``      — CLI arguments shared by every driver
  * ``prepare_scan``         — scan-folder resolution, projection/flat-field
                               loading, ROI parsing, geometry build, optional
                               detector downsampling (with correct
                               central-pixel index conversion)
  * ``resolve_detector_psi`` — the scan-keyed detector-psi calibration JSON
                               (written by muNeRF's half-scan self-calibration
                               or scripts/detector_psi_from_conjugates.py)
  * ``save_outputs``         — HU calibration + bilateral filter + VFF export

A reconstruction backend only has to honour the volume contract to be a
drop-in replacement: take ``ctx.projections`` (raw counts, (N_angles, N_b,
N_a)), ``ctx.angles`` (radians, FDK convention), ``ctx.geometry`` (dict, see
``scan_setup.build_geometry``), and return a float32 volume of shape
``geometry['vol_shape']`` = (Nx, Ny, Nz) in HU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .scan_setup import (
    _xml_field,
    auto_detect_scan_folder,
    load_scan_data,
    build_geometry,
    parse_crop_boundary,
    postprocess_and_save,
    reconstructible_fov,
)
from .support import (  # noqa: F401 (crop_to_export_roi re-exported for drivers)
    crop_to_export_roi,
    explicit_domain_bounds,
    measure_attenuating_support,
    support_to_domain_bounds,
)
from .errors import ConfigError, ScanDataError
from .paths import calibration_dir
from .preprocessing import downsample_projections
from .vff_io import detector_serial_from_scan
from .wandb_logging import ReconLogger, add_wandb_args  # noqa: F401 (ReconLogger re-exported for drivers)
from .preflight import add_preflight_args, run_preflight  # noqa: F401 (run_preflight re-exported for drivers)


# --------------------------------------------------------------------------
# CLI arguments shared by every reconstruction driver
# --------------------------------------------------------------------------

def add_common_args(parser):
    """Register the algorithm-independent arguments on ``parser``.

    Backend-specific flags (FDK filter settings, TIGRE relaxation, ...) stay
    in the individual drivers.
    """
    parser.add_argument(
        'data_folder',
        help='Path to folder containing projections and scan.xml'
    )
    parser.add_argument(
        '--scan-folder',
        help='Path to original scan folder with bright.vff/dark.vff '
             '(auto-detected if not specified)'
    )
    parser.add_argument(
        '--output',
        help='Output VFF filename (auto-generated from data_folder if not specified)'
    )
    parser.add_argument(
        '--total-angle',
        default='determined',
        help='Total angular coverage in degrees. Default: "determined" (reads '
             'IncrementAngle and ViewCount from scan.xml to compute total angle '
             'automatically). Specify a numeric value to override '
             '(e.g., --total-angle 360.0).'
    )
    parser.add_argument(
        '--projection-pattern',
        default=None,
        help='Glob pattern for projection files (default: auto-detect proj-* or acq*)'
    )
    parser.add_argument(
        '--phase',
        default='00',
        help='Acquisition phase to reconstruct for multi-phase (gated) scans, '
             'e.g. 00 or 01. Selects projection files whose name contains '
             '"-<phase>-" (default: 00). Ignored for sequential proj-* scans.'
    )
    parser.add_argument(
        '--voxel-xy',
        type=float,
        default=0.075,
        help='Reconstruction voxel size in the xy plane in mm (default: 0.075)'
    )
    parser.add_argument(
        '--voxel-z',
        type=float,
        default=0.075,
        help='Reconstruction voxel size in the z plane in mm (default: 0.075)'
    )
    parser.add_argument(
        '--fov-xy',
        default='auto',
        metavar='MM|auto',
        help='Field of view in the xy plane in mm (default: auto — the fan '
             'the detector actually subtends, tightened to the object the '
             'projections show, so the grid is a fact about this scanner and '
             'this scan rather than an assumption about the specimen). '
             'Ignored when --roi or a model domain defines the grid.'
    )
    parser.add_argument(
        '--fov-z',
        default='auto',
        metavar='MM|auto',
        help='Field of view in the z direction in mm (default: auto — the '
             'cone height at the rotation axis, tightened to the object). '
             'Beyond the cone no ray reaches the voxels at all: the historical '
             '120 mm default put two thirds of the grid outside it.'
    )
    parser.add_argument(
        '--roi',
        nargs='+',
        default=None,
        help='ROI-based reconstruction. Use "auto" to load from '
             'SubVolumeCoordinates.xml in the scan folder, or specify 6 values: '
             'x_min x_max y_min y_max z_min z_max (mm, isocenter-centered). '
             'When active, --fov-xy and --fov-z are ignored.'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Downsample projections by this factor before reconstruction '
             '(default: 1). Reduces GPU memory usage. Factor 2 halves each '
             'detector dimension (detector pixel size and central-pixel '
             'indices are converted consistently).'
    )
    parser.add_argument(
        '--bilateral-filter',
        action='store_true',
        help='Apply bilateral filter to calibrated volume '
             '(edge-preserving denoising)'
    )
    parser.add_argument(
        '--bilateral-sigma-spatial',
        type=float,
        default=1.5,
        help='Bilateral filter spatial sigma in mm (default: 1.5). '
             'Converted to voxels using --voxel-xy.'
    )
    parser.add_argument(
        '--bilateral-sigma-range',
        type=float,
        default=50.0,
        help='Bilateral filter intensity sigma in HU (default: 50.0). '
             'Controls edge-preservation threshold.'
    )
    parser.add_argument(
        '--bhc-coeffs',
        nargs='+',
        type=float,
        default=None,
        help='BHC polynomial coefficients [c1, c2, ...] for sinogram-domain '
             'beam hardening correction: p_corrected = c1*p + c2*p^2 + ... '
             'Example: 0.856 0.21 (calibrated from water phantom at 80 kVp). '
             'Default: disabled (no BHC).'
    )
    parser.add_argument(
        '--no-bhc',
        dest='bhc_coeffs',
        action='store_const',
        const=None,
        help='Disable sinogram-domain beam hardening correction'
    )
    parser.add_argument(
        '--cor-mode',
        default='center',
        choices=('center', 'xml'),
        help='Detector centre-of-rotation policy (default: center). '
             '"center" places the rotation axis at the detector geometric '
             'centre — the validated configuration: with the detector-psi '
             'calibration applied, adding the scan.xml CentreOfRotation '
             'offset on top over-corrects (decisive split-tube experiment, '
             '2026-08-13). "xml" restores the legacy scan.xml '
             'CentreOfRotation/CentralSlice values.'
    )
    parser.add_argument(
        '--geometry-autocal',
        action='store_true',
        default=True,
        help='Use the scan-keyed detector geometry calibration (in-plane '
             'rotation psi) measured reference-free from the projections. '
             'Default: on.'
    )
    parser.add_argument(
        '--no-geometry-autocal',
        dest='geometry_autocal',
        action='store_false',
        help='Assume a perfectly square, centred detector — the '
             'pre-2026-08-11 behaviour.'
    )
    parser.add_argument(
        '--ring-correction',
        action='store_true',
        default=True,
        dest='ring_correction',
        help='Enable sinogram-space ring artifact correction (default: on). '
             'Removes fixed-pattern detector column offsets that cause '
             'concentric ring artifacts in the reconstruction.'
    )
    parser.add_argument(
        '--no-ring-correction',
        action='store_false',
        dest='ring_correction',
        help='Disable ring artifact correction.'
    )
    parser.add_argument(
        '--air-normalization',
        action='store_true',
        default=True,
        dest='air_normalization',
        help='Level each projection on its object-free air columns (default: '
             'on). Removes the source gain drift over the scan — a constant '
             'additive offset per frame in the log domain, ~0.023 across '
             'Scan_1510 — which otherwise makes the same physical ray '
             'inconsistent between the start and end of a short scan.'
    )
    parser.add_argument(
        '--no-air-normalization',
        action='store_false',
        dest='air_normalization',
        help='Disable air normalization (the pre-2026-08-14 behaviour; use it '
             'to reproduce older runs, whose absolute HU level was set by an '
             'uncorrected air offset).'
    )
    parser.add_argument(
        '--soft-clip-sharpness',
        type=float,
        default=200.0,
        help='Sharpness of the softplus transmission clamps (default: 200). '
             'The clamps leak below their knee by ln(1+e^-x)/s; at the '
             'historical 50 that put +0.0049 into every air ray and made air '
             'normalization under-correct the object by ~1.8%%. Pass 50 to '
             'reproduce pre-2026-08-14 line integrals.'
    )
    parser.add_argument(
        '--ring-median-width',
        type=int,
        default=51,
        help='Median filter width for ring correction (default: 51, must be '
             'odd). Controls the scale of features removed. '
             'Larger = more aggressive.'
    )
    parser.add_argument(
        '--hu-calibration',
        choices=('auto', 'fixed'),
        default='auto',
        help='How the HU scale is set (default: auto). "auto" fits both the '
             'gain and the offset from the reconstructed volume\'s own '
             'histogram, anchoring air on -1000 HU and the bulk soft-tissue '
             'population on --tissue-hu; it needs no phantom and no scanner '
             'constant, so it survives a source or detector replacement. '
             '"fixed" pins the gain to --mu-water and puts air at zero '
             'attenuation, reproducing the classical one-point map (its '
             'second anchor is water at 0 HU, so --tissue-hu does not apply).'
    )
    parser.add_argument(
        '--mu-water',
        type=float,
        default=None,
        help='Linear attenuation of water in mm^-1, used by '
             '--hu-calibration fixed. Note this is a scanner- AND '
             'spectrum-specific constant that goes stale when the hardware '
             'changes; the historical value for this scanner at 80 kVp was '
             '0.0219.'
    )
    parser.add_argument(
        '--tissue-hu',
        type=float,
        default=None,
        help='Where the bulk soft-tissue anchor lands, in HU (default: 120, '
             'the scale this scanner\'s vendor reports for the same '
             'specimens). This single number sets the gain and is the one '
             'assumption reference-free calibration cannot escape — air pins '
             'the offset exactly, but nothing in the data pins the scale. '
             'Changing it rescales the output linearly. Pass 0 for a WATER '
             'PHANTOM, whose bulk belongs at 0 HU by definition. Applies to '
             '--hu-calibration auto only.'
    )
    parser.add_argument(
        '--save-mu',
        action='store_true',
        default=False,
        help='Also write the uncalibrated attenuation as <output>_mu.npy '
             '(float32, mm^-1). Replaces the old _uncalibrated.vff, which '
             'held already-converted, already-clipped values and was '
             'byte-identical to the calibrated file.'
    )

    parser.add_argument(
        '--withhold-eval',
        action='store_true',
        default=False,
        help='Withhold the evaluation projection (the central angle, the '
             'same one every diagnostic uses) from the reconstruction, '
             'turning the diag/* metrics into true held-out validation. '
             'Default: off — the projection is reconstructed from AND '
             'evaluated against (diagnostic, not validation).'
    )

    # Experiment logging (local PNG plots + optional Weights & Biases)
    add_wandb_args(parser)
    # Machine preflight (GPU/VRAM/RAM fit check before any big allocation)
    add_preflight_args(parser)
    return parser


# --------------------------------------------------------------------------
# Scan preparation (loading + geometry), shared by every driver
# --------------------------------------------------------------------------

@dataclass
class ScanContext:
    """Everything an algorithm needs, in the shared conventions.

    ``projections`` are RAW counts (flat-field correction happens inside the
    backends so FDK can keep its fused GPU path); ``angles`` are radians in
    the FDK convention; ``geometry`` is the dict from
    ``scan_setup.build_geometry`` (already downsample-adjusted when
    ``downsample`` > 1).
    """
    data_folder: str
    scan_folder: str
    projections: np.ndarray
    angles: np.ndarray
    bright_field: Optional[np.ndarray]
    dark_field: Optional[np.ndarray]
    xml_header: dict
    geometry: dict
    roi_bounds: Optional[dict] = None
    total_angle: float = 0.0
    downsample: int = 1
    detector_psi: Optional[dict] = field(default=None)
    # Carried on the context rather than threaded through every diagnostic
    # signature so the noise-ceiling frames are levelled exactly like the
    # sinogram the backend trains on — an unlevelled ceiling would be
    # compared against a levelled reconstruction.
    air_normalization: bool = True

    def default_output_path(self, suffix: str = '_recon') -> str:
        return self.data_folder.rstrip('/') + suffix


def _parse_roi_bounds(args, scan_folder, xml_header, required=True):
    """Resolve --roi into isocenter-centered bounds (or None).

    ``required=False`` (the backends where --roi only crops the OUTPUT) turns
    a missing SubVolumeCoordinates.xml into a fallback to the full volume
    rather than an error: saving everything is a harmless default. It stays
    fatal where --roi defines the reconstruction grid itself (FDK), since
    silently reconstructing the whole FOV instead is a 6.6x surprise.
    """
    if args.roi is None:
        return None
    if args.roi == ['auto']:
        roi_bounds = parse_crop_boundary(scan_folder, xml_header)
        if roi_bounds is None:
            where = os.path.join(scan_folder, 'Volumes', 'SubVolumeCoordinates.xml')
            if not required:
                print(f"\n  --roi auto: no SubVolumeCoordinates.xml at {where}"
                      f"\n  falling back to saving the full reconstruction volume.")
                return None
            raise ScanDataError(
                f"--roi auto: SubVolumeCoordinates.xml not found or invalid "
                f"(looked in {where}). For this backend the ROI defines the "
                f"reconstruction grid, so falling back to the full FOV would "
                f"silently reconstruct a much larger volume; pass explicit "
                f"--roi bounds or drop --roi.")
        return roi_bounds
    if len(args.roi) == 6:
        vals = [float(v) for v in args.roi]
        return {
            'x_min': vals[0], 'x_max': vals[1],
            'y_min': vals[2], 'y_max': vals[3],
            'z_min': vals[4], 'z_max': vals[5],
        }
    raise ConfigError(
        f"--roi takes 'auto' or exactly 6 values "
        f"(x_min x_max y_min y_max z_min z_max), got {len(args.roi)}")


def add_model_domain_args(parser):
    """`--model-domain`, for backends that FIT the projections.

    Only the iterative and learned families call this. FDK does not have a
    forward model, so its reconstruction grid can be any sub-box without
    consequence; SIRT/OS-SART and the learned backends cannot (see
    ct_core.support for the bias this controls).
    """
    parser.add_argument(
        '--model-domain', dest='model_domain', nargs='+', default=['auto'],
        metavar='auto|off|EXTENT_XY HALF_Z',
        help='Region the reconstruction must cover so the forward model is '
             'complete (default: auto — measured from the projections: the '
             'outermost detector channel above the air noise floor, clamped '
             'to the fan and the cone). "off" reverts to the pre-2026-08-14 '
             'behaviour where the grid is just --roi/--fov (fast, but any '
             'matter outside it — bed, cage — is forced into the '
             'reconstruction as a smooth HU bias). Two numbers pin it '
             'explicitly in muNeRF config units, e.g. "88 29".')


def model_domain_enabled(args) -> bool:
    """Whether --model-domain will replace the grid. Cheap, no data needed.

    Answered BEFORE the geometry is built, because 'off' has to keep the old
    meaning of --roi (it defines the grid) while any other setting demotes
    --roi to an export crop.
    """
    spec = getattr(args, 'model_domain', None) or ['auto']
    return not (len(spec) == 1
                and str(spec[0]).lower() in ('off', 'none', 'false'))


def resolve_model_domain(args, projections, geometry, bright_field, dark_field):
    """`--model-domain` -> bounds dict for build_geometry, or None for 'off'."""
    spec = getattr(args, 'model_domain', ['auto']) or ['auto']
    if not model_domain_enabled(args):
        return None, None
    if len(spec) == 2:
        extent_xy, half_z = (float(v) for v in spec)
        print(f"\nModel domain pinned: extent_xy {extent_xy:.1f} mm, "
              f"half_extent_z {half_z:.1f} mm")
        return explicit_domain_bounds(extent_xy, half_z), None
    if len(spec) == 1 and str(spec[0]).lower() == 'auto':
        support = measure_attenuating_support(
            projections, geometry,
            bright_field=bright_field, dark_field=dark_field)
        return support_to_domain_bounds(support), support
    raise ConfigError(
        f"--model-domain takes 'auto', 'off', or two numbers "
        f"(EXTENT_XY HALF_Z), got {' '.join(str(v) for v in spec)!r}")


def _fov_spec(value, flag: str):
    """Parse a `--fov-*` value: ``None`` for 'auto', else the number in mm."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ('auto', 'full'):
        return None
    try:
        mm = float(text)
    except ValueError:
        raise ConfigError(
            f"{flag} takes a number of millimetres or 'auto', "
            f"got {value!r}") from None
    if mm <= 0:
        raise ConfigError(f"{flag} must be positive, got {mm}")
    return mm


def resolve_fov(args, xml_header, projections, *, bright_field=None,
                dark_field=None, measure: bool = True):
    """`--fov-xy` / `--fov-z` -> the two extents build_geometry wants, in mm.

    The historical defaults (45 mm transaxial, 120 mm axial) were neither
    scanner facts nor safe: 45 mm is a mouse, and on this scanner it discards
    a measured field almost twice that wide, while 120 mm is more than twice
    the cone, so two thirds of the axial grid could never receive a ray. Both
    numbers describe a specimen, and they travel to the next specimen wrongly.

    'auto' replaces them with two things that are actually knowable:

      1. the RECONSTRUCTIBLE field — how far the fan and cone reach, from the
         scan.xml geometry alone (``scan_setup.reconstructible_fov``). This is
         the hard bound: outside it there are no measurements.
      2. the ATTENUATING SUPPORT — how far the object actually extends,
         measured from the projections (``ct_core.support``, the same
         measurement the iterative and learned backends already use for their
         model domain). This tightens the box so a small specimen in a large
         field does not pay for the empty part.

    The support is only a tightening, never an extension, and it is skipped
    (``measure=False``) when --roi or a model domain is going to replace the
    grid anyway. An explicit number always wins, and is never clamped: asking
    for a field wider than the fan is a legitimate diagnostic.
    """
    want_xy = _fov_spec(getattr(args, 'fov_xy', 'auto'), '--fov-xy')
    want_z = _fov_spec(getattr(args, 'fov_z', 'auto'), '--fov-z')
    if want_xy is not None and want_z is not None:
        return want_xy, want_z

    n_b, n_a = int(projections.shape[1]), int(projections.shape[2])
    reach = reconstructible_fov(xml_header, n_b, n_a,
                                cor_mode=getattr(args, 'cor_mode', 'center'))
    fov_xy = reach['fov_xy'] if want_xy is None else want_xy
    fov_z = reach['fov_z'] if want_z is None else want_z
    print(f"\nField of view 'auto': the detector subtends "
          f"{reach['fov_xy']:.1f} mm transaxially and {reach['fov_z']:.1f} mm "
          f"axially at the isocentre (magnification "
          f"{reach['magnification']:.3f})")

    if measure:
        # A minimal geometry for the support measurement: it reads detector
        # channels, so it needs only the detector parameters — not a grid.
        R_s = _xml_field(xml_header, 'ObjectPosition')
        pitch = _xml_field(xml_header, 'DetectorSpacing')
        det_geom = {
            'R_s': R_s,
            'R_d': _xml_field(xml_header, 'DetectorPosition') - R_s,
            'da': pitch, 'db': pitch,
            'central_pixel_a': (n_a - 1) / 2.0,
            'central_pixel_b': (n_b - 1) / 2.0,
        }
        try:
            support = measure_attenuating_support(
                projections, det_geom, bright_field=bright_field,
                dark_field=dark_field, verbose=False)
        except Exception as e:                                  # noqa: BLE001
            print(f"  (support measurement unavailable — "
                  f"{type(e).__name__}: {e}; using the full field)")
            return fov_xy, fov_z

        # Symmetric about the axis in xy (the object turns), and in z we keep
        # the box centred on the isocentre so vol_origin stays (0, 0, 0) — an
        # off-centre z box is what --roi and --model-domain are for.
        obj_xy = 2.0 * float(support['radius_xy'])
        obj_z = 2.0 * max(abs(float(support['z_min'])),
                          abs(float(support['z_max'])))
        if want_xy is None and obj_xy < fov_xy:
            print(f"  transaxial: object spans {obj_xy:.1f} mm — tightened "
                  f"from {fov_xy:.1f} mm")
            fov_xy = obj_xy
        if want_z is None and obj_z < fov_z:
            print(f"  axial:      object spans {obj_z:.1f} mm — tightened "
                  f"from {fov_z:.1f} mm")
            fov_z = obj_z
    return fov_xy, fov_z


def apply_cor_policy(geometry: dict, n_b: int, n_a: int,
                     cor_mode: str = 'center', verbose: bool = True) -> dict:
    """Detector centre-of-rotation policy, applied to every backend's
    geometry.

    DEFAULT ('center'): the rotation axis sits at the detector geometric
    centre. The scan.xml CentreOfRotation/CentralSlice offset and the
    detector-psi calibration are nearly degenerate at any single
    off-midplane height, and applying BOTH over-corrects: the decisive
    split-tube experiment (Scan_1510, 2026-08-13) showed psi alone matches
    the vendor reconstruction at the midplane AND at z=+22 mm, while
    psi + XML COR re-splits the off-midplane tube. The XML values stay
    available under ``central_pixel_*_xml`` (and via cor_mode='xml').
    """
    geometry['central_pixel_a_xml'] = geometry['central_pixel_a']
    geometry['central_pixel_b_xml'] = geometry['central_pixel_b']
    if cor_mode == 'center':
        geometry['central_pixel_a'] = (n_a - 1) / 2.0
        geometry['central_pixel_b'] = (n_b - 1) / 2.0
        if verbose:
            print(f"\n  Centre of rotation: detector geometric centre "
                  f"(a={geometry['central_pixel_a']:.1f}, "
                  f"b={geometry['central_pixel_b']:.1f}); scan.xml values "
                  f"(a={geometry['central_pixel_a_xml']:.1f}, "
                  f"b={geometry['central_pixel_b_xml']:.1f}) NOT applied — "
                  f"psi calibration covers the alignment (--cor-mode xml "
                  f"restores the legacy behaviour).")
    elif verbose:
        print("\n  Centre of rotation: scan.xml values (--cor-mode xml).")
    return geometry


def prepare_scan(args, fit_domain: bool = False) -> ScanContext:
    """Run the algorithm-independent front half of every reconstruction.

    Resolves the scan folder, loads projections + flat fields, parses the
    ROI, builds the geometry dict, and (optionally) downsamples the detector
    — identically for every backend.

    ``fit_domain=True`` (the backends that FIT the projections: iterative and
    learned) additionally measures the attenuating support and makes THAT the
    reconstruction grid, demoting --roi to an export crop. FDK leaves it False:
    it filters the full-width projections and backprojects into whatever grid
    it is given, so its ROI never enters a forward model.
    """
    data_folder = args.data_folder

    # Resolve scan folder
    if args.scan_folder:
        scan_folder = args.scan_folder
    else:
        scan_folder = auto_detect_scan_folder(data_folder)

    scan_data = load_scan_data(
        data_folder, scan_folder,
        args.projection_pattern, args.total_angle,
        sub_scan=f'-{args.phase}-',
    )
    projections = scan_data['projections']
    angles = scan_data['angles']
    bright_field = scan_data['bright_field']
    dark_field = scan_data['dark_field']

    # With a model domain, --roi selects what to SAVE and the grid comes from
    # the measured support; without one (FDK, or --model-domain off), --roi IS
    # the grid and keeps its original meaning.
    use_domain = fit_domain and model_domain_enabled(args)
    roi_bounds = _parse_roi_bounds(args, scan_folder, scan_data['xml_header'],
                                   required=not use_domain)

    # The FOV only decides anything when nothing else does: --roi and a model
    # domain both replace the grid outright, and measuring the support for a
    # grid that is about to be discarded would just cost time.
    fov_xy, fov_z = resolve_fov(
        args, scan_data['xml_header'], projections,
        bright_field=bright_field, dark_field=dark_field,
        measure=(roi_bounds is None and not use_domain))

    geometry = build_geometry(
        scan_data['xml_header'],
        fov_xy, fov_z, args.voxel_xy, args.voxel_z,
        roi_bounds=None if use_domain else roi_bounds,
        # Under a model domain this grid is provisional — it exists only to
        # carry the detector parameters into the support measurement, and is
        # replaced below. Printing it would advertise a volume nobody uses.
        verbose=not use_domain,
    )

    apply_cor_policy(geometry, projections.shape[1], projections.shape[2],
                     cor_mode=getattr(args, 'cor_mode', 'center'))

    # Optional detector downsampling (average pooling), applied consistently:
    # detector pixel pitch scales UP by the factor, and the central-pixel
    # indices convert as raw -> pooled: c' = (c - (f-1)/2) / f (the pooled
    # pixel centre sits at the mean of its f raw-pixel centres).
    factor = int(getattr(args, 'downsample', 1) or 1)
    if factor > 1:
        print(f"\nDownsampling projections by factor {factor}...")
        original_shape = projections.shape
        projections = downsample_projections(projections, factor)
        print(f"  {original_shape} -> {projections.shape}")
        if bright_field is not None:
            bright_field = downsample_projections(bright_field, factor)
        if dark_field is not None:
            dark_field = downsample_projections(dark_field, factor)
        geometry['da'] *= factor
        geometry['db'] *= factor
        for key in ('central_pixel_a', 'central_pixel_b',
                    'central_pixel_a_xml', 'central_pixel_b_xml'):
            geometry[key] = (geometry[key] - (factor - 1) / 2.0) / factor

    # ---- reconstruction domain (fitting backends only) --------------------
    # Deliberately AFTER downsampling and the COR policy: the measurement
    # reads detector channels, so it needs the da/db/central_pixel_* that the
    # reconstruction will actually use.
    if use_domain:
        domain_bounds, support = resolve_model_domain(
            args, projections, geometry, bright_field, dark_field)
        if domain_bounds is not None:
            domain_geom = build_geometry(
                scan_data['xml_header'],
                fov_xy, fov_z, args.voxel_xy, args.voxel_z,
                roi_bounds=domain_bounds)
            for key in ('vol_shape', 'vol_origin'):
                geometry[key] = domain_geom[key]
            geometry['model_domain'] = domain_bounds
            geometry['model_domain_support'] = support
            Nx, Ny, Nz = geometry['vol_shape']
            print(f"  reconstruction grid: {Nx} x {Ny} x {Nz} = "
                  f"{Nx * Ny * Nz / 1e6:.1f} M voxels")
        # --roi now selects the sub-box to WRITE, not what to reconstruct.
        geometry['export_roi'] = roi_bounds
        if roi_bounds is None:
            print("  export: full reconstruction domain "
                  "(pass --roi auto to save only the scan's ROI)")

    return ScanContext(
        data_folder=data_folder,
        scan_folder=scan_folder,
        projections=projections,
        angles=(angles.numpy() if hasattr(angles, 'numpy')
                else np.asarray(angles)),
        bright_field=bright_field,
        dark_field=dark_field,
        xml_header=scan_data['xml_header'],
        geometry=geometry,
        roi_bounds=roi_bounds,
        total_angle=scan_data['total_angle'],
        downsample=factor,
        air_normalization=bool(getattr(args, 'air_normalization', True)),
    )


# --------------------------------------------------------------------------
# Detector geometry calibration (the ONE reader of the shared psi JSON)
# --------------------------------------------------------------------------

def resolve_detector_psi(scan_folder, verbose=True,
                         fallback_note="using psi=0 and the XML CoR"
                         ) -> Optional[dict]:
    """Read the scan-keyed detector-psi calibration JSON, if it exists.

    This is the single cross-pipeline interface for the detector in-plane
    rotation: muNeRF's half-scan self-calibration (the validated estimator)
    and scripts/detector_psi_from_conjugates.py both write
    ``data/calibration/detector_psi_<serial>_<scanTag>.json``; every
    reconstruction backend reads it through this function.

    Returns the parsed record (keys: psi_deg, cpa0, method, measured_on, ...)
    or None when no calibration exists for this scan. Policy note: consumers
    apply ``psi_deg`` ONLY — the fitted ``cpa0`` intercept is known estimator
    bias, not geometry (applying it split muNeRF run zsu85kc6's z=+23 mm tube
    into two overlapped half-discs; the FBP tube test of 2026-08-12 shows the
    column CoR round at the detector's geometric centre).
    """
    try:
        serial = detector_serial_from_scan(scan_folder)
        tag = Path(scan_folder).name
        cal_path = (calibration_dir()
                    / f"detector_psi_{serial}_{tag}.json")
        if not serial or not cal_path.exists():
            if verbose:
                print(f"\n  Geometry auto-calibration: no cached measurement "
                      f"for this scan ({cal_path.name}) — {fallback_note}. "
                      f"Run muNeRF or "
                      f"scripts/detector_psi_from_conjugates.py once on this "
                      f"scan to populate it.")
            return None
        record = json.loads(cal_path.read_text())
        record['_path'] = str(cal_path)
        if verbose:
            print(f"\n  Geometry auto-calibration from {cal_path.name}:")
            print(f"    psi = {float(record['psi_deg']):+.4f} deg  (method "
                  f"{record.get('method', 'conjugate')}; fitted cpa0 "
                  f"{float(record.get('cpa0', float('nan'))):.3f} NOT applied "
                  f"— CoR stays at the geometric centre; measured "
                  f"{record.get('measured_on', '?')})")
        return record
    except Exception as e:
        if verbose:
            print(f"\n  Geometry auto-calibration skipped "
                  f"({type(e).__name__}: {e})")
        return None


def measure_detector_psi(ctx: ScanContext, verbose=True) -> Optional[dict]:
    """Measure the detector in-plane rotation from THIS scan's projections.

    Runs the half-scan-consistency estimator (ct_core.geometry_selfcal — the
    validated method ported from muNeRF) on a pooled, flat-fielded copy of
    the projections, writes the shared calibration JSON so every later run
    (FDK, iterative, muNeRF) gets a cache hit, and returns the record.

    Needs a CUDA device (~1-5 min of FBP scoring; CPU would take hours) and
    bright/dark fields. Returns None — with the reason printed — whenever
    measurement is not possible or the estimator rejects the data, so the
    caller can fall back exactly as if no calibration existed.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("  Geometry self-calibration skipped: no CUDA device "
                  "(the half-scan estimator needs a GPU).")
        return None
    if ctx.bright_field is None or ctx.dark_field is None:
        if verbose:
            print("  Geometry self-calibration skipped: no bright/dark "
                  "fields to form line integrals from.")
        return None

    from .geometry_selfcal import (
        estimate_psi_halfncc,
        prepare_estimation_sinogram,
        calibration_json_path,
        write_calibration_json,
    )

    try:
        serial = detector_serial_from_scan(ctx.scan_folder)
        # pool to ~raw-ds-3 for the estimator, on top of any --downsample
        factor = max(1, int(round(3 / max(1, ctx.downsample))))
        total_ds = ctx.downsample * factor
        if verbose:
            print(f"\n  Geometry self-calibration: measuring detector psi "
                  f"from the projections (half-scan consistency, "
                  f"pool {total_ds}x)...")
        sino = prepare_estimation_sinogram(
            ctx.projections, ctx.bright_field, ctx.dark_field, factor=factor)

        g = dict(ctx.geometry)
        g['da'] = float(g['da']) * factor
        g['db'] = float(g['db']) * factor
        for key in ('central_pixel_a', 'central_pixel_b'):
            g[key] = (float(g[key]) - (factor - 1) / 2.0) / factor

        angles_t = torch.as_tensor(np.asarray(ctx.angles, dtype=np.float64))
        result = estimate_psi_halfncc(sino, angles_t, g,
                                      downsample=total_ds, verbose=verbose)

        record = None
        if serial:
            path = calibration_json_path(
                None, serial, Path(ctx.scan_folder).name)
            write_calibration_json(path, result, downsample=total_ds,
                                   detector_serial=serial)
            if verbose:
                print(f"  Saved calibration to {path.name} — future runs on "
                      f"this scan (any pipeline) will reuse it.")
        elif verbose:
            print("  (no detector serial in the projection headers — "
                  "measurement used but not cached)")
        record = dict(result)
        ctx.detector_psi = record
        if verbose:
            print(f"  psi = {float(record['psi_deg']):+.4f} deg (halfncc, "
                  f"prominence {float(record.get('prominence', 0.0)):.2f}) — "
                  f"fitted cpa0 NOT applied (estimator bias; CoR stays at "
                  f"the geometric centre)")
        return record
    except Exception as e:
        if verbose:
            print(f"  Geometry self-calibration FAILED "
                  f"({type(e).__name__}: {e}) — continuing without it.")
        return None


def resolve_or_measure_detector_psi(ctx: ScanContext, verbose=True,
                                    fallback_note="using psi=0 and the XML CoR"
                                    ) -> Optional[dict]:
    """The standard geometry-calibration entry point for every driver.

    Cached JSON first (written by any pipeline — muNeRF or this package);
    on a miss, measure from this scan's own projections and cache. Returns
    None only when both paths are unavailable, in which case the stated
    fallback applies.
    """
    record = resolve_detector_psi(ctx.scan_folder, verbose=False)
    if record is not None:
        if verbose:
            print(f"\n  Geometry calibration (cached "
                  f"{Path(record['_path']).name}): psi = "
                  f"{float(record['psi_deg']):+.4f} deg (method "
                  f"{record.get('method', 'conjugate')}, measured "
                  f"{record.get('measured_on', '?')}; fitted cpa0 NOT "
                  f"applied)")
        ctx.detector_psi = record
        return record
    record = measure_detector_psi(ctx, verbose=verbose)
    if record is None and verbose:
        print(f"  Geometry calibration unavailable — {fallback_note}.")
    return record


# --------------------------------------------------------------------------
# Output stage, shared by every driver
# --------------------------------------------------------------------------

def save_outputs(volume, ctx: ScanContext, args, output_path: str,
                 logger=None, algorithm: str | None = None):
    """Crop to the export ROI, then HU-calibrate, filter and write the VFF.

    This is the one place the HU scale is set, for every backend. The
    reconstructors all hand over raw attenuation now, so a volume's units are
    decided here or nowhere.

    The crop happens BEFORE calibration on purpose: the anchors are read out
    of the volume's own histogram, and they should be read from the voxels
    being shipped, not from a domain padded out with bed and cage. The
    trade-off is that a very tight export ROI can crop away most of the air
    the offset anchor needs — the calibrator warns when that happens.

    ScanContext.geometry is updated to describe the cropped grid so the VFF
    header and every downstream plot agree with the pixels. No-op when no
    export ROI is set.

    Alongside the VFF goes a ``<output>.json`` sidecar recording the grid, the
    position, and the HU map that was applied — the things a GE ncaa header
    cannot carry (see scan_setup.write_sidecar).

    Returns (vff path, HUAnchors, calibrated HU volume).
    """
    volume, ctx.geometry = crop_to_export_roi(volume, ctx.geometry)
    # Sidecar identity: the scan BASENAME only, never a path (same rule as the
    # W&B config whitelist — these files travel with the volume).
    metadata = {'scan': Path(ctx.scan_folder).name,
                'downsample': int(ctx.downsample or 1)}
    if algorithm:
        metadata['algorithm'] = algorithm
    psi = ctx.geometry.get('det_psi_rad')
    if psi:
        metadata['detector_psi_deg'] = float(np.degrees(psi))
    path, anchors, volume_hu = postprocess_and_save(
        volume_mu=volume,
        geometry=ctx.geometry,
        output_path=output_path,
        bilateral_filter=args.bilateral_filter,
        bilateral_sigma_spatial=args.bilateral_sigma_spatial,
        bilateral_sigma_range=args.bilateral_sigma_range,
        voxel_xy=args.voxel_xy,
        hu_calibration=getattr(args, 'hu_calibration', 'auto'),
        mu_water=getattr(args, 'mu_water', None),
        tissue_hu=getattr(args, 'tissue_hu', None),
        save_mu=getattr(args, 'save_mu', False),
        metadata=metadata,
    )
    if logger is not None:
        logger.log_hu_calibration(anchors)
    return path, anchors, volume_hu
