"""Scene containers for learning-based (differentiable) reconstruction.

A `Scene` bundles four things:
  * preprocessed line integrals  (sinogram, after flat-field/log/ring)
  * cone-beam projection angles  (radians)
  * cone-beam geometry           (ct_core's plain dict; defines the EXPORT ROI)
  * `model_domain`               (the integration domain — usually larger than
                                  the export ROI to absorb out-of-FOV matter)

`scene.aabb_min/max` route to `model_domain` because that is what the renderer
integrates over. The geometry-derived `scene.export_aabb_min/max` is the box
that export crops the saved volume down to.

Coordinates are isocenter-centered millimetres throughout. The mm → [-1,1]
affine that the model sees uses the model domain, not the export ROI.

Moved here (verbatim) from muNeRF's ``inr_pipeline/dataset.py`` 2026-08-12 so
every learning-based algorithm — inside this submodule and in muNeRF — shares
one definition. muNeRF's ``inr_pipeline.dataset`` re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelDomain:
    """Integration domain for the renderer.

    The AABB bounds the domain. For shape='cylinder' the domain is the
    inscribed cylinder (radius_xy = extent_xy / 2), which lets the model cover
    bed/external matter while still telling the optimizer "outside the cylinder
    is air" — the AABB corners never reach the loss.

    ``ray_clip`` selects which body the renderer lays its quadrature points
    across (see ``renderer.ray_domain_intersect``):

      * ``'domain'`` (default) — the cylinder itself. Every sample lands on
        real domain, and the step is ``cylinder_chord / num_samples``.
      * ``'aabb'`` (legacy) — the bounding box, with out-of-cylinder samples
        evaluated and then multiplied by zero. MEASURED on Scan_1510's
        geometry: 19.5% of all samples are discarded this way (1 - pi/4 for a
        disc inscribed in a square, less the rays the z-slab clips first), and
        up to 71% on rays grazing the domain boundary. Both settings integrate
        the same function — the corners hold no material by definition — so
        this only trades compute and quadrature resolution, not correctness.
        Kept to reproduce runs from before 2026-08-09.

    Note the saving is concentrated in the PERIPHERY: a ray through the axis
    sees cylinder_chord ~= box_chord and gains nothing. On Scan_1510 the mouse
    sits at |b| < 13 mm, where the step improves by only 1.01x; the win there
    is throughput, not resolution.

    The AABB may be centered off-isocenter via the ``origin`` config key,
    which shifts the domain to better cover an off-center ROI (reducing
    wasted hash-grid capacity on empty space).
    """

    shape: str                       # 'cylinder' | 'box'
    aabb_min: torch.Tensor           # (3,) mm, float32
    aabb_max: torch.Tensor           # (3,) mm, float32
    radius_xy: float | None = None   # only meaningful for shape='cylinder'
    center_xy: tuple[float, float] = (0.0, 0.0)  # cylinder axis center in mm
    ray_clip: str = "domain"         # 'domain' | 'aabb' (legacy)

    def cylinder_mask(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        """Float (..., ) mask. 1 inside the cylinder (or always 1 for box)."""
        if self.shape == "box" or self.radius_xy is None:
            return torch.ones(
                xyz_mm.shape[:-1], dtype=xyz_mm.dtype, device=xyz_mm.device
            )
        dx = xyz_mm[..., 0] - self.center_xy[0]
        dy = xyz_mm[..., 1] - self.center_xy[1]
        r2 = dx ** 2 + dy ** 2
        return (r2 < self.radius_xy ** 2).to(xyz_mm.dtype)

    def normalize(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        amin = self.aabb_min.to(xyz_mm)
        amax = self.aabb_max.to(xyz_mm)
        center = (amin + amax) * 0.5
        half = (amax - amin) * 0.5
        return (xyz_mm - center) / half


@dataclass
class Scene:
    sinogram: torch.Tensor       # (N_angles, N_b, N_a) line integrals, float32
    angles: torch.Tensor         # (N_angles,) radians, float32
    geometry: dict[str, Any]     # ct_core build_geometry output (export ROI)
    scan_name: str
    model_domain: ModelDomain    # REQUIRED — no silent fallback to export ROI.

    @property
    def n_angles(self) -> int:
        return int(self.sinogram.shape[0])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return int(self.sinogram.shape[1]), int(self.sinogram.shape[2])

    @property
    def export_aabb_min(self) -> torch.Tensor:
        ox, oy, oz = self.geometry["vol_origin"]
        Nx, Ny, Nz = self.geometry["vol_shape"]
        dx, dz = self.geometry["dx"], self.geometry["dz"]
        return torch.tensor(
            [ox - Nx * dx / 2.0, oy - Ny * dx / 2.0, oz - Nz * dz / 2.0],
            dtype=torch.float32,
        )

    @property
    def export_aabb_max(self) -> torch.Tensor:
        ox, oy, oz = self.geometry["vol_origin"]
        Nx, Ny, Nz = self.geometry["vol_shape"]
        dx, dz = self.geometry["dx"], self.geometry["dz"]
        return torch.tensor(
            [ox + Nx * dx / 2.0, oy + Ny * dx / 2.0, oz + Nz * dz / 2.0],
            dtype=torch.float32,
        )

    # Renderer-facing AABB == the integration domain.
    @property
    def aabb_min(self) -> torch.Tensor:
        return self.model_domain.aabb_min

    @property
    def aabb_max(self) -> torch.Tensor:
        return self.model_domain.aabb_max


def normalize_to_unit_cube(xyz_mm: torch.Tensor, scene: Scene) -> torch.Tensor:
    """Map mm-space points (..., 3) into [-1, 1]^3 spanning the model domain.

    Per-axis affine — extents along x/y and z are not equal in general. Points
    outside the model domain return values outside [-1, 1]; the renderer
    handles masking via the AABB clip and (for cylinder domains) the cylinder
    mask.
    """
    return scene.model_domain.normalize(xyz_mm)


#: What `model_domain_from_spec` accepts, besides an (extent_xy, half_z) pair.
DOMAIN_SPECS = ("auto", "off")


def model_domain_from_bounds(bounds: dict, *, shape: str = "cylinder",
                             ray_clip: str = "domain") -> ModelDomain:
    """A ModelDomain from ct_core's bounds dict (x_min/x_max/.../z_max, mm).

    The one conversion between the submodule's bounds representation — which
    is what ``build_geometry`` takes as ``roi_bounds`` and what
    ``ct_core.support`` produces — and the renderer's domain object. Bounds may
    be asymmetric in z (nothing forces symmetry there and the saving is real);
    the cylinder axis is placed at the bounds' own xy centre.
    """
    if ray_clip not in ("domain", "aabb"):
        raise ValueError(
            f"ray_clip must be 'domain' or 'aabb', got {ray_clip!r}")
    lo = [float(bounds["x_min"]), float(bounds["y_min"]), float(bounds["z_min"])]
    hi = [float(bounds["x_max"]), float(bounds["y_max"]), float(bounds["z_max"])]
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    radius = min(hi[0] - lo[0], hi[1] - lo[1]) / 2.0 if shape == "cylinder" else None
    return ModelDomain(shape=shape,
                       aabb_min=torch.tensor(lo, dtype=torch.float32),
                       aabb_max=torch.tensor(hi, dtype=torch.float32),
                       radius_xy=radius, center_xy=(cx, cy), ray_clip=ray_clip)


def model_domain_from_spec(spec, *, geometry: dict, projections=None,
                           bright_field=None, dark_field=None,
                           shape: str = "cylinder", ray_clip: str = "domain",
                           verbose: bool = True) -> ModelDomain:
    """Resolve a domain the way every learned backend resolves it.

    `spec` mirrors the drivers' ``--model-domain``:

      * ``'auto'`` (recommended) — MEASURE the attenuating support from the
        projections and clamp it to the detector fan and the cone reach. This
        is a fact about what the rays crossed, so it sizes itself per scan and
        needs no hand-computed extents.
      * ``(extent_xy, half_z)`` — pin it, in muNeRF's config units.
      * ``'off'`` — the export FOV, i.e. domain == saved volume. Correct ONLY
        when nothing outside the FOV attenuates. Otherwise the missing
        attenuation becomes a low-frequency cup across the whole
        reconstruction (~96 HU of DC bias measured on Scan_1510) rather than a
        boundary artifact — see ``ct_core.support``.

    `projections` are raw counts when `bright_field`/`dark_field` are given, or
    line integrals when they are not — so a caller holding an already
    preprocessed sinogram passes it directly and omits the fields.
    """
    from ..ct_core.support import (explicit_domain_bounds,
                                   measure_attenuating_support,
                                   support_to_domain_bounds)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        bounds = explicit_domain_bounds(float(spec[0]), float(spec[1]))
        if verbose:
            print(f"Model domain pinned: extent_xy {float(spec[0]):.1f} mm, "
                  f"half_extent_z {float(spec[1]):.1f} mm")
        return model_domain_from_bounds(bounds, shape=shape, ray_clip=ray_clip)

    text = str(spec).strip().lower()
    if text == "off":
        return model_domain_from_geometry(geometry, shape=shape,
                                         ray_clip=ray_clip)
    if text != "auto":
        raise ValueError(
            f"model domain spec must be 'auto', 'off', or (extent_xy, half_z), "
            f"got {spec!r}")
    if projections is None:
        raise ValueError(
            "model domain 'auto' measures the support from the projections — "
            "pass projections= (raw counts with bright/dark, or line "
            "integrals without).")
    support = measure_attenuating_support(
        projections, geometry, bright_field=bright_field,
        dark_field=dark_field, verbose=verbose)
    return model_domain_from_bounds(support_to_domain_bounds(support),
                                    shape=shape, ray_clip=ray_clip)


def model_domain_from_geometry(geometry: dict, *, shape: str = "cylinder",
                               ray_clip: str = "domain") -> ModelDomain:
    """A ModelDomain covering exactly the export ROI of a ct_core geometry.

    The convenience constructor for submodule backends, where the scan's
    reconstruction FOV (``fov_xy``/``fov_z`` via ``build_geometry``) IS the
    integration domain. muNeRF configs instead build enlarged domains that
    absorb out-of-FOV matter (scan bed) — that logic stays in muNeRF's
    ``dataset._build_model_domain``.
    """
    ox, oy, oz = (float(v) for v in geometry["vol_origin"])
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])
    hx, hy, hz = Nx * dx / 2.0, Ny * dx / 2.0, Nz * dz / 2.0
    aabb_min = torch.tensor([ox - hx, oy - hy, oz - hz], dtype=torch.float32)
    aabb_max = torch.tensor([ox + hx, oy + hy, oz + hz], dtype=torch.float32)
    radius = min(hx, hy) if shape == "cylinder" else None
    return ModelDomain(shape=shape, aabb_min=aabb_min, aabb_max=aabb_max,
                       radius_xy=radius, center_xy=(ox, oy), ray_clip=ray_clip)


def build_scene(sinogram, angles, xml_header, *, scan_name, scan_folder,
                raw_detector_shape, geom_cfg, model_domain_cfg=None,
                downsample: int = 1, cor_mode: str = "center",
                use_scanner_roi: bool = True, calib_dir=None,
                verbose: bool = True) -> Scene:
    """A ready-to-render `Scene` from a preprocessed sinogram and the header.

    The whole front end of a learning-based run, in the order the pieces
    depend on each other, and all of it the submodule's:

      1. the volume grid + centre-of-rotation policy + pitch rescale
         (``ct_core.pipeline.build_scan_geometry``), optionally cropped to the
         scanner's own ROI when it published one;
      2. the per-pixel detector warp, keyed by detector serial
         (``detector_warp.resolve_detector_warp``);
      3. the in-plane detector rotation psi
         (``ct_core.detector_psi.resolve_detector_psi``) — AFTER the warp,
         because the estimator corrects for it, and after the pitch rescale,
         because it works in pooled index units;
      4. the integration domain (``model_domain_from_spec``), measured from
         this same sinogram unless it is pinned.

    Nothing here needs the raw projections, so a caller holding a cached
    sinogram never re-reads the scan. `geom_cfg` is the same geometry-config
    dict `resolve_detector_psi` and `resolve_detector_warp` already take.

    Order matters twice, and both were bugs once: psi must be resolved after
    da/db have been scaled by the binning factor, and the domain must be
    resolved after psi, because the support measurement reads detector
    channels through the geometry.
    """
    import math

    import torch as _torch

    from ..ct_core.detector_psi import resolve_detector_psi
    from ..ct_core.pipeline import build_scan_geometry
    from ..ct_core.scan_setup import parse_crop_boundary
    from .detector_warp import resolve_detector_warp

    roi_bounds = None
    if use_scanner_roi:
        from pathlib import Path as _Path
        crop_xml = _Path(scan_folder) / "Volumes" / "SubVolumeCoordinates.xml"
        if crop_xml.exists():
            roi_bounds = parse_crop_boundary(str(scan_folder), xml_header)
        elif verbose:
            print(f"  use_scanner_roi=true but {crop_xml} not found — "
                  f"falling back to fov_xy/fov_z.")

    geometry = build_scan_geometry(
        xml_header, raw_detector_shape=raw_detector_shape,
        fov_xy=geom_cfg["fov_xy"], fov_z=geom_cfg["fov_z"],
        voxel_xy=geom_cfg["voxel_xy"], voxel_z=geom_cfg["voxel_z"],
        roi_bounds=roi_bounds, cor_mode=cor_mode, downsample=downsample,
        verbose=verbose)

    ds = int(downsample)
    geometry["detector_warp"] = resolve_detector_warp(
        geom_cfg, scan_folder, raw_detector_shape, downsample=ds)

    psi_deg = resolve_detector_psi(
        geom_cfg, geometry, sinogram=sinogram, angles=angles,
        scan_folder=scan_folder, warp=geometry["detector_warp"], downsample=ds,
        calib_dir=calib_dir, verbose=verbose)
    geometry["det_psi_rad"] = math.radians(psi_deg)
    # Always recorded, whatever the mode, so a geometry term this large can
    # never again be silently wrong the way psi = 0 was for months.
    geometry["det_psi_deg"] = float(psi_deg)
    if psi_deg and verbose:
        print(f"  Detector in-plane rotation: psi = {psi_deg:+.4f} deg")

    md_cfg = dict(model_domain_cfg or {})
    resolved = md_cfg.get("resolved_bounds")
    shape = str(md_cfg.get("shape", "cylinder"))
    ray_clip = str(md_cfg.get("ray_clip", "domain")).lower()
    if resolved is not None:
        domain = model_domain_from_bounds(resolved, shape=shape,
                                          ray_clip=ray_clip)
    else:
        if bool(md_cfg.get("auto", True)):
            spec = "auto"
        else:
            exy, hez = md_cfg.get("extent_xy"), md_cfg.get("half_extent_z")
            spec = ((float(exy), float(hez))
                    if exy is not None and hez is not None else "off")
        domain = model_domain_from_spec(
            spec, geometry=geometry,
            projections=(sinogram.numpy() if hasattr(sinogram, "numpy")
                         else sinogram),
            shape=shape, ray_clip=ray_clip, verbose=verbose)
        lo, hi = domain.aabb_min.tolist(), domain.aabb_max.tolist()
        # A measured domain must be reproducible without the projections: an
        # INR's coordinate normalisation IS its domain, so inference that
        # resolved a different one would sample the wrong field. Stamped back
        # into the caller's dict, it travels with the checkpoint's config.
        md_cfg["resolved_bounds"] = {
            "x_min": lo[0], "x_max": hi[0], "y_min": lo[1], "y_max": hi[1],
            "z_min": lo[2], "z_max": hi[2]}
        if model_domain_cfg is not None:
            model_domain_cfg["resolved_bounds"] = md_cfg["resolved_bounds"]

    return Scene(sinogram=_torch.as_tensor(sinogram),
                 angles=_torch.as_tensor(angles),
                 geometry=geometry, scan_name=scan_name,
                 model_domain=domain)
