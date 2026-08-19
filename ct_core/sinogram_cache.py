"""Skip the slow half of loading a scan when nothing about it has changed.

Turning a scan folder into line integrals means reading a few hundred
full-resolution VFF projections off disk, flat-fielding them, soft-clipping,
taking -log, and running a ring correction — on Scan_1510 that is ~7 GB of
I/O and about a minute of work, repeated identically at the start of every
run. None of it depends on the reconstruction: the same scan with the same
preprocessing arguments always produces the same sinogram.

So it is cached, keyed by a hash of exactly the fields that change the OUTPUT.
Geometry (FOV, voxel size, the ROI, the centre-of-rotation policy) is
deliberately NOT in the key — it changes the volume grid built later, never
the sinogram values — so sweeping any of it is free.

Moved here from muNeRF's ``inr_pipeline/dataset.py`` 2026-08-18. It was the
one part of that loader the submodule had no equivalent for, and it is useful
to every backend, not just the learned ones: FDK, ASTRA and TIGRE re-read and
re-preprocess the same projections on every invocation too. The key algorithm
and the on-disk payload are UNCHANGED, so caches written by muNeRF before the
move are still hits.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

#: Bumped only if the payload layout changes. The KEY deliberately does not
#: include it — an old cache stays readable, and the loader validates keys.
PAYLOAD_KEYS = ("sinogram", "angles", "xml_header", "raw_detector_shape")


def cache_key(*, projections_dir, scan_folder=None, projection_pattern=None,
              total_angle="determined", ring_correction=True,
              ring_median_width=51, air_normalization=True,
              downsample=1) -> str:
    """Stable 16-char hash of every field that changes the sinogram VALUES.

    ``air_normalization`` belongs here and once did not: it used to be applied
    after the cache was written, so it was correctly absent. Now that
    ``preprocess_sinogram`` performs it, two settings produce two different
    sinograms and flipping the flag must not silently reuse the one levelled
    the other way.
    """
    payload = {
        "projections_dir": str(projections_dir or ""),
        "scan_folder": str(scan_folder or ""),
        "projection_pattern": projection_pattern,
        "total_angle": total_angle,
        "ring_correction": bool(ring_correction),
        "ring_median_width": int(ring_median_width),
        "air_normalization": bool(air_normalization),
        # In the key because the detector is binned BEFORE preprocessing (see
        # load_or_build), so the cached line integrals are specific to it.
        "downsample": int(downsample),
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(cache_root, scan_name: str, key: str) -> Path:
    return Path(cache_root) / scan_name / f"sinogram_cache_{key}.pt"


def load_or_build(*, data_folder, scan_folder, scan_name=None, cache_root=None,
                  cache_file=None,
                  projection_pattern=None, total_angle="determined",
                  sub_scan="-00-", ring_correction=True, ring_median_width=51,
                  air_normalization=True, soft_clip_sharpness=None,
                  downsample=1, force_recompute=False, verbose=True) -> dict:
    """Preprocessed line integrals for a scan, from cache when possible.

    Returns ``{"sinogram": (A, N_b, N_a) float32, "angles": (A,) float32,
    "xml_header": dict, "raw_detector_shape": (N_b, N_a) before binning,
    "cached": bool, "path": Path}``. ``sinogram`` and ``angles`` are torch
    tensors — that is the on-disk format and converting them here would only
    cost a copy for callers that want tensors anyway.

    ``downsample`` bins the detector by average pooling BEFORE preprocessing,
    on the raw counts and the bright/dark fields together. That order matters
    and is not interchangeable with pooling the finished line integrals:
    ``mean(-log x) != -log(mean x)``, and a physically larger pixel measures
    the latter. Pooling afterwards over-estimates attenuation — MEASURED on
    Scan_1510 at ds=3: +0.16% of signal systematically, 0.48% mean absolute
    and 9.2% at edges, where a bin spans the widest dynamic range. It is only
    0.03x the per-pixel noise, so it is invisible per ray, but it is a bias
    rather than noise and does not average away over 10^8 rays.

    ``raw_detector_shape`` is the TRUE panel size before binning (and before
    the pooling trim), because the detector-warp calibration is stored in raw
    pixels and validated against it — ``sinogram.shape * downsample`` is 1-2 px
    short.

    ``cache_root`` is where per-scan caches live (a results directory); the
    file lands at ``<cache_root>/<scan_name>/sinogram_cache_<key>.pt``.
    """
    import torch                      # local: ct_core stays importable without it

    # `cache_file` lets a caller decide what IDENTIFIES the cache separately
    # from where the scan is loaded. muNeRF needs that: its key is built from
    # the raw config strings, while loading uses the resolved absolute paths —
    # keying on the resolved form would mean a config that spells the same
    # scan differently never hits, and re-preprocessing costs ~4 minutes and
    # another 6.6 GiB on disk.
    if cache_file is not None:
        path = Path(cache_file)
    else:
        if scan_name is None or cache_root is None:
            raise ValueError("pass cache_file=, or scan_name= and cache_root=")
        path = cache_path(cache_root, scan_name,
                          cache_key(projections_dir=data_folder,
                                    scan_folder=scan_folder,
                                    projection_pattern=projection_pattern,
                                    total_angle=total_angle,
                                    ring_correction=ring_correction,
                                    ring_median_width=ring_median_width,
                                    air_normalization=air_normalization,
                                    downsample=downsample))

    if path.exists() and not force_recompute:
        t0 = time.time()
        if verbose:
            print(f"Loading cached sinogram from {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        missing = [k for k in PAYLOAD_KEYS if k not in blob]
        if missing:
            raise ValueError(
                f"{path} is missing {missing} — it predates this cache format. "
                f"Delete it, or pass force_recompute=True to overwrite.")
        if verbose:
            print(f"  cache load took {time.time() - t0:.1f}s "
                  f"(sinogram shape {tuple(blob['sinogram'].shape)})")
        return {**{k: blob[k] for k in PAYLOAD_KEYS}, "cached": True,
                "path": path}

    from .preprocessing import preprocess_sinogram
    from .scan_setup import load_scan_data

    if force_recompute and verbose:
        print("force_recompute — ignoring any existing cache")

    raw = load_scan_data(data_folder=str(data_folder),
                         scan_folder=str(scan_folder),
                         projection_pattern=projection_pattern,
                         total_angle=total_angle, sub_scan=sub_scan)

    projections, bright, dark = (raw["projections"], raw["bright_field"],
                                 raw["dark_field"])
    raw_detector_shape = (int(projections.shape[1]), int(projections.shape[2]))

    ds = int(downsample)
    if ds > 1:
        from .preprocessing import downsample_projections
        if verbose:
            print(f"  Binning the detector {ds}x on the RAW counts "
                  f"(before flat-field/log), as every backend does")
        projections = downsample_projections(projections, ds)
        if bright is not None:
            bright = downsample_projections(bright, ds)
        if dark is not None:
            dark = downsample_projections(dark, ds)

    kwargs = dict(ring_correction=ring_correction,
                  ring_median_width=ring_median_width,
                  air_normalization=air_normalization)
    if soft_clip_sharpness is not None:
        kwargs["soft_clip_sharpness"] = soft_clip_sharpness
    sino_np = preprocess_sinogram(projections=projections, bright_field=bright,
                                  dark_field=dark, **kwargs)

    sinogram = torch.from_numpy(np.ascontiguousarray(sino_np))
    angles = raw["angles"].clone().detach().to(torch.float32)
    blob = {"sinogram": sinogram, "angles": angles,
            "xml_header": raw["xml_header"],
            "raw_detector_shape": raw_detector_shape}

    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.save(blob, path)
    if verbose:
        print(f"Saved sinogram cache to {path} "
              f"({path.stat().st_size / 2**30:.2f} GiB, "
              f"{time.time() - t0:.1f}s to write)")
    return {**blob, "cached": False, "path": path}
