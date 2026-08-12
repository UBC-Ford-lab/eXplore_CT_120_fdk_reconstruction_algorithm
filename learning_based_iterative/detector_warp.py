"""Per-pixel detector distortion correction, applied to RAY GEOMETRY.

Background
----------
The scanner's raw ``acq-*`` frames are geometrically distorted: the detector
pixel grid is not the ideal linear grid our projector assumes. The vendor
corrects this before reconstructing (its ``Corrected/uwarp-*`` frames), which
is why the vendor volume never registered to reconstructions made from raw
frames.

The field was measured by registering the PHOTON NOISE between paired
``acq``/``uwarp`` frames (signal-domain metrics are blind to it — a 20 px warp
moves whole-frame NCC only in the 4th decimal), fitted to a bicubic, and
validated on held-out frames: noise correlation 0.0007 -> 0.58, deterministic
content reproduced to 0.42-0.62% of signal.

Two properties make it usable as a general calibration:

* it is a property of the DETECTOR, not the scan — independently fitted on a
  220-view short scan and a 440-view full scan of different objects, the two
  fields agree to 0.178 px mean / 1.276 px max on a 13.77 px field;
* its NON-AFFINE part transfers across a vendor recalibration (verified on a
  2022 scan using a field measured in 2026, same detector serial), while its
  affine part does not — the affine part is per-scan geometry that
  ``central_pixel_a/b``, ``da``/``db`` and the scan's own ``full.center`` /
  ``full.rfan`` already carry.

Hence the default mode is ``nonaffine``.

Why this is applied to rays and not to the images
-------------------------------------------------
Resampling every projection by ~14 px would apply a bilinear low-pass to the
data, destroying the high frequencies the reconstruction is trying to recover.
muNeRF casts one ray per detector sample, so correcting the sample POSITION is
exact and costs no interpolation. This module therefore returns corrected
detector indices; ``ray_sampler.rays_from_indices`` consumes them.

Convention
----------
The measured field ``d`` satisfies ``ideal(p) = raw(p + d(p))``: the ideal
(undistorted) detector coordinate ``p`` reads the raw frame at ``p + d(p)``.
We hold a raw sample index ``q`` and need its ideal coordinate ``p``, i.e. we
must solve ``q = p + d(p)``. ``d`` is smooth and small relative to the
detector, so a two-step fixed point ``p <- q - d(p)`` converges well inside a
hundredth of a pixel.

All indices here are in PIPELINE units (post-downsampling); the stored
polynomial is in RAW detector pixels, and the mapping between them is applied
internally from ``downsample``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CALIB_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"


# ---------------------------------------------------------------- helpers ---

def poly_terms(deg: int) -> list[tuple[int, int]]:
    """Exponent pairs (i, j) for r^i c^j, matching the fitting basis order."""
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]


def _basis(rn, cn, deg: int):
    """Design matrix for the 2-D monomial basis, numpy or torch."""
    if isinstance(rn, torch.Tensor):
        return torch.stack([rn ** i * cn ** j for i, j in poly_terms(deg)], -1)
    return np.stack([rn ** i * cn ** j for i, j in poly_terms(deg)], -1)


def _moment(k: int) -> float:
    """Mean of x^k over x in [-1, 1]: 1/(k+1) for even k, 0 for odd."""
    return 0.0 if k % 2 else 1.0 / (k + 1)


def strip_affine(coefs: np.ndarray, deg: int) -> np.ndarray:
    """Return coefficients for the polynomial minus its least-squares affine fit.

    The residual of a polynomial minus an affine function is a polynomial in
    the same basis, so this is exact in coefficient space. The affine fit must
    be subtracted because the monomial basis is not orthogonal — r^2 has a
    non-zero mean, so removing the affine part also shifts the constant term.

    The projection is done ANALYTICALLY over the continuous square [-1, 1]^2
    rather than on a sampling grid: {1, r, c} are orthogonal there, giving
    alpha = (<f,1>, 3<f,r>, 3<f,c>) in closed form. A discrete grid would make
    the result depend on its own resolution (mean(r^2) differs by ~1% between
    a 64- and a 96-point grid), which is not a property a calibration should
    have.
    """
    out = np.array(coefs, dtype=np.float64, copy=True)
    terms = poly_terms(deg)
    idx = {t: k for k, t in enumerate(terms)}
    for axis in range(out.shape[0]):
        f = out[axis]
        a0 = sum(f[k] * _moment(i) * _moment(j) for k, (i, j) in enumerate(terms))
        a1 = 3.0 * sum(f[k] * _moment(i + 1) * _moment(j)
                       for k, (i, j) in enumerate(terms))
        a2 = 3.0 * sum(f[k] * _moment(i) * _moment(j + 1)
                       for k, (i, j) in enumerate(terms))
        out[axis][idx[(0, 0)]] -= a0
        out[axis][idx[(1, 0)]] -= a1
        out[axis][idx[(0, 1)]] -= a2
    return out


# ------------------------------------------------------------------ model ---

class DetectorWarp:
    """A detector distortion calibration, evaluated on pipeline indices."""

    def __init__(self, coefs, deg: int, raw_shape, *, mode: str = "nonaffine",
                 detector_serial: str | None = None, source: str | None = None):
        coefs = np.asarray(coefs, dtype=np.float64)
        if coefs.ndim != 2 or coefs.shape[0] != 2:
            raise ValueError(f"coefs must be (2, n_terms); got {coefs.shape}")
        if coefs.shape[1] != len(poly_terms(deg)):
            raise ValueError(
                f"coefs has {coefs.shape[1]} terms but degree {deg} needs "
                f"{len(poly_terms(deg))}")
        if mode not in ("nonaffine", "full"):
            raise ValueError(f"mode must be 'nonaffine' or 'full'; got {mode!r}")
        self.deg = int(deg)
        self.mode = mode
        self.raw_shape = (int(raw_shape[0]), int(raw_shape[1]))
        self.detector_serial = detector_serial
        self.source = source
        self.coefs = strip_affine(coefs, self.deg) if mode == "nonaffine" else coefs
        self._cache: dict = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path, mode: str = "nonaffine") -> "DetectorWarp":
        z = np.load(str(path))
        return cls(z["coefs"], int(z["deg"]), z["raw_shape"], mode=mode,
                   detector_serial=(str(z["detector_serial"])
                                    if "detector_serial" in z else None),
                   source=str(path))

    @classmethod
    def for_detector(cls, serial: str, mode: str = "nonaffine",
                     calib_dir: Path | None = None) -> "DetectorWarp | None":
        """Look a calibration up by DETECTOR SERIAL.

        Keyed by serial rather than by scan because the field is a property of
        the hardware: one file serves every scan taken on that detector, which
        is what makes this work across scans without per-scan fitting.
        """
        d = Path(calib_dir or CALIB_DIR)
        p = d / f"detector_warp_{serial}.npz"
        return cls.load(p, mode=mode) if p.exists() else None

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, coefs=self.coefs, deg=self.deg,
                 raw_shape=np.array(self.raw_shape),
                 detector_serial=str(self.detector_serial or ""),
                 mode=self.mode)
        return path

    # -- evaluation ------------------------------------------------------
    def _displacement_raw(self, r_raw, c_raw):
        """Field in RAW detector pixels at raw (possibly fractional) coords."""
        H, W = self.raw_shape
        rn = (r_raw - (H - 1) / 2.0) / ((H - 1) / 2.0)
        cn = (c_raw - (W - 1) / 2.0) / ((W - 1) / 2.0)
        B = _basis(rn, cn, self.deg)
        if isinstance(B, torch.Tensor):
            co = torch.as_tensor(self.coefs, dtype=B.dtype, device=B.device)
            return B @ co[0], B @ co[1]
        return B @ self.coefs[0], B @ self.coefs[1]

    def ideal_indices(self, b_idx, a_idx, downsample: int = 1, iters: int = 2):
        """Map raw pipeline indices to IDEAL detector indices.

        Solves ``q = p + d(p)`` for ``p`` by fixed-point iteration, in raw
        pixel units, then converts back to pipeline index units.

        Parameters
        ----------
        b_idx, a_idx : index tensors in pipeline (post-downsample) units;
            may be fractional.
        downsample : the sinogram's average-pool factor. Pipeline index k
            covers raw samples [ds*k, ds*k + ds), centred at
            ``ds*k + (ds-1)/2``.
        """
        ds = int(downsample)
        off = (ds - 1) / 2.0
        b_f = b_idx.to(torch.float64) if torch.is_tensor(b_idx) else np.asarray(b_idx, float)
        a_f = a_idx.to(torch.float64) if torch.is_tensor(a_idx) else np.asarray(a_idx, float)
        q_r = b_f * ds + off
        q_c = a_f * ds + off

        p_r, p_c = q_r, q_c
        for _ in range(max(1, iters)):
            d_r, d_c = self._displacement_raw(p_r, p_c)
            p_r = q_r - d_r
            p_c = q_c - d_c
        return (p_r - off) / ds, (p_c - off) / ds

    # -- diagnostics -----------------------------------------------------
    def summary(self, downsample: int = 1) -> str:
        H, W = self.raw_shape
        rr, cc = np.meshgrid(np.linspace(0, H - 1, 64), np.linspace(0, W - 1, 64),
                             indexing="ij")
        d_r, d_c = self._displacement_raw(rr, cc)
        mag = np.hypot(d_r, d_c)
        return (f"mode={self.mode} deg={self.deg} raw={H}x{W} "
                f"serial={self.detector_serial or '?'} | "
                f"|d| mean {mag.mean():.2f} px max {mag.max():.2f} px "
                f"(= {mag.mean()/max(downsample,1):.2f} / "
                f"{mag.max()/max(downsample,1):.2f} pipeline px)")


# ------------------------------------------------------------ scan lookup ---

def detector_serial_from_scan(scan_folder) -> str | None:
    """Read ``serialNumber`` from any acquisition frame's VFF header."""
    scan_folder = Path(scan_folder)
    for pattern in ("acq-*.vff", "proj-*.vff"):
        frames = sorted(scan_folder.glob(pattern))
        if frames:
            try:
                head = frames[0].open("rb").read(2048).replace(b"\x00", b"")
                for line in head.decode("latin-1").splitlines():
                    if line.startswith("serialNumber="):
                        return line.split("=", 1)[1].rstrip(";").strip()
            except OSError:
                return None
    return None


def resolve_detector_warp(geom_cfg: dict, scan_folder, raw_detector_shape,
                          downsample: int = 1) -> "DetectorWarp | None":
    """Build the warp for a scan from ``geometry.detector_warp`` config.

    Modes
    -----
    ``off``       disabled.
    ``auto``      use the calibration for this detector serial if one exists,
                  otherwise proceed uncorrected with a loud warning (default —
                  keeps scanners we have no calibration for runnable).
    ``nonaffine`` require a calibration; apply its non-affine part (validated).
    ``full``      require a calibration; apply the whole field. Only correct
                  when the calibration comes from the SAME geometry epoch as
                  the scan.

    Raises on a shape mismatch rather than silently applying a field fitted for
    a different detector format — a wrong warp is worse than none.
    """
    cfg = (geom_cfg or {}).get("detector_warp", {}) or {}
    mode = str(cfg.get("mode", "auto")).lower()
    if mode in ("off", "none", "false", "disabled"):
        return None

    explicit = cfg.get("path")
    serial = detector_serial_from_scan(scan_folder)
    apply_mode = "nonaffine" if mode in ("auto", "nonaffine") else "full"

    if explicit:
        warp = DetectorWarp.load(explicit, mode=apply_mode)
    elif serial:
        warp = DetectorWarp.for_detector(serial, mode=apply_mode,
                                         calib_dir=cfg.get("calib_dir"))
    else:
        warp = None

    if warp is None:
        msg = (f"  Detector warp: NO CALIBRATION for serial "
               f"{serial or 'unknown'} (looked in {CALIB_DIR}).")
        if mode == "auto":
            print(msg + " Proceeding UNCORRECTED — the projector assumes an "
                        "ideal linear detector grid.")
            return None
        raise FileNotFoundError(
            msg + f" geometry.detector_warp.mode={mode!r} requires one; set "
                  f"mode: auto or off, or supply an explicit path.")

    if tuple(warp.raw_shape) != tuple(raw_detector_shape):
        raise ValueError(
            f"Detector warp calibration was fitted on a "
            f"{warp.raw_shape[0]}x{warp.raw_shape[1]} detector but this scan "
            f"is {raw_detector_shape[0]}x{raw_detector_shape[1]}. Refusing to "
            f"apply a mismatched field.")

    print(f"  Detector warp: {warp.summary(downsample)}")
    if warp.source:
        print(f"                 from {warp.source}")
    return warp
