"""
Reference-free HU calibration: fit the volume's own histogram to two knowns.

A reconstruction produces attenuation that is, to a good approximation, an
affine image of the truth: ``mu_hat = g * mu + c``. The gain ``g`` comes from
flat-field/I0 error, detector gain and effective-spectrum mismatch; the offset
``c`` from scatter, truncation and the support bias. Two unknowns, so HU needs
two anchors. The classical one-point map

    HU = 1000 * (mu - mu_water) / mu_water

fits only the gain and silently assumes ``c == 0``. It also needs a scanner
constant that goes stale whenever the source or detector is replaced.

This module fits both, using only facts that hold for every scan we take:

  * **Air is exactly zero attenuation.** The lowest real mode of the histogram
    is air, and it belongs at -1000 HU. This anchor is free and exact, and it
    is what removes the offset.
  * **What we image is mostly soft tissue.** The largest remaining population
    is the specimen's bulk tissue, and we declare where it belongs (+120 HU by
    default, the scale this scanner's vendor reports for the same specimens).
    This anchor sets the gain.

The second one is an assumption, not a measurement, and it is the only thing
here that can be wrong — see ``TISSUE_HU_DEFAULT`` below for what it costs.

**Scale-freedom is the design constraint.** The anchors are located from the
shape of the histogram alone: the value range comes from percentiles of the
data, the smoothing and peak-separation scales are expressed in bins, and the
peak-selection thresholds are fractions of the histogram itself. No step ever
compares a voxel against an absolute number. So if the input is rescaled,
``mu -> a*mu + b``, both anchors move with it and the calibrated HU output is
unchanged. That is what makes the estimator work identically on raw mu, on a
half-calibrated volume, and on someone else's HU volume — and it is asserted
in the tests.

Note the older code did the opposite: it selected air with ``vol < -500`` and
read water out of a fixed 30x30 central box. The first is circular (it needs
the HU scale it is trying to find) and the second locks onto whatever happens
to sit in the middle of the volume, which for a mouse is lung.

Typical use, on raw attenuation straight out of a reconstruction::

    anchors = find_attenuation_anchors(mu)
    print(format_calibration(anchors))
    volume_hu = anchors.apply(mu)
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

# Air. Exact by definition of the Hounsfield scale, and exact in the physics
# too: mu_air is ~0.0002 mm^-1 against water's ~0.02, i.e. 1 % of the air-water
# span, well under the width of the air peak itself.
AIR_HU_DEFAULT = -1000.0

# Water, by definition of the Hounsfield scale. Used by the fixed (one-point)
# path, where the second anchor is a material of KNOWN attenuation rather than
# a measured tissue peak.
WATER_HU = 0.0

# Bulk soft tissue. This is the assumption that sets the gain, so it is worth
# being explicit about what it buys and what it costs.
#
# +120 HU is a CONVENTION, chosen to match the scale this scanner's vendor
# reports for the same specimens — not a measurement. Measured on the vendor's
# own reconstructions (Scan_1510 thumbnail +115, Scan_1510 mouse ROI +139,
# Scan_1955 mouse ROI +100), the mouse soft-tissue mode sits at +100 to +139 HU,
# while the vendor's multi-insert phantom reconstruction puts water at -5 HU.
# Whether mouse tissue really reads ~+120 HU on this beam, or the vendor's mouse
# gain is simply off by that much, CANNOT BE DECIDED WITHOUT A PHANTOM: under a
# one-point map air lands on -1000 for any gain whatsoever (because mu_air = 0),
# so an air anchor that looks perfect is no evidence at all about the scale.
# Adopting the vendor's number keeps our volumes comparable with the vendor's
# and with the historical metrics computed against them.
#
# What the anchor buys, whatever its value, is comparability: every algorithm
# and every scan lands on the same scale, so HU differences between
# reconstructions mean something even though the absolute zero rests on an
# assumption. Changing it rescales every output linearly and nothing else.
#
# NOTE this default assumes the bulk of the specimen is soft tissue. It is
# wrong for a WATER PHANTOM, whose bulk belongs at 0 HU by definition — pass
# ``tissue_hu=WATER_HU`` for those.
TISSUE_HU_DEFAULT = 120.0


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class HUAnchors:
    """The fitted map, its inputs, and everything needed to audit it."""

    value_air: float
    value_tissue: float
    air_hu: float = AIR_HU_DEFAULT
    tissue_hu: float = TISSUE_HU_DEFAULT
    method: str = "histogram"
    warnings: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def scale(self) -> float:
        """HU per unit of input (per mm^-1 when the input is mu)."""
        return (self.tissue_hu - self.air_hu) / (self.value_tissue - self.value_air)

    @property
    def offset(self) -> float:
        """HU at input zero, i.e. ``HU = scale * value + offset``."""
        return self.air_hu - self.scale * self.value_air

    @property
    def implied_mu_water(self) -> float:
        """The mu_water the one-point map would have needed to agree with this
        fit. Only meaningful when the input really is mu; it is reported so a
        run can be compared against the old hardcoded 0.0219 mm^-1."""
        return float(self.value_tissue - self.value_air)

    def apply(self, values, dtype=np.float32) -> np.ndarray:
        """Map input values onto the HU scale. No clipping — clamping is a
        storage concern and belongs at write time, not here."""
        values = np.asarray(values)
        return (values * self.scale + self.offset).astype(dtype)


# --------------------------------------------------------------------------
# Anchor search
# --------------------------------------------------------------------------

def _subsample(values, max_samples):
    """Deterministic stride down to at most ``max_samples`` finite values.

    A stride, not a random draw, so the same volume always yields the same
    anchors — reproducibility matters more here than sampling purity, and the
    populations involved are 10^7 voxels either way.
    """
    flat = np.asarray(values).reshape(-1)
    if max_samples and flat.size > max_samples:
        flat = flat[:: int(np.ceil(flat.size / max_samples))]
    flat = flat[np.isfinite(flat)]
    return flat


def _smooth(counts, sigma_bins):
    """Gaussian blur along the histogram. Local import so the module stays
    usable (minus smoothing quality) if scipy is ever unavailable."""
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(counts.astype(np.float64), float(sigma_bins))
    except ImportError:  # pragma: no cover - scipy is a hard dep in practice
        k = int(max(1, round(3 * sigma_bins)))
        w = np.exp(-0.5 * (np.arange(-k, k + 1) / float(sigma_bins)) ** 2)
        w /= w.sum()
        return np.convolve(counts.astype(np.float64), w, mode="same")


def _local_maxima(h):
    """Indices of strict local maxima, plateau-safe."""
    idx = []
    n = len(h)
    i = 1
    while i < n - 1:
        if h[i] > h[i - 1]:
            j = i
            while j < n - 1 and h[j + 1] == h[i]:
                j += 1
            if j < n - 1 and h[j] > h[j + 1]:
                idx.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return np.asarray(idx, dtype=int)


def _prominence(h, peak):
    """Topological prominence of ``peak``, and the basin it dominates.

    Walk out in both directions until the curve rises above the peak (or the
    histogram ends), tracking the lowest point reached on each side. The
    prominence is the peak's height above the higher of those two saddles, and
    the basin runs between them.

    The "walk until something higher" part is what makes this the right
    measure here. A small ripple riding on the broad soft-tissue hump is
    dominated by the hump's own apex, so it gets a tiny prominence and a tiny
    basin, while the apex inherits the whole hump — which is exactly the
    discrimination we need, because a naive descending-valley walk would give
    the ripple the hump's entire mass and its own negligible prominence.
    """
    n = len(h)
    lo = peak
    while lo > 0 and h[lo - 1] <= h[peak]:
        lo -= 1
    hi = peak
    while hi < n - 1 and h[hi + 1] <= h[peak]:
        hi += 1
    left_base = lo + int(np.argmin(h[lo:peak + 1]))
    right_base = peak + int(np.argmin(h[peak:hi + 1]))
    saddle = max(h[left_base], h[right_base])
    return float(h[peak] - saddle), left_base, right_base


def find_attenuation_anchors(
    values,
    *,
    air_hu: float = AIR_HU_DEFAULT,
    tissue_hu: float = TISSUE_HU_DEFAULT,
    bins: int = 1024,
    smooth_bins: float = 3.0,
    clip_pct: Sequence[float] = (0.02, 99.98),
    pad_frac: float = 0.03,
    min_air_mass_frac: float = 0.005,
    min_air_prominence: float = 0.25,
    gap_frac: float = 0.02,
    min_tissue_prominence: float = 0.05,
    min_tissue_mass_frac: float = 0.005,
    max_samples: int = 60_000_000,
) -> HUAnchors:
    """Locate the air and bulk-tissue anchors from the histogram's shape.

    ``values`` is raw attenuation (or any affine image of it — the result is
    the same map either way). Every tunable below is expressed in bins or in
    fractions of the histogram, never in physical units, which is what keeps
    the estimator equivariant.

    Args:
        air_hu, tissue_hu: where the two anchors should land.
        bins: histogram resolution over the robust value range.
        smooth_bins: Gaussian width, in bins, applied before peak finding.
            Suppresses noise ripple without moving a peak that is many bins
            wide.
        clip_pct: percentile range histogrammed, so a few extreme voxels
            cannot stretch the axis.
        pad_frac: extra range added on both sides, as a fraction of the span.
            Without it a peak that sits at the very bottom of the data — which
            is exactly where air sits — falls in bin 0 and has no left
            neighbour to be a maximum against.
        min_air_mass_frac: an air candidate's basin must hold this share of
            the volume. Together with min_air_prominence this replaces a
            height threshold, which scaled with peak width and so rejected a
            narrow air peak in a tight crop.
        min_air_prominence: an air candidate's prominence as a fraction of its
            own height — what separates a population from a noise ripple in
            the sparse low tail.
        gap_frac: minimum separation from the air peak, as a fraction of the
            bin count, so the tissue anchor cannot land on air's own
            partial-volume skirt.
        min_tissue_prominence: the tissue peak's prominence as a fraction of
            its own height. Below this it is a ripple on a monotone decay
            rather than a mode, and the fit is flagged.
        min_tissue_mass_frac: the tissue basin must hold at least this share
            of all voxels.
        max_samples: stride the input down to at most this many values.

    Returns:
        HUAnchors. Check ``.warnings`` — the estimator reports rather than
        raises, so a caller can decide whether a flagged fit is still usable.
    """
    warnings_: list = []
    flat = _subsample(values, max_samples)
    if flat.size == 0:
        raise ValueError("no finite values to calibrate from")

    lo, hi = (float(x) for x in np.percentile(flat, list(clip_pct)))
    span = hi - lo
    if not span > 0:
        raise ValueError(
            "the volume has no dynamic range (all values effectively equal) — "
            "nothing to calibrate")
    lo -= pad_frac * span
    hi += pad_frac * span

    counts, edges = np.histogram(flat, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = _smooth(counts, smooth_bins)

    # --- clipping check -------------------------------------------------
    # Not a histogram question but a data question: a hard clamp leaves a
    # large share of voxels sitting on exactly one value. This matters here
    # more than anywhere else, because the value it clamps them to is usually
    # near air — so the clamp masquerades as the air anchor and the fit is
    # quietly wrong rather than loudly broken.
    vmin, vmax = float(flat.min()), float(flat.max())
    frac_at_min = float(np.count_nonzero(flat == vmin) / flat.size)
    frac_at_max = float(np.count_nonzero(flat == vmax) / flat.size)
    if frac_at_min > 0.01:
        warnings_.append(
            f"{100 * frac_at_min:.1f} % of voxels sit on exactly {vmin:.1f} — "
            f"this volume is clipped from below, and the air anchor is "
            f"measuring the clamp rather than air")
    if frac_at_max > 0.001:
        warnings_.append(
            f"{100 * frac_at_max:.3f} % of voxels sit on exactly {vmax:.1f} — "
            f"this volume is saturated from above")

    # --- air: the lowest local maximum that is a real population --------
    # Qualified by MASS and prominence rather than by height. Height relative
    # to the tallest bin sounds equivalent but is not: it scales with the
    # peak's width, so it rejects a narrow-but-real air peak in a tight crop
    # while accepting a broad shoulder. Mass is the physical quantity — "how
    # much of the volume is this?" — and prominence is what separates a
    # population from a ripple.
    maxima = _local_maxima(h)
    air_i = None
    for p in maxima:
        p = int(p)
        if h[p] <= 0:
            continue
        prom, b_lo, b_hi = _prominence(h, p)
        mass = float(counts[b_lo:b_hi + 1].sum()) / flat.size
        if mass >= min_air_mass_frac and prom / h[p] >= min_air_prominence:
            air_i = p
            break
    if air_i is None:
        raise ValueError(
            "no population in this volume is large and distinct enough to be "
            "the air anchor; the histogram has no usable mode")

    # --- tissue: the biggest population above air -----------------------
    # Chosen by basin MASS rather than by height or prominence. Mass is what
    # "mostly soft tissue" actually means, and it is the criterion that
    # rejects both lung (a smaller population between air and tissue) and
    # bone (a much smaller one above it). Height would follow whichever peak
    # happens to be narrow; prominence relative to the global maximum misses
    # the tissue peak entirely whenever air outnumbers it ~100:1, which is the
    # normal situation in a full-FOV reconstruction.
    # Ripples are filtered out FIRST, by prominence relative to their own
    # height, and only the survivors compete on mass. Doing it the other way
    # round hands the win to whichever ripple happens to sit on the tissue
    # hump: it inherits the hump's mass while having no prominence of its own.
    gap = max(1, int(gap_frac * bins))
    tissue_i, best_mass, basin, best_ratio = None, -1.0, None, 0.0
    fallback = None
    for p in maxima:
        p = int(p)
        if p <= air_i + gap or h[p] <= 0:
            continue
        prom, b_lo, b_hi = _prominence(h, p)
        ratio = prom / h[p]
        b_lo = max(b_lo, air_i + 1)
        mass = float(counts[b_lo:b_hi + 1].sum())
        if fallback is None or mass > fallback[1]:
            fallback = (p, mass, (b_lo, b_hi), ratio)
        if ratio < min_tissue_prominence:
            continue
        if mass > best_mass:
            best_mass, tissue_i, basin, best_ratio = mass, p, (b_lo, b_hi), ratio

    if tissue_i is None:
        # Nothing above air is a real mode. Fall back to the largest
        # population so the caller still gets a usable (if flagged) map
        # rather than an exception — a monotone decay with no tissue hump is
        # a bad volume, not a bad calibrator.
        if fallback is None:
            raise ValueError(
                "found the air anchor but no separate population above it; "
                "the volume may be air-only, or too small a crop to contain "
                "bulk tissue")
        tissue_i, best_mass, basin, best_ratio = fallback
    prominence_ratio = float(best_ratio)
    mass_frac = best_mass / flat.size
    if prominence_ratio < min_tissue_prominence:
        warnings_.append(
            f"the tissue anchor is only {100 * prominence_ratio:.1f} % "
            f"prominent above its own shoulders — the histogram has no clear "
            f"soft-tissue mode, so the gain is poorly determined")
    if mass_frac < min_tissue_mass_frac:
        warnings_.append(
            f"the tissue population is only {100 * mass_frac:.2f} % of the "
            f"volume — too little bulk tissue in this crop to anchor the gain "
            f"reliably")

    # Is the thing we called air actually air? The height threshold that
    # rejects noise ripple can also reject a genuine but tiny air population,
    # and then the lowest SURVIVING peak is soft tissue — at which point the
    # fit silently anchors tissue to -1000 and bone to 0, which looks perfectly
    # well-conditioned by every other measure.
    #
    # The signature is specific: a real population, with real prominence and
    # real mass, sitting BELOW the chosen anchor. In a volume that does contain
    # air there is nothing below the air peak but its own noise skirt, whose
    # ripples carry a millionth of the voxels; the mass floor is what separates
    # those from a true population that was merely outvoted on height.
    for p in maxima:
        p = int(p)
        if p >= air_i or h[p] <= 0:
            continue
        prom, b_lo, b_hi = _prominence(h, p)
        below_mass = float(counts[b_lo:b_hi + 1].sum()) / flat.size
        if prom / h[p] >= 0.25 and below_mass >= 0.001:
            warnings_.append(
                f"a distinct population holding {100 * below_mass:.2f} % of "
                f"voxels sits BELOW the chosen air anchor — that lower "
                f"population is probably the real air, and this crop contains "
                f"too little of it to outvote the bulk; the anchors are likely "
                f"one material off")
            break

    # An aggressive export ROI can crop away the air the offset anchor needs.
    air_frac = float(counts[:air_i + 1].sum() / flat.size)
    if air_frac < 0.02:
        warnings_.append(
            f"only {100 * air_frac:.2f} % of voxels lie at or below the air "
            f"anchor — this crop barely contains any air, so the offset is "
            f"weakly determined; calibrate on the full reconstruction domain "
            f"if possible")

    air_value = float(centers[air_i])
    tissue_value = float(centers[tissue_i])
    if not tissue_value > air_value:
        raise ValueError("tissue anchor did not land above the air anchor")

    return HUAnchors(
        value_air=air_value,
        value_tissue=tissue_value,
        air_hu=float(air_hu),
        tissue_hu=float(tissue_hu),
        method="histogram",
        warnings=warnings_,
        diagnostics=dict(
            counts=counts,
            centers=centers,
            smoothed=h,
            air_index=air_i,
            tissue_index=tissue_i,
            tissue_basin=basin,
            tissue_mass_frac=float(mass_frac),
            tissue_prominence_ratio=prominence_ratio,
            air_frac=air_frac,
            frac_at_min=frac_at_min,
            frac_at_max=frac_at_max,
            value_min=vmin,
            value_max=vmax,
            n_samples=int(flat.size),
            bins=int(bins),
        ),
    )


def resolve_anchors(values, mode: str = "auto", *,
                    mu_water: Optional[float] = None,
                    tissue_hu: Optional[float] = None,
                    verbose: bool = False) -> HUAnchors:
    """Mode string -> fitted anchors. The ONE place the choice is made.

    ``mode='auto'`` fits both anchors from this volume's own histogram;
    ``'fixed'`` pins the gain to ``mu_water`` and air to zero attenuation,
    reproducing the classical one-point map through the same code path.

    Every consumer goes through here — the drivers via
    ``scan_setup.postprocess_and_save``, muNeRF via ``inr_pipeline.infer.to_hu``
    — so a volume calibrated by one lands on exactly the scale the other would
    have produced. Both used to dispatch on their own mode names, which is how
    muNeRF came to anchor bulk tissue at 0 HU while the drivers used +120.
    """
    if mode == "fixed":
        if mu_water is None:
            raise ValueError("hu_calibration='fixed' needs an explicit mu_water")
        # tissue_hu deliberately does NOT apply here: this mode's second anchor
        # is water at 0 HU by definition, not a measured tissue peak. Say so
        # rather than ignoring the flag in silence.
        if tissue_hu is not None and verbose:
            print(f"  NOTE: tissue_hu {float(tissue_hu):.0f} ignored — it "
                  f"places the measured bulk-tissue peak, and "
                  f"hu_calibration='fixed' anchors WATER (0 HU) via mu_water "
                  f"instead.")
        return fixed_anchors(float(mu_water))
    if mode == "auto":
        return find_attenuation_anchors(
            values,
            tissue_hu=(TISSUE_HU_DEFAULT if tissue_hu is None
                       else float(tissue_hu)))
    raise ValueError(f"unknown hu_calibration mode {mode!r} "
                     f"(expected 'auto' or 'fixed')")


def fixed_anchors(mu_water: float, mu_air: float = 0.0, *,
                  air_hu: float = AIR_HU_DEFAULT,
                  water_hu: float = WATER_HU) -> HUAnchors:
    """The one-point map, expressed in the same two-anchor form.

    With ``mu_air=0`` and the default targets this reproduces
    ``HU = 1000 * (mu - mu_water) / mu_water`` exactly, so a run can pin the
    scale to a known constant and still go through the same code path, the
    same logging and the same figure as a self-calibrated run.

    Note the second anchor here is WATER, at 0 HU by the definition of the
    Hounsfield scale — not the measured bulk-tissue peak that the histogram
    fit uses, and so not subject to ``TISSUE_HU_DEFAULT``. The two modes
    anchor different materials: ``find_attenuation_anchors`` says "the bulk of
    this specimen is soft tissue and belongs at +120 HU", while this says "the
    material with attenuation mu_water is water and belongs at 0 HU". Letting
    the tissue convention leak in here would silently change the gain of an
    escape hatch whose entire purpose is to reproduce a known map.
    """
    return HUAnchors(
        value_air=float(mu_air),
        value_tissue=float(mu_air + mu_water),
        air_hu=float(air_hu),
        tissue_hu=float(water_hu),
        method="fixed",
        warnings=[],
        diagnostics=dict(mu_water=float(mu_water)),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

# Landmarks that the fit does NOT use, so their positions afterwards are free
# evidence about whether the calibration is sane. Ranges are deliberately
# generous — they are meant to catch a scale that is wrong by tens of percent,
# not to police normal biological variation.
LANDMARKS = (
    ("lung", -900.0, -300.0),
    ("fat", -250.0, -20.0),
    ("trabecular_bone", 200.0, 900.0),
    ("cortical_bone", 900.0, 2500.0),
)


def landmark_check(volume_hu, anchors: Optional[HUAnchors] = None,
                   max_samples: int = 60_000_000) -> dict:
    """Where the non-anchor tissue classes land after calibration.

    Reports the share of voxels in each landmark band plus the calibrated
    percentiles. Nothing here feeds back into the fit; it exists so a run can
    be judged on evidence the fit did not consume.
    """
    flat = _subsample(volume_hu, max_samples)
    out = {}
    for name, lo, hi in LANDMARKS:
        out[f"hu/frac_{name}"] = float(
            np.count_nonzero((flat >= lo) & (flat < hi)) / flat.size)
    for q in (1, 50, 99, 99.9):
        out[f"hu/p{q:g}"] = float(np.percentile(flat, q))
    out["hu/min"] = float(flat.min())
    out["hu/max"] = float(flat.max())
    return out


def calibration_scalars(anchors: HUAnchors) -> dict:
    """Flat, loggable summary of the fit."""
    d = anchors.diagnostics
    out = {
        "hu/anchor_air_value": anchors.value_air,
        "hu/anchor_tissue_value": anchors.value_tissue,
        "hu/target_air": anchors.air_hu,
        "hu/target_tissue": anchors.tissue_hu,
        "hu/scale": anchors.scale,
        "hu/offset": anchors.offset,
        "hu/implied_mu_water": anchors.implied_mu_water,
        "hu/n_warnings": len(anchors.warnings),
    }
    for key in ("tissue_mass_frac", "tissue_prominence_ratio", "air_frac",
                "frac_at_min", "frac_at_max"):
        if key in d:
            out[f"hu/{key}"] = float(d[key])
    return out


def format_calibration(anchors: HUAnchors) -> str:
    """Human-readable block for the driver's stdout."""
    d = anchors.diagnostics
    lines = [
        f"  Method:        {anchors.method}",
        f"  Air anchor:    {anchors.value_air:12.6f}  ->  "
        f"{anchors.air_hu:+.1f} HU",
        f"  Tissue anchor: {anchors.value_tissue:12.6f}  ->  "
        f"{anchors.tissue_hu:+.1f} HU",
        f"  Map:           HU = {anchors.scale:.4f} * value "
        f"{anchors.offset:+.2f}",
        f"  Implied mu_water: {anchors.implied_mu_water:.6f} "
        f"(one-point equivalent)",
    ]
    if "tissue_mass_frac" in d:
        lines.append(
            f"  Tissue population: {100 * d['tissue_mass_frac']:.1f} % of "
            f"voxels, {100 * d['tissue_prominence_ratio']:.0f} % prominent; "
            f"air below anchor: {100 * d['air_frac']:.1f} %")
    for w in anchors.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


def calibration_figure(anchors: HUAnchors, title: str = "HU calibration"):
    """Histogram with the two anchors marked, on the calibrated HU axis.

    Returns None for a fit that carries no histogram (``fixed_anchors``).
    """
    d = anchors.diagnostics
    if "centers" not in d:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hu_axis = d["centers"] * anchors.scale + anchors.offset
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hu_axis, d["counts"], lw=0.7, color="0.7", label="histogram")
    ax.plot(hu_axis, d["smoothed"], lw=1.2, color="0.25", label="smoothed")
    ax.set_yscale("log")

    b_lo, b_hi = d["tissue_basin"]
    ax.axvspan(hu_axis[b_lo], hu_axis[b_hi], color="tab:red", alpha=0.10,
               label="tissue basin")
    ax.axvline(anchors.air_hu, color="tab:blue", lw=1.6,
               label=f"air anchor -> {anchors.air_hu:+.0f} HU")
    ax.axvline(anchors.tissue_hu, color="tab:red", lw=1.6,
               label=f"tissue anchor -> {anchors.tissue_hu:+.0f} HU")
    for name, lo, hi in LANDMARKS:
        ax.axvspan(lo, hi, color="tab:green", alpha=0.05)

    ax.set_xlabel("HU (after calibration)")
    ax.set_ylabel("voxels")
    ax.set_title(f"{title} — {anchors.method}, "
                 f"HU = {anchors.scale:.4f}*value {anchors.offset:+.1f}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
