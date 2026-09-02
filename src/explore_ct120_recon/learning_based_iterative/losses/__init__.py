"""Losses for learning-based reconstruction — one registry, selected by name.

Reconstruction-as-optimisation needs two independent choices: how a predicted
line integral is compared with a measured one (the DATA TERM), and what is
assumed about the volume where the data does not determine it (the PRIORS).
Both are backend-agnostic — a voxel grid, a hash-grid INR and anything else
consume the identical functions — so they live here rather than inside any one
reconstructor.

The registry exists so a loss is a NAME. Before it, choosing a data term meant
an if/elif chain repeated at each call site, which is why they drifted and why
adding one meant editing several places. Now:

    from .losses import build_data_term
    loss_fn = build_data_term("msssim", data_range=..., patch=True)
    loss = loss_fn(pred, target)

``build_data_term`` returns a plain callable ``(pred, target) -> scalar``, so a
trainer never branches on the name. ``DATA_TERMS`` is the authoritative list;
``describe_data_terms()`` renders it for a CLI ``--help``.

Two families, and the difference matters when reading a config:

* DATA TERMS are mutually exclusive — exactly one is in force.
* PRIORS are additive and independent — any number, each with its own weight.
  They are exported here but not registered, because a prior needs more than a
  name (a target, a patch sampler, a schedule), so the trainer wires them
  explicitly rather than by string.

Adding a data term is one entry in ``DATA_TERMS`` and nothing else.
"""

from __future__ import annotations

from .data_terms import (
    _build_ramp_kernel,
    build_phase_wiener_gate,
    build_wiener_kernel,
    filtered_mse,
    make_huber_loss,
    mse,
    ssim_loss,
    weighted_mse,
    wiener_mse,
)
from .priors import (
    bone_anchors_to_mu,
    bone_band_fraction,
    bone_fraction,
    build_census_quantile_weights,
    build_census_target,
    build_fdk_edge_weight,
    build_radial_bins,
    build_spectral_target,
    concave_gradient_3d,
    estimate_bone_anchors_hu,
    modica_mortola_3d,
    sample_tv_patch_xyz,
    sparsity_l1,
    spectral_match_loss,
    spectral_split_db,
    tv_3d_anisotropic,
    volume_norm_l2,
    wasserstein_census,
)


# --------------------------------------------------------------------------
# Data-term registry
# --------------------------------------------------------------------------

def _mse_factory(**_):
    return mse


def _weighted_factory(**_):
    return weighted_mse


def _huber_factory(*, delta=None, huber_sigma_mult=1.345, **_):
    return make_huber_loss(delta=delta, sigma_mult=huber_sigma_mult)


def _filtered_factory(*, ramp_kernel=None, filtered_weight=1.0, **_):
    """Ramp-filtered L2 added to plain L2.

    Needs the kernel prebuilt (``_build_ramp_kernel``) because it depends on the
    detector row length and the ramp exponent, which are properties of the scan
    and the schedule rather than of the loss.

    The plain L2 is kept alongside deliberately: a ramp has no DC response, so a
    filtered-only objective cannot see a constant offset in the volume.
    """
    if ramp_kernel is None:
        raise ValueError(
            "loss 'filtered' needs ramp_kernel=_build_ramp_kernel(...); it "
            "depends on the detector row length, which the loss cannot know.")

    def _fn(pred, target):
        return mse(pred, target) + filtered_weight * filtered_mse(
            pred, target, ramp_kernel)
    return _fn


def _wiener_factory(*, wiener_kernel=None, wiener_weight=1.0, **_):
    """Ramp-filtered L2 gated by measured SNR, added to plain L2."""
    if wiener_kernel is None:
        raise ValueError(
            "loss 'wiener' needs a kernel built from measured SNR: pass "
            "wiener_kernel=build_wiener_kernel(...). Without it there is no "
            "SNR estimate and the gate has nothing to gate on.")

    def _fn(pred, target):
        return mse(pred, target) + wiener_weight * wiener_mse(
            pred, target, wiener_kernel)
    return _fn


def _sart_factory(*, chord_state=None, sart_floor_frac=1e-3,
                  sart_clamp_lo=0.25, sart_clamp_hi=4.0,
                  sart_reduction="mean", **_):
    """SART/SIRT row weighting: R = diag(1/L_i).

    Needs ``chord_state`` — a dict the sampler refreshes with this batch's row
    lengths under the key ``"chord"`` — because L_i is a property of the RAYS,
    not of the loss, and it changes every step. The indirection keeps the
    registry's uniform ``(pred, target)`` contract instead of giving this one
    term a wider signature that every trainer would have to special-case.

    ``sart_reduction`` selects the mean (default, MSE-scaled — a learning rate
    transfers to and from ``mse``) or the sum (the classical misfit, whose
    gradient is exactly ``-A^T R r``). ``--emulate-sart`` sets the sum, because
    only then does the C preconditioner's absolute scale make ``lambda = 1``
    one classical update.

    See ``learning_based_iterative.sart`` for what R is and for why L is measured
    over the object ROI rather than the full model domain.
    """
    if chord_state is None:
        raise ValueError(
            "loss 'sart' needs chord_state={} — a dict the sampler fills with "
            "this batch's row lengths (sart.ray_support_lengths). Without it "
            "there is no R and the term is just MSE.")

    def _fn(pred, target):
        from ..sart import sart_weighted_mse
        chord = chord_state.get("chord")
        if chord is None:
            raise RuntimeError(
                "chord_state is empty — the sampler must populate "
                "chord_state['chord'] before the loss is evaluated.")
        loss, _w = sart_weighted_mse(
            pred, target, chord, floor_frac=sart_floor_frac,
            w_clamp_lo=sart_clamp_lo, w_clamp_hi=sart_clamp_hi,
            reduction=sart_reduction)
        return loss
    return _fn


def _structural_factory(*, data_range, ms, ssim_weight=1.0, window_size=11,
                        sigma=1.5, ms_weights=None, with_mse=True, **_):
    """1 - SSIM (or 1 - MS-SSIM) on a projection patch, plus plain L2.

    The L2 term anchors the absolute level: SSIM's luminance factor is a ratio,
    so a structural-only objective is nearly invariant to a global scale on the
    volume. Pass ``with_mse=False`` to score structure alone.
    """
    def _fn(pred, target):
        s = ssim_weight * ssim_loss(pred, target, data_range=data_range,
                                    window_size=window_size, sigma=sigma,
                                    ms=ms, ms_weights=ms_weights)
        return mse(pred, target) + s if with_mse else s
    return _fn


def _ssim_factory(**kw):
    return _structural_factory(ms=False, **kw)


def _msssim_factory(**kw):
    return _structural_factory(ms=True, **kw)


#: name -> factory. The factory takes keyword options and returns a callable
#: ``(pred, target) -> scalar``. Unknown keywords are ignored, so one options
#: dict can be passed whichever term is selected.
DATA_TERMS = {
    "mse": _mse_factory,
    "weighted": _weighted_factory,
    "huber": _huber_factory,
    "filtered": _filtered_factory,
    "wiener": _wiener_factory,
    "sart": _sart_factory,
    "ssim": _ssim_factory,
    "msssim": _msssim_factory,
}

#: One line per term, for CLI help and for logging what a run actually used.
DATA_TERM_HELP = {
    "mse": "plain L2 on line integrals; the objective classical SIRT descends",
    "weighted": "L2 weighted by transmission, i.e. by measurement confidence",
    "huber": "L2 near zero, L1 in the tail; crossover from the residual's own "
             "robust spread, bounding the influence of bad rays",
    "filtered": "L2 + L2 on ramp-filtered detector rows, weighting spatial "
                "frequencies the way the analytic inverse does",
    "wiener": "filtered, but gated by measured SNR so the ramp's gain is spent "
              "on the recoverable band (needs wiener_kernel=)",
    "sart": "row-weighted L2, w_i = 1/L_i — the R of the classical SIRT/SART "
            "update; see --emulate-sart, which also adds the C preconditioner "
            "and switches to SGD",
    "ssim": "L2 + (1 - SSIM) on a projection patch; sensitive to local "
            "contrast and structure, which per-pixel L2 is not",
    "msssim": "L2 + (1 - MS-SSIM), the multi-scale variant",
}

DEFAULT_DATA_TERM = "mse"


def build_data_term(name: str = DEFAULT_DATA_TERM, **options):
    """Resolve a data-term name to a callable ``(pred, target) -> scalar``.

    ``options`` is passed to every factory and each takes what it needs, so a
    caller can hand over one options dict without knowing which term is
    selected. Raises on an unknown name rather than falling back to MSE: a
    silently ignored loss selection is a run that did not do what its config
    says.
    """
    key = str(name).strip().lower()
    if key not in DATA_TERMS:
        raise ValueError(
            f"unknown data term {name!r}. Available: "
            f"{', '.join(sorted(DATA_TERMS))}")
    return DATA_TERMS[key](**options)


def describe_data_terms(indent: str = "  ") -> str:
    """The registry as text, for a CLI epilog."""
    width = max(len(k) for k in DATA_TERMS)
    return "\n".join(
        f"{indent}{k:<{width}}  {DATA_TERM_HELP.get(k, '')}"
        for k in DATA_TERMS)


__all__ = [
    # registry
    "DATA_TERMS", "DATA_TERM_HELP", "DEFAULT_DATA_TERM",
    "build_data_term", "describe_data_terms",
    # data terms
    "mse", "weighted_mse", "make_huber_loss", "filtered_mse", "wiener_mse",
    "ssim_loss", "_build_ramp_kernel", "build_wiener_kernel",
    "build_phase_wiener_gate",
    # priors
    "wasserstein_census", "build_census_target",
    "build_census_quantile_weights",
    "tv_3d_anisotropic", "sample_tv_patch_xyz",
    "spectral_match_loss", "build_spectral_target", "build_radial_bins",
    "spectral_split_db",
    "build_fdk_edge_weight",
    "sparsity_l1", "volume_norm_l2",
    "modica_mortola_3d", "concave_gradient_3d",
    "estimate_bone_anchors_hu", "bone_anchors_to_mu", "bone_fraction",
    "bone_band_fraction",
]
