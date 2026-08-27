"""When to stop iterating — one implementation, three criteria, every backend.

Iterative and learning-based reconstruction both SEMI-CONVERGE: the residual
keeps falling long after the reconstruction has started getting worse, because
the extra fit is going into noise. So "run to convergence" is the wrong
instruction and every backend needs a stopping rule. Three are offered here,
and they are not redundant — they disagree about what "worse" means:

* **Held-out projection** (`HoldoutScorer` + `EarlyStopper`). Withhold one
  projection, forward-project the current estimate at that angle, and score it.
  Measures GENERALISATION directly, which is the thing you actually want, and
  needs no assumption about the noise. Costs one projection's worth of data and
  one forward projection per checkpoint. SSIM, PSNR and MSE are all available as
  the criterion, because they peak at different iterations: MSE is the objective
  and peaks last, SSIM is structural and turns over earliest.

* **L-curve** (`LCurve`). Plot the residual norm against the solution norm in
  log-log as the iteration advances; the corner is where further residual
  reduction starts buying disproportionate solution growth, i.e. where noise
  amplification takes over. Needs **no held-out data at all**, which is its real
  advantage — on a short scan, or any scan where you cannot afford to drop an
  angle, it is the only one of the three that applies. Classical: Hansen &
  O'Leary (1993).

* **Plateau LR reduction** (`PlateauLRReducer`). Learning-based backends only:
  the *closed-loop* half of stopping. An open-loop LR schedule assumes a fixed
  horizon the run never reaches, so the annealing phase never happens. This
  drives the LR down on the same signal the stopper watches, and the loop stops
  only once the LR is at its floor and the metric still will not improve.

WHY THIS FILE EXISTS. There were three separate implementations of the first
criterion — TIGRE's inside `iterative/tigre/reconstructor.py`, the voxel
trainer's inline, and muNeRF's `inr_pipeline/early_stop.py` — and ASTRA had
none. They differed in ways that mattered:

* TIGRE scored its stopping SSIM with **skimage's default 7x7 uniform window**
  on the FULL projection, while the `diag/ssim` it logged to W&B came from
  `ct_core.ssim`'s 11x11 Gaussian window on the domain-covered CROP. The number
  that decided when to stop was therefore not the number you could see, and
  neither was comparable with any other backend's.
* The voxel trainer stopped on held-out MSE but then returned the LAST iterate,
  not the best one — so it detected the turnover and kept the volume from after
  it.

One scorer, one tracker, one figure. `inr_pipeline/early_stop.py` is now an
alias to this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Metric -> optimisation direction. SSIM/PSNR: higher is better. MSE: lower.
_MODE = {"ssim": "max", "psnr": "max", "mse": "min"}

#: Criteria a caller may select for the held-out rule.
STOP_METRICS = tuple(_MODE)

# --------------------------------------------------------------------------
# How long to wait before believing a plateau
# --------------------------------------------------------------------------
# Patience used to be an absolute number of EVALUATIONS, which is not a
# well-defined amount of patience: 8 evals at eval_every=250 is 10% of a
# 20000-iteration run and 100% of a 2000-iteration one, so the same default
# meant different things in every config. Both numbers below are FRACTIONS OF
# THE SCHEDULED RUN, which is the only definition that transfers.
#
# They are deliberately generous, because the two errors are not symmetric.
# `EarlyStopper` captures the best state and the callers restore it, so
# stopping LATE costs wall-clock and nothing else — the volume returned is the
# same one. Stopping EARLY returns a premature reconstruction and hides
# whatever the run would have done next. MEASURED case that set these: an
# --emulate-sart run at lambda=1 plateaued from iteration 500 while its cosine
# schedule still intended to reduce the step by three orders of magnitude; the
# old default ended it at iteration 2500 of 20000 (12.5%), during the
# high-step phase, so the annealing the schedule existed for never happened.

#: Floor on patience in evaluations, so short runs behave as they always did.
DEFAULT_PATIENCE_EVALS = 8

#: Floor on patience when a PlateauLRReducer is driving the LR. Lower than the
#: open-loop floor on purpose: by the time the stop counter is even allowed to
#: run, the LR is at its floor, which means the reducer has ALREADY observed a
#: plateau six separate times and annealed the step size by 50x. Demanding
#: another eight in a row re-asks a question that has been answered. MEASURED
#: on run ny96yzab: 8 -> 4 is the single largest saving available (11.7 h ->
#: 8.6 h on its own, 5.1 h combined with a tolerance), for 0.0004 of held-out
#: SSIM on a projection already fitted below its own noise floor.
CLOSED_LOOP_PATIENCE_EVALS = 4

#: A plateau must cover this fraction of the scheduled run to end it.
PATIENCE_FRACTION = 0.25

#: No rule may end a run before this fraction of the schedule has elapsed.
#: At 0.5 a cosine schedule has annealed to half its peak step, so a plateau
#: past this point is a property of the reconstruction rather than of a step
#: size the schedule was still going to shrink.
MIN_PROGRESS_FRACTION = 0.5

# How much better than the running best a value has to be to COUNT as better,
# as a fraction of the best itself. Zero — "any decrease at all is an
# improvement" — is what a plateau rule cannot survive: MEASURED on run
# ny96yzab (Scan_1510 voxel/adam, 700k iterations, 207 evaluations), the
# longest run of consecutive non-improving evaluations in the WHOLE run was 3,
# against a stop that needs 8. The rule was not slow, it was UNREACHABLE, and
# the run could only ever end on its iteration cap: 11.7 h, of which the last
# 5.2 h bought 1.6 % of held-out MSE.
#
# The threshold has to be RELATIVE because the metrics differ by orders of
# magnitude (SSIM in [0, 1], PSNR in tens of dB, MSE ~1e-5 here), which is
# exactly why the absolute `min_delta` below has always been left at 0. Same
# convention as torch's ReduceLROnPlateau `threshold_mode='rel'`.
DEFAULT_MIN_DELTA_REL = 0.0


def resolve_patience(total_iterations, eval_every, patience=None, *,
                     closed_loop: bool = False,
                     fraction: float = PATIENCE_FRACTION,
                     floor: int = DEFAULT_PATIENCE_EVALS,
                     closed_loop_floor: int = CLOSED_LOOP_PATIENCE_EVALS) -> int:
    """Patience in EVALUATIONS, scaled to the length of the run.

    ``patience`` explicitly set is always honoured — this only supplies the
    default. Returns at least ``floor`` so short runs are unaffected.

    ``closed_loop=True`` (a ``PlateauLRReducer`` is driving the LR) returns
    ``closed_loop_floor`` instead of the fraction, because waiting is then the
    REDUCER's job: a plateau buys an LR cut, and only a plateau at the LR floor
    ends the run. Scaling this patience up as well would just delay that final
    exit — and even the open-loop floor is more waiting than the closed loop
    needs, since reaching the LR floor already required six detected plateaus.
    """
    if patience is not None:
        return int(patience)
    if closed_loop:
        return int(closed_loop_floor)
    total = max(1, int(total_iterations))
    every = max(1, int(eval_every))
    n_evals = max(1, total // every)
    return max(int(floor), int(math.ceil(fraction * n_evals)))


def resolve_min_iter(total_iterations, min_iter=None, *,
                     closed_loop: bool = False,
                     fraction: float = MIN_PROGRESS_FRACTION) -> int:
    """Iteration before which no stopping rule may fire.

    ``min_iter`` explicitly set is honoured, including 0 to disable the guard.

    ``closed_loop=True`` returns 0: this floor is a PROXY for "do not stop
    while the step size is still large", and a ``PlateauLRReducer`` enforces
    that exactly, by gating the stop on the LR actually being at its floor. The
    proxy is only needed where there is no LR to close the loop on — the
    classical backends.
    """
    if min_iter is not None:
        return int(min_iter)
    if closed_loop:
        return 0
    return int(fraction * max(0, int(total_iterations)))


# ---------------------------------------------------------------------------
# Scoring a held-out projection
# ---------------------------------------------------------------------------

def resolve_holdout_index(index, n_angles: int) -> int:
    """`None` -> the middle angle. An int is bounds-checked.

    The middle is the default because it is the angle furthest (in either
    direction) from the ends of the scan, so its neighbours on both sides stay
    in the training set and the interpolation it tests is the easiest one the
    geometry offers. A held-out END angle would be an extrapolation and would
    read as a harder problem than the reconstruction actually faces.
    """
    if index is None:
        return int(n_angles) // 2
    idx = int(index)
    if not 0 <= idx < int(n_angles):
        raise ValueError(f"holdout_index={idx} out of range [0, {n_angles})")
    return idx


class HoldoutScorer:
    """Scores a predicted projection against a measured one, the same way for
    every backend.

    Holds the target and a FIXED `data_range` (taken from the target once), so
    SSIM and PSNR at iteration 10 and iteration 10 000 are on the same scale and
    the curve between them means something. Recomputing the range per checkpoint
    — from the prediction, or from the pair — would let the metric move because
    the normalisation moved.

    SSIM/PSNR come from `ct_core.projection_diag`, i.e. `ct_core.ssim`: the one
    implementation the structural LOSS also uses. A backend that brings its own
    SSIM produces a stopping decision nobody else can reproduce.
    """

    def __init__(self, target, *, data_range: float | None = None,
                 window: tuple | None = None, label: str = ""):
        """`window` is an optional (b0, b1, a0, a1) crop applied to BOTH sides.

        Pass the domain-covered detector window here: a ray that leaves the
        reconstruction domain integrates through matter the volume does not
        contain, so scoring it measures FOV truncation rather than the
        reconstruction, and it does so identically at every iteration — a
        constant handicap that flattens the very curve the stopper reads.
        """
        t = np.asarray(target, dtype=np.float32)
        if t.ndim != 2:
            raise ValueError(f"HoldoutScorer expects a 2-D projection, got {t.ndim}-D")
        if window is not None:
            b0, b1, a0, a1 = (int(v) for v in window)
            t = t[b0:b1, a0:a1]
        self.crop_window = window
        self.target = t
        self.label = label
        self.data_range = (float(t.max() - t.min()) if data_range is None
                           else float(data_range))
        self.degenerate = self.data_range <= 1e-12
        if self.degenerate:
            # A constant projection has no contrast, so SSIM and PSNR are not
            # measuring anything — but MSE still is, and refusing to run would
            # turn a degenerate slice of a synthetic into a crash. Fall back to
            # a unit range (what the TIGRE backend has always done) and say so,
            # so nobody reads the resulting SSIM as a number about the image.
            print("  WARNING: the evaluation projection is constant "
                  f"({float(t.min()):.6g} everywhere). SSIM and PSNR are "
                  f"computed on a unit data range and are not meaningful; "
                  f"prefer stop_metric='mse' here.")
            self.data_range = 1.0

        # SSIM's kernel must fit inside the image ('valid' convolution), which a
        # tiny projection — a small test scan, or an aggressive eval stride —
        # does not guarantee. Shrink to the largest odd size that fits, floored
        # at 3, matching `evaluate_projection`. Announced, because a different
        # kernel is a different SSIM and the curve should not be compared with
        # one measured at 11. NOTE this is the KERNEL size, distinct from
        # `crop_window` above (the detector region being scored).
        self.ssim_window = min(11, int(min(t.shape)))
        if self.ssim_window % 2 == 0:
            self.ssim_window -= 1
        self.ssim_window = max(3, self.ssim_window)
        if self.ssim_window != 11:
            print(f"  NOTE: evaluation projection is {t.shape[0]}x{t.shape[1]}, "
                  f"so SSIM uses a {self.ssim_window}x{self.ssim_window} kernel "
                  f"instead of 11x11 — comparable within this run, not across "
                  f"runs.")

    def crop(self, pred) -> np.ndarray:
        p = np.asarray(pred, dtype=np.float32)
        if self.crop_window is not None and p.shape != self.target.shape:
            b0, b1, a0, a1 = (int(v) for v in self.crop_window)
            p = p[b0:b1, a0:a1]
        return p

    def score(self, pred) -> dict:
        """-> {'ssim', 'psnr', 'mse'} on the fixed data range."""
        import torch

        from .projection_diag import psnr_2d, ssim_2d

        p = self.crop(pred)
        if p.shape != self.target.shape:
            raise ValueError(f"prediction {p.shape} vs target {self.target.shape}")
        pt = torch.from_numpy(np.ascontiguousarray(p))
        tt = torch.from_numpy(np.ascontiguousarray(self.target))
        mse = float(np.mean((p - self.target) ** 2))
        return {"ssim": float(ssim_2d(pt, tt, data_range=self.data_range,
                                      window_size=self.ssim_window)),
                "psnr": float(psnr_2d(pt, tt, data_range=self.data_range)),
                "mse": mse}


# ---------------------------------------------------------------------------
# Criterion 1: peak held-out metric with patience
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Peak-metric tracker with patience, best-state capture, and history."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0,
                 metric: str = "ssim", min_iter: int = 0,
                 min_delta_rel: float = DEFAULT_MIN_DELTA_REL):
        if metric not in _MODE:
            raise ValueError(
                f"early-stop metric must be one of {list(_MODE)}, got {metric!r}")
        if not (0.0 <= float(min_delta_rel) < 1.0):
            raise ValueError("min_delta_rel is a FRACTION of the running best "
                             f"and must be in [0, 1), got {min_delta_rel}")
        self.patience = int(patience)
        # Iterations before which `should_stop` stays False no matter how long
        # the plateau. Best-state capture is UNAFFECTED — an early peak is
        # still recorded and still restored, so this delays the exit, never the
        # answer. See MIN_PROGRESS_FRACTION.
        self.min_iter = int(min_iter)
        self._last_iter = -1
        # NOTE `min_delta` is in the METRIC'S OWN UNITS, and those units differ by
        # orders of magnitude: SSIM lives in [0, 1], PSNR in tens of dB, MSE in
        # whatever the line integrals square to (~1e-3 here). A min_delta that is
        # a sensible threshold for SSIM or PSNR is enormous next to an MSE, so
        # every checkpoint "fails to improve" and the run stops at the first one.
        # MEASURED on a semi-convergent trace: min_delta=0.01 stops `psnr` at the
        # right place (iter 15) and `mse` at iteration 1. `_check_min_delta` below
        # catches that on the first update rather than letting it look like fast
        # convergence.
        self.min_delta = float(min_delta)
        # ...which is why the useful threshold is the RELATIVE one: a fraction
        # of the running best, so one number means the same thing to every
        # metric and at every stage of a run. See DEFAULT_MIN_DELTA_REL.
        self.min_delta_rel = float(min_delta_rel)
        self.metric = metric
        self.mode = _MODE[metric]
        self._min_delta_checked = False
        self.reset()
        # History persists across resets (for a continuous convergence figure).
        self.history: dict[str, list] = {
            k: [] for k in ("iters", "ssim", "psnr", "mse",
                            "fdk_ssim", "fdk_psnr", "fdk_mse")
        }
        self.stop_iter: int | None = None

    def reset(self) -> None:
        """Clear the best/patience state (e.g. at a staged-training switch) so
        the next phase is judged on its own. History is preserved."""
        self.best = -float("inf") if self.mode == "max" else float("inf")
        self.best_iter: int | None = None
        self.best_state = None
        self.num_bad = 0

    def _check_min_delta(self, v: float) -> None:
        """Warn once if min_delta is large enough to make every step a failure.

        A threshold of more than a tenth of the metric's own magnitude means the
        run will stop after `patience` checkpoints almost regardless of what the
        reconstruction does — which reads as fast convergence, not as a
        misconfiguration.
        """
        self._min_delta_checked = True
        if self.min_delta <= 0 or not np.isfinite(v) or v == 0:
            return
        if self.min_delta > 0.1 * abs(v):
            print(f"  WARNING: early-stop min_delta={self.min_delta:g} is large "
                  f"next to the {self.metric} it is thresholding "
                  f"({v:.4g}). min_delta is in the metric's OWN units, so this "
                  f"will call almost every checkpoint 'no improvement' and stop "
                  f"after {self.patience} of them. For {self.metric}, try "
                  f"~{0.001 * abs(v):.2g}-{0.01 * abs(v):.2g} or 0.")

    def improvement_threshold(self) -> float:
        """How much better than the running best a value must be to count.

        The larger of the absolute and relative thresholds. Before the first
        update ``best`` is +-inf, where a relative threshold is meaningless (and
        would be inf), so the absolute one stands alone and the first finite
        value is always an improvement.
        """
        if not np.isfinite(self.best) or self.best == 0.0:
            return self.min_delta
        return max(self.min_delta, self.min_delta_rel * abs(self.best))

    def _is_best(self, v: float) -> bool:
        """Is this the best value seen? NO tolerance — the best iterate is the
        best iterate, and which one gets restored must not depend on a
        threshold whose job is to decide when to STOP."""
        if not np.isfinite(v):
            return False
        return v > self.best if self.mode == "max" else v < self.best

    def _is_better(self, v: float) -> bool:
        """Does this evaluation beat the best by enough to count as progress?

        Measured against ``self.best``, which is maintained WITHOUT the
        tolerance (see ``_is_best``). That separation is load-bearing: when the
        threshold also gated the best, a value that just missed left the best
        at an older, easier number, so the NEXT value cleared it more easily
        and reset the patience — making the stopping iteration a non-monotone,
        knife-edge function of the tolerance. MEASURED on run ny96yzab: moving
        the threshold by 0.2 % moved the stop from 306 852 to 441 732, and the
        STRICTER setting ran LONGER. With the best decoupled, raising the
        tolerance can only ever stop the run sooner.
        """
        if not np.isfinite(v):
            return False
        t = self.improvement_threshold()
        if self.mode == "max":
            return v > self.best + t
        return v < self.best - t

    def update(self, iteration: int, metrics: dict, snapshot_fn=None) -> bool:
        """Record one evaluation. Returns True if it was a new best.

        Returns True if this counted as PROGRESS (beat the best by more than
        the tolerance) — the flag the reducer and the patience counter run on.
        That is not the same question as "was this the best", which is asked
        without any tolerance so the restored iterate is always the true best.

        ``metrics`` must contain the criterion key (self.metric); ssim/psnr/mse
        and optional fdk_* are recorded for the figure. ``snapshot_fn`` (if
        given) is called only on a new best and its return value stored as the
        best state — so the cost of copying a volume is paid only when there is
        something better to keep.
        """
        self._last_iter = int(iteration)
        self.history["iters"].append(int(iteration))
        for k in ("ssim", "psnr", "mse", "fdk_ssim", "fdk_psnr", "fdk_mse"):
            self.history[k].append(metrics.get(k))

        v = metrics[self.metric]
        if not self._min_delta_checked:
            self._check_min_delta(v)
        # Two questions, deliberately answered separately: is this the best
        # iterate (keep it), and is this enough progress to count (keep going).
        # `improved` is judged against the best BEFORE this evaluation, so the
        # comparison never moves under the threshold's feet.
        improved = self._is_better(v)
        if self._is_best(v):
            self.best = v
            self.best_iter = int(iteration)
            if snapshot_fn is not None:
                self.best_state = snapshot_fn()
        if improved:
            self.num_bad = 0
        else:
            self.num_bad += 1
        return improved

    @property
    def should_stop(self) -> bool:
        if self._last_iter < self.min_iter:
            return False
        return self.num_bad >= self.patience

    def reason(self) -> str:
        return (f"held-out {self.metric} peaked at iteration {self.best_iter} "
                f"({self.best:.6g}); no improvement for {self.patience} "
                f"evaluations")


# ---------------------------------------------------------------------------
# Criterion 2: the L-curve corner
# ---------------------------------------------------------------------------

def menger_curvature(x, y):
    """Discrete curvature at each interior point, by circumscribed circle.

    kappa = 4 * area(P- , P, P+) / (|P- P| * |P P+| * |P- P+|)

    No derivatives, so it does not need the points evenly spaced — which they
    are not, since the iteration count is the parameter and the curve crowds up
    near convergence. Returns an array the length of `x` with NaN at both ends.

    SIGNED, from the cross product — so the sign says which way the curve turns
    but depends on the direction of traversal. Callers that care about the shape
    rather than the direction should use the magnitude and test the shape
    separately; see `LCurve.corner`, where "is this a corner" is decided by the
    residual/solution trade-off, not by this sign.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = np.full(x.size, np.nan)
    for i in range(1, x.size - 1):
        p0 = np.array([x[i - 1], y[i - 1]])
        p1 = np.array([x[i], y[i]])
        p2 = np.array([x[i + 1], y[i + 1]])
        d01, d12, d02 = (np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1),
                         np.linalg.norm(p2 - p0))
        if min(d01, d12, d02) < 1e-12:
            continue
        cross = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                 - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        k[i] = 2.0 * cross / (d01 * d12 * d02)
    return k


class LCurve:
    """Semi-convergence detection from the residual/solution-norm trade-off.

    Records, per checkpoint, the residual norm rho = ||A x - b|| and a solution
    norm eta = ||x|| (or a seminorm ||L x||). In log-log these trace an L: a
    steep drop while the reconstruction is still fitting SIGNAL, then a flat run
    where the residual barely improves but the solution norm climbs, which is
    noise being fitted. The corner is the stopping point.

    NEEDS NO HELD-OUT DATA. That is the whole point of having it alongside the
    held-out rule: it is the criterion that still works when dropping an angle
    is not affordable, and it is computed from quantities the backend already
    has.

    SCALE. Curvature in log-log is not scale-invariant, and the two axes here
    typically span very different numbers of decades — so the axes are
    normalised to [0, 1] over the observed range before the curvature is taken.
    Without that the corner drifts towards whichever axis happens to span more.
    Hansen makes the same point; it is not a detail.

    WHAT `residual_kind` IS FOR. A volume backend can afford the exact residual
    over the whole sinogram (one forward projection). A learning-based backend
    usually scores a fixed subset of rays instead. A FIXED subset is a
    legitimate substitute — it is a deterministic linear functional of x, so the
    curve is smooth and its corner is meaningful — but it is not the same
    number, so which one was used is recorded and plotted rather than left to
    the reader to guess.
    """

    #: minimum points before a corner is claimed at all. Three points define
    #: exactly one curvature value, and calling that a corner is meaningless.
    MIN_POINTS = 5

    #: Minimum |curvature| on the range-normalised log-log curve. The curve is
    #: scaled into the unit square, so this says the corner's osculating circle
    #: must be no larger than twice the curve's own extent — a real corner is
    #: far tighter than that, and a straight line (a power-law residual, which
    #: is what a NON-semi-convergent run gives) sits at ~1e-12 and is refused.
    MIN_CURVATURE = 0.5

    #: Savitzky-Golay window (odd, order 2) applied to the normalised log-log
    #: coordinates before the curvature. Menger curvature is a three-point
    #: formula and is therefore very sensitive to point-to-point noise, which a
    #: real residual curve has. MEASURED on a kinked synthetic L (40 points,
    #: multiplicative noise on both axes, 200 trials, corner at index 17):
    #:
    #:     window   clean   0.5% noise      2% noise       5% noise
    #:        off      17    17 [17,17]   17 [15,34]    24 [15,32]
    #:          7      16    16 [16,16]   16 [14,16]    16 [14,29]
    #:          9      15    16 [15,16]   15 [15,16]    15 [14,16]
    #:
    #: (median and inter-quartile range of the located index). Smoothing costs
    #: about two checkpoints of bias on a hard KINK — a real corner is rounded
    #: and loses less — and buys robustness at the noise levels an actual run
    #: has. Two checkpoints early is also the safe direction to be wrong in: for
    #: a semi-convergent method, stopping early under-fits rather than fitting
    #: noise. 9 is the default; pass `smooth=0` for the raw curve.
    SMOOTH_WINDOW = 9

    def __init__(self, *, patience: int = 3, norm: str = "l2",
                 residual_kind: str = "full", smooth: int | None = None):
        if norm not in ("l2", "gradient"):
            raise ValueError(f"L-curve norm must be 'l2' or 'gradient', got {norm!r}")
        self.patience = int(patience)
        self.norm = norm
        self.residual_kind = str(residual_kind)
        self.smooth = int(self.SMOOTH_WINDOW if smooth is None else smooth)
        if self.smooth and self.smooth % 2 == 0:
            raise ValueError(f"L-curve smooth window must be odd, got {self.smooth}")
        self.iters: list[int] = []
        self.residual: list[float] = []
        self.solution: list[float] = []
        #: corner as estimated after each `add`, so `should_stop` can require it
        #: to have STOPPED MOVING. Without that the rule fires on the first few
        #: points, whose provisional corner is always "several points back".
        self._corner_history: list = []

    # -- recording ---------------------------------------------------------
    def add(self, iteration: int, residual_norm: float, solution_norm: float) -> None:
        self.iters.append(int(iteration))
        self.residual.append(float(residual_norm))
        self.solution.append(float(solution_norm))
        self._corner_history.append(self.corner()[0])

    def add_from(self, iteration: int, volume, residual_norm: float) -> None:
        """Convenience: take the solution norm off the volume itself."""
        self.add(iteration, residual_norm, solution_norm(volume, self.norm))

    # -- corner ------------------------------------------------------------
    def _smoothed(self, x, y):
        """Light Savitzky-Golay smoothing of the log-log coordinates."""
        win = self.smooth
        if not win or x.size < max(win, 5):
            return x, y
        win = min(win, x.size if x.size % 2 else x.size - 1)
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            # Moving average of the same width — blunter, same purpose.
            k = np.ones(win) / win
            pad = win // 2
            sm = lambda a: np.convolve(np.pad(a, pad, mode="edge"), k, "valid")
            return sm(x), sm(y)
        return savgol_filter(x, win, 2), savgol_filter(y, win, 2)

    def curvature(self):
        """(iters, kappa) on the range-normalised, lightly smoothed log-log curve."""
        if len(self.iters) < 3:
            return np.asarray(self.iters), np.full(len(self.iters), np.nan)
        rho = np.log(np.maximum(np.asarray(self.residual, dtype=np.float64), 1e-300))
        eta = np.log(np.maximum(np.asarray(self.solution, dtype=np.float64), 1e-300))
        # Range-normalise so curvature is not dominated by the wider axis.
        def unit(a):
            span = float(a.max() - a.min())
            return (a - a.min()) / span if span > 0 else np.zeros_like(a)
        x, y = self._smoothed(unit(rho), unit(eta))
        return np.asarray(self.iters), menger_curvature(x, y)

    def corner(self):
        """(iteration, index) of the corner, or (None, None) if there isn't one.

        A candidate must pass three tests, because "argmax of curvature" alone
        will happily return a corner for a curve that has none:

        1. **Not at an endpoint, and not inside the smoothing filter's edge
           region.** An endpoint maximum means the corner is at or beyond the
           data recorded so far — the curve has not turned yet as far as anyone
           can tell. And Savitzky-Golay's edge handling extrapolates a
           polynomial over the outermost `smooth // 2` points, which can invent
           curvature there: MEASURED on a smooth semi-convergent L whose true
           peak is at checkpoint 10, the filter produced a marginally larger
           spurious peak at checkpoint 5 (2.32 vs 2.31) and won the argmax.
        2. **Curvature above `MIN_CURVATURE`.** A pure power-law residual is a
           straight line in log-log; its numerical curvature is ~1e-12 and its
           argmax is pure rounding noise. Refusing it is what makes "no corner
           yet" a possible answer.
        3. **It is a TRADE-OFF point** — the residual is falling and the
           solution norm is rising across it. That is the definition of the L
           corner, and unlike the sign of the curvature it does not depend on
           which way the curve is traversed.
        """
        if len(self.iters) < self.MIN_POINTS:
            return None, None
        _, k = self.curvature()
        mag = np.abs(k)
        rho = np.asarray(self.residual, dtype=np.float64)
        eta = np.asarray(self.solution, dtype=np.float64)

        # The filter's edge handling touches indices [0, edge) and
        # [size-edge, size). A three-point curvature at index i reads i-1..i+1,
        # so it is contaminated for i <= edge and i >= size-edge-1.
        edge = (self.smooth // 2) if (self.smooth and mag.size >= self.smooth) else 0
        lo, hi = max(1, edge + 1), min(mag.size - 1, mag.size - edge - 1)
        if lo >= hi:
            return None, None

        best_i, best_k = None, 0.0
        for i in range(lo, hi):
            if not np.isfinite(mag[i]) or mag[i] < self.MIN_CURVATURE:
                continue
            trade_off = (rho[i + 1] <= rho[i - 1]) and (eta[i + 1] >= eta[i - 1])
            if trade_off and mag[i] > best_k:
                best_i, best_k = i, float(mag[i])
        if best_i is None:
            return None, None
        return int(self.iters[best_i]), best_i

    @property
    def should_stop(self) -> bool:
        """True once the corner has STOPPED MOVING and is `patience` behind us.

        Both halves are needed. The curvature at the newest point cannot be
        computed at all (it needs a successor), so some lag is unavoidable — but
        lag alone is not enough: early in a run the provisional corner is always
        a few points back, so a pure lag test fires almost immediately. Requiring
        the estimate to be unchanged over the last `patience` checkpoints is what
        distinguishes a corner from a curve that is still bending.

        There is also a floor on how few checkpoints can end a run. Until the
        smoothing window is full the curvature is being taken on a shorter, more
        weakly smoothed curve, so its corner is not the one a few checkpoints
        later would give — and a provisional corner CAN look stable for a while.
        MEASURED on a smooth semi-convergent L: without this floor the rule fired
        at checkpoint 8 on a provisional corner that later moved to 10.
        """
        it, i = self.corner()
        if i is None:
            return False
        if len(self.iters) < max(self.MIN_POINTS, self.smooth) + self.patience:
            return False
        if (len(self.iters) - 1 - i) < self.patience:
            return False
        recent = self._corner_history[-self.patience:]
        return len(recent) == self.patience and all(c == it for c in recent)

    def reason(self) -> str:
        it, _ = self.corner()
        return (f"L-curve corner at iteration {it} "
                f"({self.residual_kind} residual, {self.norm} solution norm); "
                f"{self.patience} checkpoints past it")

    def as_dict(self) -> dict:
        it, i = self.corner()
        _, k = self.curvature()
        return {"iters": list(self.iters), "residual": list(self.residual),
                "solution": list(self.solution),
                "curvature": [None if not np.isfinite(v) else float(v) for v in k],
                "corner_iter": it, "norm": self.norm,
                "residual_kind": self.residual_kind}


def solution_norm(volume, norm: str = "l2") -> float:
    """||x||_2, or the gradient seminorm ||grad x||_2.

    The gradient seminorm is the more sensitive of the two for CT: noise
    amplification is high-frequency, and the L2 norm of a mouse volume is
    dominated by the soft-tissue bulk that barely changes between iterations, so
    its corner is shallow. `l2` is kept as the default because it is the
    classical choice and needs no assumption about what the noise looks like.
    """
    v = np.asarray(volume, dtype=np.float64)
    if norm == "l2":
        return float(np.sqrt(np.sum(v * v)))
    if norm == "gradient":
        g = 0.0
        for ax in range(v.ndim):
            d = np.diff(v, axis=ax)
            g += float(np.sum(d * d))
        return float(np.sqrt(g))
    raise ValueError(f"unknown norm {norm!r}")


# ---------------------------------------------------------------------------
# Criterion 3: metric-driven LR decay (learning-based backends)
# ---------------------------------------------------------------------------

class PlateauLRReducer:
    """Metric-driven LR decay that shares the held-out signal with EarlyStopper.

    This is the *closed-loop* half of the LR/early-stop coupling. An open-loop
    cosine schedule assumes a fixed horizon (``num_iterations``) that the run
    never reaches — held-out early stopping fires far earlier, while the LR is
    still ~peak, so the annealing phase never happens. This reducer instead
    drives the LR down only when the SAME held-out metric the stopper watches
    stops improving, so the two mechanisms never disagree about "converged".

    Mechanics (a plateau scheduler operating on the held-out metric):
      * Consumes the ``improved`` flag returned by ``EarlyStopper.update`` — no
        second notion of "improvement", so the signals are shared by
        construction.
      * After ``patience`` consecutive non-improving evaluations it multiplies
        every param group's LR by ``factor``, clamped to a per-group floor of
        ``min_lr_fraction`` x the post-warmup base LR, then waits ``cooldown``
        evals before counting again.
      * ``activate`` is called once, when warmup ends and this reducer takes
        over LR control from the (frozen) warmup scheduler; it captures each
        group's base LR as the reference for the factor and the floor.

    Coupling to the stopper (done by the training loop):
      * On each reduction the loop resets ``EarlyStopper.num_bad`` to 0, so the
        stopper's patience counts only *after* the LR can no longer drop.
      * The loop gates the actual stop on ``at_floor`` — training stops only
        once the LR is at its floor and the metric still will not improve.
    """

    def __init__(self, factor: float = 0.5, patience: int = 2,
                 min_lr_fraction: float = 0.02, cooldown: int = 1):
        if not (0.0 < factor < 1.0):
            raise ValueError(f"lr_plateau.factor must be in (0, 1), got {factor}")
        if not (0.0 < min_lr_fraction <= 1.0):
            raise ValueError(
                f"lr_plateau.min_lr_fraction must be in (0, 1], got {min_lr_fraction}")
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr_fraction = float(min_lr_fraction)
        self.cooldown = int(cooldown)
        self.active = False
        self.num_bad = 0
        self.cooldown_left = 0
        self.num_reductions = 0
        self.base_lrs: list[float] | None = None
        self.min_lrs: list[float] | None = None

    def activate(self, optimizer) -> None:
        """Take over LR control. Capture each param group's current LR (the
        post-warmup base) as the reference for the decay factor and floor."""
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.min_lrs = [lr * self.min_lr_fraction for lr in self.base_lrs]
        self.num_bad = 0
        self.cooldown_left = 0
        self.active = True

    def at_floor(self, optimizer) -> bool:
        """True once every param group has decayed to its LR floor."""
        if not self.active or self.min_lrs is None:
            return False
        return all(float(g["lr"]) <= mn * (1.0 + 1e-9)
                   for g, mn in zip(optimizer.param_groups, self.min_lrs))

    def current_lrs(self, optimizer) -> list[float]:
        return [float(g["lr"]) for g in optimizer.param_groups]

    def step(self, improved: bool, optimizer) -> bool:
        """Consume one held-out evaluation. ``improved`` is the flag from
        ``EarlyStopper.update``. Returns True iff the LR was reduced this call."""
        if not self.active:
            return False
        if improved:
            self.num_bad = 0
            return False
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return False
        self.num_bad += 1
        if self.num_bad < self.patience or self.at_floor(optimizer):
            return False
        for g, mn in zip(optimizer.param_groups, self.min_lrs):
            g["lr"] = max(float(g["lr"]) * self.factor, mn)
        self.num_bad = 0
        self.cooldown_left = self.cooldown
        self.num_reductions += 1
        return True


# ---------------------------------------------------------------------------
# Combining them
# ---------------------------------------------------------------------------

class StoppingRules:
    """The rules a run has switched on, asked once per checkpoint.

    Both criteria are recorded whenever they are enabled, even if only one is
    allowed to stop the run — a curve you did not steer by is still evidence,
    and it is free once the numbers are in hand. `stop_on` names which rules may
    actually terminate; the others are diagnostic.
    """

    def __init__(self, stopper: EarlyStopper | None = None,
                 lcurve: LCurve | None = None,
                 stop_on: tuple = ("holdout",)):
        unknown = set(stop_on) - {"holdout", "lcurve"}
        if unknown:
            raise ValueError(f"unknown stopping rule(s) {sorted(unknown)} "
                             f"(expected 'holdout' and/or 'lcurve')")
        if "holdout" in stop_on and stopper is None:
            raise ValueError("stop_on includes 'holdout' but no EarlyStopper given")
        if "lcurve" in stop_on and lcurve is None:
            raise ValueError("stop_on includes 'lcurve' but no LCurve given")
        self.stopper = stopper
        self.lcurve = lcurve
        self.stop_on = tuple(stop_on)
        self.fired: str | None = None

    def should_stop(self) -> bool:
        """First rule that fires wins, and is remembered in `self.fired`."""
        if "holdout" in self.stop_on and self.stopper.should_stop:
            self.fired = "holdout"
            return True
        if "lcurve" in self.stop_on and self.lcurve.should_stop:
            self.fired = "lcurve"
            return True
        return False

    def reason(self) -> str:
        if self.fired == "holdout":
            return self.stopper.reason()
        if self.fired == "lcurve":
            return self.lcurve.reason()
        return "max iterations"

    def best_iter(self):
        """Which iteration to keep. The held-out rule knows its peak; the
        L-curve's answer is its corner."""
        if self.fired == "lcurve":
            it, _ = self.lcurve.corner()
            return it
        return None if self.stopper is None else self.stopper.best_iter


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def metrics_dict(stopper: EarlyStopper | None, lcurve: LCurve | None,
                 *, holdout_index=None, holdout_deg=None,
                 stop_iter=None, fired=None, delivered_iter=None) -> dict:
    """Everything a run learned about its own convergence, JSON-ready.

    Replaces TIGRE's `crossval_metrics` and the voxel trainer's
    `crossval_history` with one shape, so `log_convergence` and the figure do
    not need to know which backend produced it.

    ``best_iter`` is **the iteration the DELIVERED volume comes from** — not
    necessarily the held-out peak, because an L-curve stop delivers the corner
    instead. That distinction is not cosmetic: `data_budget.classical_budget`
    reads `best_iter` to report how much measured data the saved reconstruction
    actually saw, so reporting the held-out peak after an L-curve stop would
    credit the volume with iterations it does not contain. The held-out peak is
    still recorded, under ``holdout_peak_iter``.
    """
    out: dict = {"holdout_index": holdout_index, "holdout_deg": holdout_deg,
                 "stop_iter": stop_iter, "stopped_by": fired}
    if stopper is not None:
        h = stopper.history
        peak = stopper.best_iter
        out.update({
            # "iters", not "iterations": `wandb_logging.log_convergence` keys
            # its per-step series off this name, and a mismatch makes the
            # convergence figure silently empty rather than wrong-looking.
            "iters": list(h["iters"]),
            "ssim": list(h["ssim"]), "psnr": list(h["psnr"]), "mse": list(h["mse"]),
            "metric": stopper.metric,
            "holdout_peak_iter": peak,
            "best_iter": peak if delivered_iter is None else int(delivered_iter),
            "best_ssim": _at(h, "ssim", peak),
            "best_psnr": _at(h, "psnr", peak),
            "best_mse": _at(h, "mse", peak),
        })
    elif delivered_iter is not None:
        out["best_iter"] = int(delivered_iter)
    if lcurve is not None:
        out["lcurve"] = lcurve.as_dict()
    return out


def _at(history: dict, key: str, iteration):
    if iteration is None or not history["iters"]:
        return None
    try:
        i = history["iters"].index(int(iteration))
    except ValueError:
        return None
    return history[key][i]


def write_metrics(path, metrics: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
    return p


def plot_convergence(history: dict, best_iter, stop_iter, eval_every: float,
                     save_prefix: str | Path, title: str | None = None,
                     lcurve: LCurve | None = None):
    """Write the convergence figure to ``{save_prefix}_convergence.png`` / ``.pdf``.

    Three panels of the held-out metrics (SSIM / PSNR / log-MSE) with the peak
    marked and the patience window shaded, plus a FOURTH L-curve panel when an
    L-curve was recorded — on that panel the point the held-out rule chose and
    the corner the L-curve chose are both marked, so when the two criteria
    disagree the figure says so instead of hiding it.

    Returns the PNG path (or None if matplotlib is unavailable / no data).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping convergence figure.")
        return None

    iters = np.asarray(history["iters"], dtype=float)
    if iters.size == 0:
        return None
    ssim = np.asarray(history["ssim"], dtype=float)
    psnr = np.asarray(history["psnr"], dtype=float)
    mse = np.asarray(history["mse"], dtype=float)

    def _fdk(key):
        vals = history.get(key) or []
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        return arr if arr.size == iters.size and np.isfinite(arr).any() else None
    f_ssim, f_psnr, f_mse = _fdk("fdk_ssim"), _fdk("fdk_psnr"), _fdk("fdk_mse")

    best_iter = int(best_iter) if best_iter is not None else int(iters[-1])
    stop_iter = int(stop_iter) if stop_iter is not None else int(iters[-1])

    rc = {
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.framealpha": 0.85,
        "axes.linewidth": 0.75, "lines.linewidth": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    blue, red, green, grey, fdkc = "#2166ac", "#d6604d", "#1a9641", "#555555", "#999999"
    purple = "#762a83"

    n_panels = 4 if (lcurve is not None and len(lcurve.iters) >= 3) else 3
    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_panels, 1, figsize=(3.6, 2.07 * n_panels))
        metric_axes = axes[:3]
        for ax in metric_axes[:-1]:
            ax.tick_params(labelbottom=False)
        if stop_iter > best_iter:
            for ax in metric_axes:
                ax.axvspan(best_iter, stop_iter, color="#ffcccc", alpha=0.45,
                           zorder=0, linewidth=0)
        for ax in metric_axes:
            ax.axvline(best_iter, color=grey, linestyle="--", linewidth=0.9, zorder=2)

        best_idx = int(np.argmin(np.abs(iters - best_iter)))

        # Panel 1: SSIM
        ax = metric_axes[0]
        ax.plot(iters, ssim, color=blue, zorder=3, label="reconstruction")
        if f_ssim is not None:
            ax.plot(iters, f_ssim, color=fdkc, linestyle=":", linewidth=1.2,
                    zorder=2, label="FDK (same angle)")
        ax.scatter([iters[best_idx]], [ssim[best_idx]], color=blue, s=70,
                   zorder=4, marker="*", linewidths=0)
        ax.set_ylabel("Holdout SSIM")
        handles = [
            plt.Line2D([0], [0], color=grey, linestyle="--", linewidth=0.9,
                       label=f"Peak iter {best_iter}"),
            plt.Line2D([0], [0], color="#ffcccc", linewidth=6,
                       solid_capstyle="butt", label="Patience window"),
        ]
        if f_ssim is not None:
            handles.append(plt.Line2D([0], [0], color=fdkc, linestyle=":",
                                      linewidth=1.2, label="FDK (same angle)"))
        ax.legend(handles=handles, loc="lower right", handlelength=1.4)

        # Panel 2: PSNR
        ax = metric_axes[1]
        ax.plot(iters, psnr, color=red, zorder=3)
        if f_psnr is not None:
            ax.plot(iters, f_psnr, color=fdkc, linestyle=":", linewidth=1.2, zorder=2)
        ax.scatter([iters[best_idx]], [psnr[best_idx]], color=red, s=70,
                   zorder=4, marker="*", linewidths=0)
        ax.set_ylabel("Holdout PSNR (dB)")

        # Panel 3: MSE (log)
        ax = metric_axes[2]
        ax.semilogy(iters, mse, color=green, zorder=3)
        if f_mse is not None:
            ax.semilogy(iters, f_mse, color=fdkc, linestyle=":", linewidth=1.2, zorder=2)
        ax.scatter([iters[best_idx]], [mse[best_idx]], color=green, s=70,
                   zorder=4, marker="*", linewidths=0)
        ax.set_ylabel("Holdout MSE")
        ax.set_xlabel("Iteration")
        ax.grid(True, which="minor", color="#e0e0e0", linewidth=0.4, zorder=0)

        for ax in metric_axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, which="major", color="#dddddd", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
        pad = max(eval_every * 0.5, 1.0)
        metric_axes[0].set_xlim(iters[0] - pad, iters[-1] + pad)

        # Panel 4: the L-curve itself
        if n_panels == 4:
            ax = axes[3]
            rho = np.asarray(lcurve.residual, dtype=float)
            eta = np.asarray(lcurve.solution, dtype=float)
            li = np.asarray(lcurve.iters, dtype=float)
            ax.loglog(rho, eta, color=purple, marker="o", markersize=2.5,
                      linewidth=1.2, zorder=3)
            corner_it, ci = lcurve.corner()
            if ci is not None:
                ax.scatter([rho[ci]], [eta[ci]], s=80, marker="*", color=purple,
                           zorder=5, linewidths=0,
                           label=f"L-corner iter {corner_it}")
            j = int(np.argmin(np.abs(li - best_iter)))
            ax.scatter([rho[j]], [eta[j]], s=45, marker="D", facecolors="none",
                       edgecolors=blue, linewidths=1.1, zorder=4,
                       label=f"Holdout peak iter {best_iter}")
            ax.set_xlabel(f"Residual $\\|Ax-b\\|$  ({lcurve.residual_kind})")
            ax.set_ylabel(f"Solution norm ({lcurve.norm})")
            ax.legend(loc="best", handlelength=1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, which="major", color="#dddddd", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

        if title:
            fig.suptitle(title, fontsize=9, y=0.997)
        plt.tight_layout(pad=0.6, h_pad=0.4)

        prefix = Path(save_prefix)
        png = prefix.with_name(prefix.name + "_convergence.png")
        pdf = prefix.with_name(prefix.name + "_convergence.pdf")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        return png
