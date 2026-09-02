"""How much measured data a reconstruction actually consumed.

Iteration counts do not compare: an OS-SART iteration sweeps the whole
sinogram, a voxel-trainer iteration touches one random ray batch, and FDK has
no iterations at all. The comparable quantity is how many times each
measurement (one detector pixel at one angle) was used on average:

    visits = rays used / measurements available

which is 1.00 for FDK, exactly N for N classical iterations, and
``iterations x rays_per_batch / measurements`` for the stochastic learned
backend — where it is routinely far below 1, i.e. the run never looked at
most of the data. Every driver reports this through :func:`data_budget` and
``ReconLogger.set_data_budget`` so the numbers in two logs mean the same
thing.

Coverage — the share of measurements used at least once — depends on HOW the
data is visited:

  sweep   deterministic full passes (FDK, ASTRA, TIGRE). Every measurement is
          used once per pass, so coverage is 1.0 for any complete pass.
  random  uniform sampling WITH replacement (the voxel trainer draws random
          (angle, row, col) indices). After k draws over N measurements each
          has been missed with probability (1-1/N)^k, so coverage is
          1 - e^(-visits) — 63% at one visit, not 100%.
"""

from __future__ import annotations

import math

SWEEP = "sweep"
RANDOM = "random"


def measurement_count(n_angles: int, n_b: int, n_a: int,
                      excluded_angles: int = 0) -> int:
    """Measurements available to the reconstruction.

    ``excluded_angles`` removes withheld projections (``--withhold-eval``)
    from the pool — data the algorithm was not allowed to use is not data it
    could have visited.
    """
    return int(max(0, int(n_angles) - int(excluded_angles))) * int(n_b) * int(n_a)


def data_budget(n_measurements: int, *, visits: float | None = None,
                rays_drawn: int | None = None,
                sampling: str = SWEEP) -> dict:
    """Visits + coverage for one reconstruction.

    Give either ``visits`` directly (classical: visits = passes = iterations)
    or ``rays_drawn`` (stochastic: total rays the optimizer consumed).
    """
    n = int(max(0, n_measurements))
    if visits is None:
        visits = (int(rays_drawn) / n) if (n and rays_drawn is not None) else 0.0
    visits = float(visits)
    if sampling == RANDOM:
        coverage = float(1.0 - math.exp(-visits))
    else:
        coverage = float(min(1.0, max(0.0, visits)))
    return {"measurements": n, "visits": visits, "coverage": coverage,
            "sampling": sampling}


def classical_budget(n_measurements: int, *, requested_iterations: int,
                     crossval_metrics: dict | None = None):
    """Budget for a sweep backend, honouring early-stop rollback.

    When cross-validation stops early the SAVED volume is the peak-SSIM one
    (``best_iter``), not the iteration the loop stopped at (``stop_iter``) —
    so the data budget OF THE DELIVERED RECONSTRUCTION is best_iter, and the
    larger number the run burned is reported separately rather than credited
    to the volume.

    Returns ``(budget, note, extra)`` ready for ReconLogger.set_data_budget.
    """
    cv = crossval_metrics or {}
    saved = int(cv.get("best_iter") or requested_iterations)
    stopped = int(cv.get("stop_iter") or requested_iterations)
    note = "each iteration uses every measurement exactly once"
    if stopped != saved:
        note += (f"; the run consumed {float(stopped):.2f} before early "
                 f"stopping rolled back to iteration {saved}")
    return (data_budget(n_measurements, visits=float(saved), sampling=SWEEP),
            note,
            {"data/iterations_run": stopped, "data/iterations_saved": saved})


def format_budget(budget: dict, *, note: str = "") -> str:
    """One-line summary, identical wording in every backend's log."""
    n = budget["measurements"]
    line = (f"Data budget: {budget['visits']:.2f} visits per measurement "
            f"({n / 1e6:.1f} M measurements, {100 * budget['coverage']:.1f}% "
            f"used at least once)")
    if note:
        line += f" — {note}"
    return line
