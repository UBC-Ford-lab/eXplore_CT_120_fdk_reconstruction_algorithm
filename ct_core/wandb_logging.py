"""Shared experiment logging for every reconstruction backend.

Two layers, both driven by the same figures:

  * LOCAL PLOTS — always on (``--no-plots`` disables). Every recon writes a
    small set of PNGs next to its output volume: orthogonal central slices,
    an HU histogram, a sinogram preview, and (when the backend produced one)
    a convergence curve.
  * WEIGHTS & BIASES — strictly opt-in (``--wandb``). The same figures are
    uploaded, plus native per-step charts (training loss / holdout metrics)
    for backends that log live.

PRIVACY — this is a public repository, so the code must not carry or leak
anything identifying:

  * No project, entity, API key, username, hostname, or path is hardcoded
    anywhere. The W&B project comes from ``--wandb-project`` or the
    ``WANDB_PROJECT`` env var; the entity from ``--wandb-entity`` /
    ``WANDB_ENTITY`` (else the account default). Auth is out-of-band
    (``wandb login`` or ``WANDB_API_KEY``) — never in code or configs.
  * The run config is a WHITELIST of geometry/algorithm numbers. The scan is
    identified by its folder BASENAME only — never an absolute path. The raw
    scan.xml header (which carries site/hardware metadata such as serial
    numbers) is never uploaded.
  * W&B's implicit capture channels are turned off: console capture (stdout
    echoes absolute local paths), code upload, and git metadata.
  * Logging is best-effort: any W&B failure prints a notice and the
    reconstruction continues.

Where the data goes: W&B runs land in the USER'S OWN project under their
account's privacy settings — nothing is published by this code. Keep the
target project private if the scans themselves are sensitive.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .projection_diag import (
    evaluate_projection,
    power_spectrum_figure,
    ssim_heatmap_figure,
)


# ---------------------------------------------------------------- CLI args --

def add_wandb_args(parser) -> None:
    """Logging flags shared by every driver (called from add_common_args)."""
    parser.add_argument(
        '--wandb', action='store_true', default=False,
        help='Log this reconstruction to Weights & Biases (default: off). '
             'Requires --wandb-project or the WANDB_PROJECT env var; auth via '
             '`wandb login` or WANDB_API_KEY. Runs go to your own account.')
    parser.add_argument(
        '--wandb-project', default=None,
        help='W&B project name (default: $WANDB_PROJECT). Never hardcoded — '
             'this is a public repo.')
    parser.add_argument(
        '--wandb-entity', default=None,
        help='W&B entity/team (default: $WANDB_ENTITY, else your account '
             'default).')
    parser.add_argument(
        '--wandb-run-name', default=None,
        help='Run name (default: <scan-folder-basename>_<algorithm>).')
    parser.add_argument(
        '--wandb-mode', default='online', choices=('online', 'offline'),
        help='W&B mode (default: online). Use offline on air-gapped compute '
             'nodes and `wandb sync` later.')
    parser.add_argument(
        '--no-plots', action='store_true', default=False,
        help='Disable the local PNG plots written next to the output volume.')


# ---------------------------------------------------------------- figures ---

HU_WINDOW = (-1000.0, 2000.0)


def midplane_views(volume, geometry: dict):
    """The three central orthogonal slices of an (Nx, Ny, Nz) volume.

    Returns [(name, slice_2d, xlabel, ylabel, extent_mm), ...] for
    axial / coronal / sagittal — the single source both the triptych figure
    and the per-view W&B images draw from.
    """
    Nx, Ny, Nz = volume.shape
    dx = float(geometry.get("dx", 1.0))
    dz = float(geometry.get("dz", dx))
    ox, oy, oz = (float(v) for v in geometry.get("vol_origin", (0.0, 0.0, 0.0)))
    ex = (ox - Nx * dx / 2, ox + Nx * dx / 2)
    ey = (oy - Ny * dx / 2, oy + Ny * dx / 2)
    ez = (oz - Nz * dz / 2, oz + Nz * dz / 2)
    return [
        ("axial", volume[:, :, Nz // 2].T, "x [mm]", "y [mm]", ex + ey),
        ("coronal", volume[:, Ny // 2, :].T, "x [mm]", "z [mm]", ex + ez),
        ("sagittal", volume[Nx // 2, :, :].T, "y [mm]", "z [mm]", ey + ez),
    ]


def single_view_figure(name, sl, xl, yl, extent, hu_window=HU_WINDOW) -> Figure:
    """One midplane view as its own figure (per-view W&B panels).

    No colorbar — the window is fixed ([-1000, 2000] HU) and stated in the
    title, so the bar only cost image area.
    """
    fig = Figure(figsize=(6, 5.5), dpi=110)
    ax = fig.add_subplot(111)
    ax.imshow(sl, cmap="gray", origin="lower", extent=extent,
              vmin=hu_window[0], vmax=hu_window[1])
    ax.set_title(f"{name} (midplane, {hu_window[0]:.0f}..{hu_window[1]:.0f} HU)",
                 fontsize=10)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def hu_histogram_figure(volume) -> Figure:
    """Linear-count HU histogram with tissue landmarks.

    Two panels, both linear-y — one linear axis cannot span the air spike
    (~1e8), the tissue peak (~1e6) and the bone tail (~1e3) at once:

      * left: full range, y scaled to the NON-AIR maximum (tallest bin above
        -900 HU, x1.3 headroom); the air spike runs off the top with its true
        height annotated.
      * right: bone-tail zoom (>= +300 HU) on its own y-scale.
    """
    counts, edges = np.histogram(np.asarray(volume).ravel(), bins=256,
                                 range=(-1100, 3100))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = Figure(figsize=(10.5, 4), dpi=110)

    ax = fig.add_subplot(1, 2, 1)
    ax.stairs(counts, edges, fill=True, color="steelblue")
    non_air = counts[centers > -900.0]
    if non_air.size and non_air.max() > 0:
        y_top = 1.3 * float(non_air.max())
        if counts.max() > y_top:
            ax.set_ylim(0, y_top)
    for hu, label in ((-1000, "air"), (0, "water"), (1500, "bone")):
        ax.axvline(hu, color="crimson", lw=0.8, ls="--")
        ax.text(hu, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=7,
                color="crimson", rotation=90, va="top")
    ax.set_xlabel("HU")
    ax.set_ylabel("voxel count")
    ax.set_title("HU distribution")

    axb = fig.add_subplot(1, 2, 2)
    i0 = int(np.argmax(centers >= 300.0))
    axb.stairs(counts[i0:], edges[i0:], fill=True, color="steelblue")
    axb.axvline(1500, color="crimson", lw=0.8, ls="--")
    axb.text(1500, axb.get_ylim()[1] * 0.95, " bone", fontsize=7,
             color="crimson", rotation=90, va="top")
    axb.set_xlabel("HU")
    axb.set_title("bone tail (zoom, >= 300 HU)")

    fig.tight_layout()
    return fig


def sinogram_preview_figure(projections) -> Figure:
    """Central projection + central-row sinogram of the raw input stack."""
    proj = np.asarray(projections)
    n_ang, n_b, _ = proj.shape
    fig = Figure(figsize=(10, 4), dpi=110)

    ax = fig.add_subplot(1, 2, 1)
    p = proj[n_ang // 2].astype(np.float32)
    im = ax.imshow(p, cmap="gray", aspect="auto")
    ax.set_title(f"projection {n_ang // 2} / {n_ang}", fontsize=10)
    ax.set_xlabel("detector a")
    ax.set_ylabel("detector b")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(proj[:, n_b // 2, :].astype(np.float32),
                   cmap="gray", aspect="auto")
    ax.set_title("sinogram (central detector row)", fontsize=10)
    ax.set_xlabel("detector a")
    ax.set_ylabel("angle index")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def convergence_figure(series: dict[str, tuple[list, list]],
                       best_iter=None) -> Figure:
    """One subplot per metric series {name: (iters, values)}."""
    n = max(1, len(series))
    fig = Figure(figsize=(4.5 * n, 3.6), dpi=110)
    for k, (name, (iters, values)) in enumerate(sorted(series.items())):
        ax = fig.add_subplot(1, n, k + 1)
        ax.plot(iters, values, lw=1.2)
        if best_iter is not None:
            ax.axvline(best_iter, color="crimson", lw=0.8, ls="--",
                       label=f"best @ {best_iter}")
            ax.legend(fontsize=7)
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        if values and min(values) > 0 and max(values) / max(min(values), 1e-30) > 50:
            ax.set_yscale("log")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ run config ---

_GEOMETRY_KEYS = ("R_s", "R_d", "da", "db", "dx", "dz", "vol_shape",
                  "vol_origin", "central_pixel_a", "central_pixel_b",
                  "det_psi_rad", "sinogram_downsample")


def sanitized_config(ctx, args, algorithm: str, params: dict | None) -> dict:
    """WHITELISTED run config — geometry + recon numbers, no identifiers.

    Deliberately excluded: absolute paths, the raw scan.xml header (carries
    site/hardware metadata incl. serial numbers), usernames, hostnames.
    """
    cfg = {
        "algorithm": algorithm,
        "scan": Path(ctx.scan_folder).name,
        "n_angles": int(np.asarray(ctx.angles).shape[0]),
        "total_angle_deg": float(getattr(ctx, "total_angle", 0.0) or 0.0),
        "downsample": int(getattr(ctx, "downsample", 1) or 1),
        "bhc_coeffs": list(args.bhc_coeffs) if getattr(args, "bhc_coeffs", None) else None,
        "ring_correction": bool(getattr(args, "ring_correction", False)),
    }
    for key in _GEOMETRY_KEYS:
        if key in ctx.geometry:
            v = ctx.geometry[key]
            cfg[f"geometry/{key}"] = (list(v) if isinstance(v, (tuple, list))
                                      else float(v) if np.isscalar(v) else v)
    if params:
        cfg.update({f"{algorithm}/{k}": v for k, v in params.items()})
    return cfg


# ------------------------------------------------------------------ logger --

class ReconLogger:
    """Best-effort plotting + W&B logging; every method is safe to call.

    Construct once per reconstruction (after the output path is known), use
    ``log`` for live scalars, the ``log_*`` helpers for the standard figures,
    and ``finish`` at the end. When W&B is disabled or unavailable everything
    degrades to the local PNGs (or a silent no-op with ``--no-plots``).
    """

    def __init__(self, args, ctx, algorithm: str, output_path,
                 params: dict | None = None):
        self.plots_enabled = not getattr(args, "no_plots", False)
        self.plot_dir = Path(str(output_path)).with_suffix("")
        self.plot_dir = self.plot_dir.parent / (self.plot_dir.name + "_plots")
        self.run = None
        self._t0 = time.time()
        # Projection-diagnostics state (see log_projection_diag).
        self.noise_ceiling = None
        self._da = float(ctx.geometry.get("da", 1.0)) if ctx is not None else 1.0
        self._n_a = (int(ctx.projections.shape[2])
                     if getattr(ctx, "projections", None) is not None else 0)
        self._max_step = 0

        if not getattr(args, "wandb", False):
            return
        try:
            import wandb
        except ImportError:
            print("W&B requested but `wandb` is not installed "
                  "(pip install wandb). Continuing without it.")
            return
        project = args.wandb_project or os.environ.get("WANDB_PROJECT")
        if not project:
            print("W&B requested but no project set — pass --wandb-project "
                  "or set WANDB_PROJECT. Continuing without W&B.")
            return
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
        name = (args.wandb_run_name
                or f"{Path(ctx.scan_folder).name}_{algorithm}")
        try:
            # Console capture would upload stdout (which echoes absolute local
            # paths); code/git capture would upload repo state. All off.
            try:
                settings = wandb.Settings(console="off", disable_code=True,
                                          disable_git=True)
            except TypeError:  # older/newer wandb without these fields
                settings = None
            self.run = wandb.init(
                project=project, entity=entity, name=name,
                mode=getattr(args, "wandb_mode", "online"),
                config=sanitized_config(ctx, args, algorithm, params),
                settings=settings,
            )
            print(f"  W&B: logging to project '{project}' as run '{name}'")
        except Exception as e:  # never let logging kill a reconstruction
            print(f"W&B init failed ({type(e).__name__}: {e}). "
                  f"Continuing without W&B.")
            self.run = None

    # -- primitives ------------------------------------------------------

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Native per-step scalars (live charts). No-op without W&B."""
        if step is not None:
            self._max_step = max(self._max_step, int(step))
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            print(f"W&B log failed ({type(e).__name__}: {e})")

    def _emit(self, name: str, fig: Figure, step: int | None = None) -> None:
        """Save a figure locally and (if enabled) upload it.

        With ``step`` the W&B image joins a per-step sequence (slider in the
        UI); the local PNG is overwritten so it always holds the latest.
        """
        path = None
        if self.plots_enabled:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            path = self.plot_dir / f"{name}.png"
            fig.savefig(path)
            print(f"  Plot: {path}")
        if self.run is not None:
            try:
                import wandb
                image = wandb.Image(str(path) if path else fig)
                if step is not None:
                    self.log({f"plots/{name}": image}, step=step)
                else:
                    self.run.log({f"plots/{name}": image})
            except Exception as e:
                print(f"W&B image log failed ({type(e).__name__}: {e})")

    # -- standard figures ------------------------------------------------

    def log_sinogram_preview(self, projections) -> None:
        if self.plots_enabled or self.run is not None:
            self._emit("sinogram", sinogram_preview_figure(projections))

    def log_convergence(self, metrics, best_iter=None,
                        replay_steps: bool = True) -> None:
        """Convergence curves + native per-step series.

        Accepts either the TIGRE form ({'iters': [...], '<name>': [...], ...})
        or the voxel form ([{'iter': i, '<name>': v}, ...]). Set
        ``replay_steps=False`` when the backend already streamed these values
        live through ``log`` — the figure is still produced, but the per-step
        series is not logged twice.
        """
        series: dict[str, tuple[list, list]] = {}
        if isinstance(metrics, dict):
            iters = list(metrics.get("iters", []))
            for key, vals in metrics.items():
                if key == "iters" or not isinstance(vals, (list, tuple)):
                    continue
                if len(vals) == len(iters):
                    series[key] = (iters, list(vals))
            best_iter = best_iter if best_iter is not None else metrics.get("best_iter")
        else:  # list of {'iter': ..., name: value}
            for entry in metrics or []:
                it = entry.get("iter")
                for key, v in entry.items():
                    if key == "iter":
                        continue
                    series.setdefault(key, ([], []))
                    series[key][0].append(it)
                    series[key][1].append(v)
        if not series:
            return
        if self.plots_enabled or self.run is not None:
            self._emit("convergence", convergence_figure(series, best_iter))
        if self.run is not None and replay_steps:
            names = list(series)
            iters0 = series[names[0]][0]
            for j, it in enumerate(iters0):
                self.log({f"crossval/{n}": series[n][1][j] for n in names
                          if j < len(series[n][1])}, step=int(it))

    def log_volume_summary(self, volume_hu, ctx) -> None:
        """Three individual midplane views + HU histogram + summary scalars
        for any backend."""
        vol = np.asarray(volume_hu)
        if self.plots_enabled or self.run is not None:
            for name, sl, xl, yl, extent in midplane_views(vol, ctx.geometry):
                self._emit(f"view_{name}",
                           single_view_figure(name, sl, xl, yl, extent))
            self._emit("hu_histogram", hu_histogram_figure(vol))
        if self.run is not None:
            p = np.percentile(vol, (1, 50, 99, 99.9))
            self.run.summary.update({
                "volume/hu_p1": float(p[0]), "volume/hu_p50": float(p[1]),
                "volume/hu_p99": float(p[2]), "volume/hu_p99.9": float(p[3]),
                "volume/shape": list(vol.shape),
                "elapsed_min": (time.time() - self._t0) / 60.0,
            })

    # -- projection diagnostics (diag/*) --------------------------------

    def set_noise_ceiling(self, ceiling: dict | None) -> None:
        """Attach the measured noise ceiling (projection_diag.
        measure_noise_ceiling). Its ssim/psnr/mse constants are logged
        alongside every diag step so they overlay as reference lines, and its
        frame pair feeds the ceiling panels of the diagnostic figures."""
        self.noise_ceiling = ceiling
        if self.run is not None and ceiling is not None:
            try:
                self.run.summary.update({
                    "diag/noise_ceil_ssim": ceiling["ssim"],
                    "diag/noise_ceil_psnr": ceiling["psnr"],
                    "diag/noise_ceil_mse": ceiling["mse"],
                    "diag/noise_ceil_source": ceiling["source"],
                })
            except Exception as e:
                print(f"W&B noise-ceiling log failed ({type(e).__name__}: {e})")

    def log_projection_diag(self, pred, target, step: int | None = None,
                            figures: bool = True, scalars: bool = True,
                            verbose: bool = True) -> dict:
        """The per-projection diagnostic bundle every backend shares:
        diag/ssim, diag/psnr, diag/mse scalars (+ the noise-ceiling
        constants), and — when ``figures`` — the local-SSIM heatmap and the
        power-spectrum figure (both without any FDK baseline).

        Backends call this through the driver-provided ``diag_fn`` with
        their evaluation-projection prediction; drivers call it once more
        (or for the first time, for single-shot backends like FDK/ASTRA)
        with the final volume's forward projection. Returns the metrics.
        """
        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        m = evaluate_projection(pred, target)
        if verbose:
            tag = f"@{step}" if step is not None else "(final)"
            ceil = (f"  [ceiling ssim={self.noise_ceiling['ssim']:.4f}]"
                    if self.noise_ceiling else "")
            print(f"  diag{tag}  ssim={m['ssim']:.4f}  "
                  f"psnr={m['psnr']:6.2f} dB  mse={m['mse']:.6e}{ceil}")
        values = {"diag/ssim": m["ssim"], "diag/psnr": m["psnr"],
                  "diag/mse": m["mse"]}
        if self.noise_ceiling is not None:
            values.update({
                "diag/noise_ceil_ssim": self.noise_ceiling["ssim"],
                "diag/noise_ceil_psnr": self.noise_ceiling["psnr"],
                "diag/noise_ceil_mse": self.noise_ceiling["mse"],
            })
        if scalars:
            if step is not None:
                self.log(values, step=step)
            elif self.run is not None:
                try:
                    self.run.summary.update(values)
                except Exception as e:
                    print(f"W&B diag summary failed ({type(e).__name__}: {e})")

        if figures and (self.plots_enabled or self.run is not None):
            try:
                pair = (self.noise_ceiling or {}).get("pair")
                suffix = f" (iter {step})" if step is not None else " (final)"
                self._emit("diag_ssim_heatmap",
                           ssim_heatmap_figure(pred, target, noise_pair=pair,
                                               title=suffix), step=step)
                # Detector pixel pitch of THIS prediction: the geometry's
                # pitch times whatever stride the caller rendered at.
                factor = (max(1, int(round(self._n_a / pred.shape[1])))
                          if self._n_a else 1)
                self._emit("diag_power_spectrum",
                           power_spectrum_figure(pred, target,
                                                 det_px_mm=self._da * factor,
                                                 noise_pair=pair,
                                                 title=suffix),
                           step=step)
            except Exception as e:  # diagnostics must never kill a recon
                print(f"  diag figures failed ({type(e).__name__}: {e})")
        return m

    def log_recon_slices(self, volume_hu, hu_window=HU_WINDOW,
                         max_slices: int = 240) -> None:
        """The finished volume as a scrollable axial-slice sequence (W&B
        only — hundreds of local PNGs would just be clutter). Slices log at
        steps beyond the last training step so they never collide."""
        if self.run is None:
            return
        try:
            import wandb
            vol = np.asarray(volume_hu)
            Nz = vol.shape[2]
            stride = max(1, int(np.ceil(Nz / max_slices)))
            lo, hi = hu_window
            # W&B's internal step also advances on every un-stepped log call
            # (each figure upload), so start beyond BOTH counters — otherwise
            # the first slices land on already-passed steps and are dropped.
            wb_step = int(getattr(self.run, "step", 0) or 0)
            base = max(self._max_step, wb_step) + 1
            n_logged = 0
            for k in range(0, Nz, stride):
                sl = np.clip((vol[:, :, k].T - lo) / (hi - lo), 0.0, 1.0)
                img = (sl * 255.0).astype(np.uint8)[::-1]  # origin-lower
                self.run.log(
                    {"recon_slices": wandb.Image(
                        img, caption=f"z-slice {k}/{Nz}")},
                    step=base + k)
                n_logged += 1
            print(f"  W&B: logged {n_logged} axial slices (recon_slices"
                  f"{f', stride {stride}' if stride > 1 else ''})")
        except Exception as e:
            print(f"W&B recon_slices failed ({type(e).__name__}: {e})")

    def log_preflight(self, report) -> None:
        """Record the machine-fit report on the run (config + summary)."""
        if self.run is None:
            return
        try:
            self.run.config.update({
                "preflight/verdict": report.verdict,
                "preflight/gpu": report.gpu_name or "none",
                "preflight/vram_needed_gib": report.vram_needed / 2**30,
                "preflight/vram_free_gib": (report.vram_free or 0) / 2**30,
                "preflight/ram_needed_gib": report.ram_needed / 2**30,
            }, allow_val_change=True)
            self.run.summary["preflight/verdict"] = report.verdict
        except Exception as e:
            print(f"W&B preflight log failed ({type(e).__name__}: {e})")

    def abort(self, reason: str) -> None:
        """Mark the W&B run FAILED and close it (fatal preflight, etc.)."""
        if self.run is not None:
            try:
                self.run.summary["abort_reason"] = reason
                self.run.finish(exit_code=1)
            except Exception as e:
                print(f"W&B abort failed ({type(e).__name__}: {e})")
            self.run = None

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"W&B finish failed ({type(e).__name__}: {e})")
            self.run = None
