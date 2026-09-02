"""Where this package keeps things that belong to a SCANNER, not to a scan.

The detector calibrations (in-plane rotation psi, the per-pixel unwarp) are
measured once per detector/epoch and then reused by every pipeline and every
scan, so they cannot live next to a scan folder or next to an output. They go
in one shared directory, and every reader and writer has to agree on where
that is — muNeRF, the reconstruction drivers, and the standalone calibration
scripts alike.

Resolving it as "three levels up from this file" (what the code did before)
encodes one particular checkout layout: it is right when this package sits
inside the muNeRF repo as a submodule, and wrong for a standalone clone, where
it points at the PARENT of the clone — a directory the package does not own
and may not be able to write. The failure is silent, because a missing
calibration is a legitimate state (measure it and move on), so a standalone
user would re-measure psi on every run and scatter JSON outside their repo.

Resolution order, first hit wins:

  1. ``$CT_CALIBRATION_DIR`` — an explicit override, for a shared read-only
     calibration store or a scratch directory on a cluster.
  2. an existing ``data/calibration`` in any ancestor of this file — this is
     what keeps the submodule layout working exactly as before, and what lets
     a clone inherit calibrations that are already there.
  3. ``data/calibration`` under the enclosing project root, identified by a
     ``pyproject.toml``/``.git`` marker — the directory to CREATE when nothing
     exists yet. This is the editable-install / checkout case.
  4. the per-user data directory (``platformdirs.user_data_dir``), e.g.
     ``~/.local/share/explore_ct120_recon/calibration`` on Linux — the
     directory to create when the package was installed into
     ``site-packages``, where no checkout exists and the package must not
     write into the interpreter's own tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

ENV_VAR = "CT_CALIBRATION_DIR"
_MARKERS = ("pyproject.toml", ".git")
_INSTALL_DIRS = ("site-packages", "dist-packages")
APP_NAME = "explore_ct120_recon"


def is_installed_copy() -> bool:
    """True when this file lives inside an interpreter's package tree.

    An editable install (``pip install -e .``) keeps the sources in the
    checkout, so the markers below still find a project root; a regular
    install puts them under ``site-packages``, where the nearest
    ``pyproject.toml`` (if any) belongs to somebody else.
    """
    return any(p.name in _INSTALL_DIRS for p in Path(__file__).resolve().parents)


def project_root() -> Path:
    """The enclosing project directory, by marker file.

    Note ``.git`` is a FILE, not a directory, inside a submodule — hence
    ``exists()`` rather than ``is_dir()``. For an installed copy there is
    no project, and the per-user data directory stands in for it.
    """
    here = Path(__file__).resolve()
    if is_installed_copy():
        return Path(user_data_dir(APP_NAME))
    for parent in here.parents:
        if any((parent / m).exists() for m in _MARKERS):
            return parent
    return here.parents[1]


def calibration_dir() -> Path:
    """The shared scanner-calibration directory (see module docstring).

    Never creates anything — writers call ``mkdir(parents=True)`` themselves,
    so a read on a fresh checkout stays side-effect free.
    """
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "calibration"
        if candidate.is_dir():
            return candidate

    return project_root() / "data" / "calibration"
