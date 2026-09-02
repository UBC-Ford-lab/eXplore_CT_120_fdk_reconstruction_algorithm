"""Failures a caller can act on, and the one place a driver turns them into
an exit code.

WHY THIS EXISTS. This package is two things at once: a set of command-line
drivers, and a library that muNeRF imports (``load_scan_data``,
``build_geometry``, ``preprocess_sinogram``, ...). Those two want opposite
things from a bad input. A driver wants a one-line message and a non-zero exit
status, with no traceback to read past. A library must not decide that the
process ends — muNeRF's dataset loader calls straight into ``load_scan_data``,
and a ``sys.exit`` there kills a training run with no exception to catch, no
way to try another scan folder, and nothing for a test to assert on beyond
``SystemExit``.

So the library raises and only the drivers exit:

    ct_core.*        raise ScanDataError / ConfigError / PreflightAbort
    run_*.py main()  let them propagate (still just raising, so it can be
                     called from a notebook or a test)
    __main__         cli_main(main) -> "Error: <message>" + exit(1)

All of them derive from ``ReconstructionError``, which reads as "the inputs or
the machine are wrong", never "the code is wrong" — a genuine bug should still
surface as an unhandled traceback rather than a tidy one-liner.
"""

from __future__ import annotations

import sys


class ReconstructionError(Exception):
    """A problem with the inputs or the machine, not a bug in the code."""


class ScanDataError(ReconstructionError):
    """The scan on disk is missing something, or contradicts itself."""


class ConfigError(ReconstructionError):
    """An argument, or a combination of them, that cannot be honoured."""


class PreflightAbort(ReconstructionError):
    """This machine cannot fit this job (see ct_core.preflight)."""


def cli_main(main_fn, *args, **kwargs) -> None:
    """Run a driver's ``main`` as a command-line program.

    Turns the errors above into ``Error: <message>`` on stderr and exit 1;
    everything else keeps its traceback, because an unexpected exception is
    a bug report. Ctrl-C exits 130 (the shell convention) rather than dumping
    a KeyboardInterrupt traceback out of a half-finished reconstruction.
    """
    try:
        main_fn(*args, **kwargs)
    except ReconstructionError as e:
        print(f"\nError: {e}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
