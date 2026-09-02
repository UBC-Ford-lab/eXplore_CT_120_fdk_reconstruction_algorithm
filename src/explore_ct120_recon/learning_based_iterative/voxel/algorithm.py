"""The voxel grid as a selectable ``--algorithm``.

Three things the driver cannot know about a representation, kept with the
representation: the flags it adds to the CLI, the constructor arguments it
takes, and how much of the machine it needs. See ``..registry`` for why each
of them is here rather than in ``run_learned_recon.py``.

This is also the file a second representation is copied from, so it is written
to be read that way: every entry says what makes it voxel-specific, not just
what it is.
"""

from __future__ import annotations

from ...ct_core.preflight import learned_footprint
from ..registry import Footprint, LearnedAlgorithm, MachineRequest
from .reconstructor import VoxelReconstructor

#: Near AIR, not water. Outside the object that is already the right answer, so
#: gradient pressure only has to raise mu where the projections demand it,
#: rather than walk every air voxel back down. Mirrored in
#: ``VoxelReconstructor.__init__``; kept here because this is what the CLI
#: advertises and what an unspecified run records.
DEFAULT_INIT_DENSITY = 0.001


def add_args(parser) -> None:
    """Flags that mean something only to a dense grid."""
    parser.add_argument(
        '--init-density', type=float, default=DEFAULT_INIT_DENSITY,
        metavar='MU',
        help='Starting attenuation of every voxel, mm^-1 (default: '
             '%(default)s). Voxel-grid only: here the parameter IS mu, so its '
             'initial value is a modelling choice — near air, not water, so '
             'the optimizer only has to raise mu where the projections demand '
             'it. A network sets its output scale in its head instead.')


def options(args) -> dict:
    """Constructor kwargs, and the run-config entries, from one list."""
    return {'init_density': float(args.init_density)}


def footprint(args, req: MachineRequest) -> Footprint:
    """VRAM/RAM for a dense grid — the shared learned shape, one number filled.

    THE VOXEL-SPECIFIC LINE is ``param_bytes=req.vol_bytes``: the parameters
    ARE the exported voxels, so the count is already in the request and
    ``args`` goes unread. That is exactly what a network cannot do — its
    weight count is a property of the architecture (table size, levels, MLP
    width), none of which the export grid knows.

    A representation whose footprint DOES read ``args`` must still answer when
    ``args`` is None, falling back to the same defaults its CLI advertises:
    ``estimate('<name>', ...)`` sizes a job from shapes alone, with no argparse
    namespace anywhere (see ``registry._StaticFootprint``).
    """
    del args                       # deliberately unread; see the docstring
    return learned_footprint(req, param_bytes=req.vol_bytes,
                             note="Voxel grid: one parameter per exported "
                                  "voxel (the grid IS the volume).")


ALGORITHM = LearnedAlgorithm(
    name='voxel',
    reconstructor=VoxelReconstructor,
    summary="dense voxel grid — SIRT's representation, trained with Adam",
    add_args=add_args,
    options=options,
    footprint=footprint,
)
