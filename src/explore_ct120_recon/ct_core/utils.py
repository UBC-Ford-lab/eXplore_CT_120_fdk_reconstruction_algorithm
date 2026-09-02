"""Small shared system utilities for the reconstruction backends."""

from __future__ import annotations

import os
import subprocess
from typing import Optional


def resolve_visible_gpu(gpu_index: int = 0) -> Optional[str]:
    """Map a logical CUDA device index to the id nvidia-smi expects.

    ``gpu_index`` is a *logical* index — the one torch uses, relative to
    ``CUDA_VISIBLE_DEVICES``. nvidia-smi's ``--id`` is a *physical* index, so
    on a shared node where the scheduler handed out a subset (e.g.
    ``CUDA_VISIBLE_DEVICES=2``), logical 0 is physical 2 and querying
    ``--id=0`` reports the wrong card. Under cgroup-isolated allocations (the
    usual SLURM case) only the allocated GPUs are visible at all and the
    mapping is the identity, so this is a no-op there.

    Returns the id string to pass to ``nvidia-smi --id=`` (an index, or a
    ``GPU-<uuid>`` / ``MIG-<uuid>`` string — nvidia-smi accepts both), or None
    when ``CUDA_VISIBLE_DEVICES`` is set but empty (no GPU is usable).
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(int(gpu_index))
    entries = [e.strip() for e in visible.split(',') if e.strip()]
    if not entries:
        return None                       # explicitly no visible devices
    if not 0 <= int(gpu_index) < len(entries):
        return str(int(gpu_index))        # out of range — let nvidia-smi fail
    return entries[int(gpu_index)]


def query_gpu_memory(gpu_index: int = 0) -> Optional[dict]:
    """Query GPU name and memory via nvidia-smi.

    Uses nvidia-smi directly (not torch) so backends that avoid initializing
    a CUDA context (~300-500 MB overhead) can still report memory. Returns
    ``{'name': str, 'total_bytes': int, 'free_bytes': int}`` or None when
    nvidia-smi is unavailable, times out, or no device is visible.

    ``gpu_index`` is interpreted the way torch interprets it (relative to
    CUDA_VISIBLE_DEVICES) — see :func:`resolve_visible_gpu`.
    """
    device_id = resolve_visible_gpu(gpu_index)
    if device_id is None:
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader,nounits', f'--id={device_id}'],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split(', ')
    return {
        'name': parts[0],
        'total_bytes': int(parts[1]) * 2**20,  # MiB -> bytes
        'free_bytes': int(parts[2]) * 2**20,
    }
