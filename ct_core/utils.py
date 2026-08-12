"""Small shared system utilities for the reconstruction backends."""

from __future__ import annotations

import subprocess
from typing import Optional


def query_gpu_memory(gpu_index: int = 0) -> Optional[dict]:
    """Query GPU name and memory via nvidia-smi.

    Uses nvidia-smi directly (not torch) so backends that avoid initializing
    a CUDA context (~300-500 MB overhead) can still report memory. Returns
    ``{'name': str, 'total_bytes': int, 'free_bytes': int}`` or None when
    nvidia-smi is unavailable or times out.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader,nounits', f'--id={gpu_index}'],
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
