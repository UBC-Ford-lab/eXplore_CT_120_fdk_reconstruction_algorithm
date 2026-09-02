"""Classical iterative reconstruction — one subpackage per algorithm family.

* ``astra``  — ASTRA toolbox (SIRT3D_CUDA, CGLS3D_CUDA)
* ``tigre``  — TIGRE toolbox (OS-SART, SART, SIRT, MLEM)

Both honour the ct_core backend contract (see ``ct_core.pipeline``): consume
``ScanContext`` raw projections + angles + geometry, return a float32
(Nx, Ny, Nz) volume in HU.
"""
