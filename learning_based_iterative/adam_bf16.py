"""Adam with bfloat16 moment buffers, for optimizers whose state IS a volume.

On the voxel backend every optimizer buffer is a full CT volume: at the
Scan_1988 100 um grid one fp32 copy is 1.80 GiB, so Adam's ``m`` and ``v``
cost 3.60 GiB on top of the parameters and their gradient. That is not a
rounding error on a 16 GiB card — it is the difference between a 19 456-ray
batch and a 40 960-ray one, and it is why Adam does not fit at all without
detector downsampling. Holding the two moments in bfloat16 gives one of those
volumes back (4x resident copies -> 3x) while the parameters, the gradient and
every arithmetic operation stay fp32.

Three things about this are not obvious, and each is load-bearing:

**bfloat16, not float16.** ``v`` accumulates ``g**2``. Gradients here run to
1e-6 and smaller on poorly-covered voxels, so ``g**2`` reaches 1e-12 — below
float16's smallest subnormal (6e-8) and gone. bfloat16 keeps float32's
8-bit exponent and spends the difference on mantissa, which is exactly the
trade this state wants: enormous range, coarse precision.

**Stochastic rounding is not optional.** bfloat16 keeps 8 significand bits, so
its half-ulp is ~0.39% of the value. The second moment updates by
``v <- v + (1-beta2)*(g**2 - v)``, and at the default beta2=0.999 that
increment is 0.1% of the gap — SMALLER than half an ulp whenever ``g**2`` is
within a factor of ~4 of ``v``. Under round-to-nearest the write is a no-op and
``v`` freezes at whatever coarse level it first reached, silently turning Adam
into something else. Stochastic rounding rounds up with probability equal to
the fractional position, so the increment is applied in expectation and the EMA
tracks correctly even though no individual step can represent it. ``m`` does
not have this problem (beta1=0.9 makes its increments ~10% of the gradient,
far above the ulp) but gets the same treatment for uniformity and to keep the
update unbiased.

**The step is chunked.** Upcasting a whole moment buffer to fp32 to do the
arithmetic would allocate the 1.80 GiB we just saved, at exactly the moment of
peak memory. The parameter is walked in flat slices instead, so the fp32
working set is a few times ``chunk_elems`` rather than a few times the volume.
The chunking is arithmetically invisible: with stochastic rounding off, any
chunk size gives bit-identical results.

The optimizer draws its own randomness from a private generator. Consuming the
global stream would make the ray sampler produce a different batch sequence,
so turning stochastic rounding on would change the reconstruction through a
second, hidden channel — the same trap the quadrature probe had to avoid.
"""

from __future__ import annotations

import math

import torch

__all__ = ["AdamBF16", "jit_kernel", "round_bf16"]

#: Elements per fp32 working slice. 2^24 elements = 64 MiB per fp32 temporary,
#: and the step needs a handful of them at once. Large enough that the kernel
#: launches are amortised (a 483 M-voxel grid is ~29 slices), small enough that
#: the working set is noise next to the buffers it is protecting.
DEFAULT_CHUNK_ELEMS = 1 << 24

#: float32 keeps 23 mantissa bits, bfloat16 keeps 7 — bfloat16 IS float32 with
#: the low 16 bits dropped. So truncation is a mask, and rounding is a mask
#: applied after adding something to those 16 bits.
_BF16_DROPPED_BITS = 16
_BF16_MASK = -(1 << _BF16_DROPPED_BITS)          # 0xFFFF0000 as a signed int32


def round_bf16(x: torch.Tensor, *, generator=None) -> torch.Tensor:
    """float32 -> bfloat16, stochastically when a generator is given.

    Adds a uniform random value over the 16 bits bfloat16 discards and then
    truncates, so a value sitting a fraction f of the way between two
    representable neighbours rounds up with probability f: unbiased, which is
    the entire point (see the module docstring on why ``v`` needs it).

    Without a generator this is plain truncation-free ``.to(bfloat16)``, i.e.
    round-to-nearest-even — deterministic, and what the tests compare against.

    Arithmetic on the bit pattern works for both signs: IEEE-754 is
    sign-magnitude, so incrementing the pattern always increases MAGNITUDE, and
    the rounding is therefore unbiased in magnitude with the sign untouched.
    """
    if generator is None:
        return x.to(torch.bfloat16)
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << _BF16_DROPPED_BITS, x.shape,
                          dtype=torch.int32, device=x.device,
                          generator=generator)
    # No saturation guard: a pattern large enough to carry out of int32 is
    # already inf or NaN, and the step has bigger problems than its rounding.
    rounded = torch.bitwise_and(bits + noise, _BF16_MASK)
    return rounded.view(torch.float32).to(torch.bfloat16)


# ---------------------------------------------------------------- CUDA JIT --
# The eager path above is correct but pays for it in bandwidth: it upcasts each
# moment slice to a full fp32 buffer, runs a chain of elementwise kernels over
# it, and converts back — ~112 bytes of traffic per parameter where torch's
# FUSED fp32 Adam moves ~28. MEASURED on a P100 at 34 M parameters: 11.8 ms
# against fused Adam's 1.9.
#
# The fix is one kernel that keeps the values in registers. torch.compile
# cannot build it here (Inductor needs Triton, i.e. compute capability >= 7.0,
# and this is Pascal), but `torch.cuda.jiterator` compiles an elementwise CUDA
# kernel at runtime through the nvrtc that PyTorch already ships — no CUDA
# toolkit, and it runs on any CUDA architecture.
#
# The one constraint that shapes the kernel: jiterator promotes MIXED input
# dtypes to fp32 and returns fp32 outputs, which would force the moments back
# through full-width memory. Feed it nothing but bfloat16 and the outputs come
# back bfloat16. So the kernel takes (m, v, g, u) all bfloat16 and returns
# (m', v', step) all bfloat16 — the parameter never enters it, and the caller
# applies `p += step` separately.
#
# MEASURED, 34 M parameters, P100: 3.4-3.6 ms/step depending on chunk size,
# against 11.8 eager and 1.9 for fused fp32 Adam. It does not reach parity —
# the g cast, the dither tensor and the write-back are traffic fused Adam does
# not pay — but it turns the bfloat16 option from "half the speed" into "about
# four fifths of it".

#: Elements per slice on the JIT path. Smaller than the eager path's, because
#: five bfloat16 temporaries are live per slice and the whole point is to keep
#: PEAK memory below what fp32 Adam holds. MEASURED: 2^22 gives 3.56 ms/step at
#: ~3.3 parameter-copies peak, 2^24 gives 3.39 ms at ~4.25 — i.e. 5% faster for
#: more memory than the optimizer it is trying to undercut. Not a good trade.
JIT_CHUNK_ELEMS = 1 << 22

_JIT_SOURCE = r"""
template <typename T>
void adam_bf16(T m, T v, T g, T u, T b1, T b2, T lrc, T rb2, T eps, T s,
               T& nm, T& nv, T& step) {
  float gf = static_cast<float>(g);
  float mf = static_cast<float>(b1) * static_cast<float>(m)
           + (1.f - static_cast<float>(b1)) * gf;
  float vf = static_cast<float>(b2) * static_cast<float>(v)
           + (1.f - static_cast<float>(b2)) * gf * gf;
  // Stochastic rounding, in registers. `u` carries ~8 bits of dither per
  // element (all it can, being bfloat16) and `s` is a fresh scalar per step;
  // adding them mod 1 slides the quantisation grid so the dither is continuous
  // ACROSS steps. Both halves are needed: with a fixed grid the second moment
  // decays measurably too slowly, and with no dither at all it does not decay.
  float uf = static_cast<float>(u) + static_cast<float>(s);
  uf -= floorf(uf);
  unsigned int uu = static_cast<unsigned int>(uf * 65536.f) & 0xFFFFu;
  unsigned int hv = (uu * 0x9e3779b9u) >> 16;      // decorrelate m from v
  nm = static_cast<T>(__uint_as_float(
        (__float_as_uint(mf) + uu) & 0xFFFF0000u));
  nv = static_cast<T>(__uint_as_float(
        (__float_as_uint(vf) + hv) & 0xFFFF0000u));
  float den = sqrtf(vf) * static_cast<float>(rb2) + static_cast<float>(eps);
  step = static_cast<T>(-static_cast<float>(lrc) * mf / den);
}
"""

_JIT_CACHE: list = []          # [] = not tried, [None] = unavailable, [fn] = ok


def jit_kernel():
    """The compiled kernel, or None if this build/device cannot provide one.

    Compiled once on first use and cached — including the failure, so a torch
    without `jiterator` costs one exception rather than one per step.
    `_create_multi_output_jit_fn` is underscore-prefixed and therefore not a
    stable API; when it moves or changes signature the eager path takes over
    and the only symptom is a slower run.
    """
    if _JIT_CACHE:
        return _JIT_CACHE[0]
    fn = None
    try:
        if torch.cuda.is_available():
            from torch.cuda import jiterator
            fn = jiterator._create_multi_output_jit_fn(
                _JIT_SOURCE, num_outputs=3,
                b1=0.9, b2=0.999, lrc=1e-4, rb2=1.0, eps=1e-8, s=0.0)
    except Exception:                              # pragma: no cover - env dep
        fn = None
    _JIT_CACHE.append(fn)
    return fn


class AdamBF16(torch.optim.Optimizer):
    """Adam whose exponential moving averages live in bfloat16.

    Drop-in for ``torch.optim.Adam`` with the same defaults and the same
    update; the only differences are where the state is stored and that the
    step walks the parameter in slices. Parameters and gradients are untouched
    fp32 — this trades the PRECISION of the moment estimates for their SIZE,
    and never the precision of the weights.

    ``stochastic_rounding=False`` is offered for A/B use only. It is the
    configuration in which ``v`` demonstrably stalls, so it is not a sensible
    production setting; ``tests/test_adam_bf16.py`` uses it to show the failure
    the default avoids.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, *, stochastic_rounding: bool = True,
                 chunk_elems: int = DEFAULT_CHUNK_ELEMS, seed: int = 1234567,
                 use_jit: bool = True):
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if int(chunk_elems) <= 0:
            raise ValueError(f"chunk_elems must be positive, got {chunk_elems}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps,
                                      weight_decay=weight_decay))
        self.stochastic_rounding = bool(stochastic_rounding)
        self.chunk_elems = int(chunk_elems)
        self.use_jit = bool(use_jit)
        self._seed = int(seed)
        # The per-step dither offset is drawn on the HOST. Drawing it on the
        # device would mean a .item() every step, and a sync per optimizer step
        # serialises the whole training loop for a single float.
        self._scalar_gen = torch.Generator().manual_seed(int(seed) + 104729)
        # One generator per device, made on first use: a CUDA generator cannot
        # be constructed before the device is known, and a CPU one cannot feed
        # a CUDA randint.
        self._generators: dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device):
        if not self.stochastic_rounding:
            return None
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._seed + len(self._generators))
            self._generators[device] = gen
        return gen


    @torch.no_grad()
    def _jit_step(self, kernel, p, state, *, beta1, beta2, lr, eps, wd,
                  bias1, bias2):
        """One step through the compiled kernel, sliced to bound peak memory.

        Sliced for MEMORY, not for correctness: five bfloat16 temporaries are
        live per slice (the cast gradient, the dither, and three outputs), and
        at full width those would push peak allocation above the fp32 Adam this
        is supposed to undercut — faster and fatter, which is the wrong trade.
        """
        dev = p.device
        gen = self._generator(dev)
        pf = p.view(-1)
        gf = p.grad.contiguous().view(-1)
        if wd:
            # Only allocation on this path that scales with the parameter, and
            # only when weight decay is actually in use (it is not, on the
            # voxel backend).
            gf = gf.add(pf, alpha=wd)
        mf = state["exp_avg"].view(-1)
        vf = state["exp_avg_sq"].view(-1)
        lrc = lr / bias1
        rb2 = 1.0 / math.sqrt(bias2)
        # One scalar per STEP, not per slice: it only has to decorrelate the
        # dither grid between steps.
        s = float(torch.rand((), generator=self._scalar_gen))
        n = pf.numel()
        for start in range(0, n, JIT_CHUNK_ELEMS):
            sl = slice(start, min(start + JIT_CHUNK_ELEMS, n))
            gb = gf[sl].to(torch.bfloat16)
            u = torch.rand(gb.shape, generator=gen, device=dev,
                           dtype=torch.bfloat16)
            nm, nv, step = kernel(mf[sl], vf[sl], gb, u, b1=beta1, b2=beta2,
                                  lrc=lrc, rb2=rb2, eps=eps, s=s)
            mf[sl] = nm
            vf[sl] = nv
            # The kernel returns the STEP rather than the new parameter: the
            # parameter is fp32 and passing it in would promote every output
            # to fp32, which is exactly the memory traffic being avoided.
            pf[sl].add_(step)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = float(group["lr"]), float(group["eps"])
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AdamBF16 does not support sparse gradients — the "
                        "dense voxel grid never produces them.")
                if p.dtype is not torch.float32:
                    # The update is computed in fp32 and written back in
                    # place, which silently changes meaning if the weights are
                    # half: refuse rather than guess. Every model in this
                    # project keeps fp32 parameters and casts only ACTIVATIONS
                    # under AMP, so this is a guard, not a limitation anyone
                    # currently meets.
                    raise RuntimeError(
                        f"AdamBF16 needs float32 parameters, got {p.dtype}. "
                        f"Only the moment buffers are bfloat16 — the weights "
                        f"are not, and a half-precision weight would lose the "
                        f"update instead of storing it.")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    # bfloat16 from the start: allocating fp32 and casting
                    # would put the peak this class exists to remove right at
                    # the first step.
                    zeros = dict(dtype=torch.bfloat16,
                                 memory_format=torch.preserve_format)
                    state["exp_avg"] = torch.zeros_like(p, **zeros)
                    state["exp_avg_sq"] = torch.zeros_like(p, **zeros)
                state["step"] += 1
                t = state["step"]
                # Python floats, not tensors: the bias corrections are scalars
                # and making them tensors would force a host sync per step.
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t

                kernel = (jit_kernel()
                          if (self.use_jit and self.stochastic_rounding
                              and p.is_cuda and p.is_contiguous()) else None)
                if kernel is not None:
                    self._jit_step(kernel, p, state, beta1=beta1, beta2=beta2,
                                   lr=lr, eps=eps, wd=wd, bias1=bias1,
                                   bias2=bias2)
                    continue

                gen = self._generator(p.device)
                pf = p.view(-1)
                gf = p.grad.contiguous().view(-1)
                mf = state["exp_avg"].view(-1)
                vf = state["exp_avg_sq"].view(-1)
                n = pf.numel()
                for start in range(0, n, self.chunk_elems):
                    sl = slice(start, min(start + self.chunk_elems, n))
                    g = gf[sl].float()
                    if wd:
                        g = g.add(pf[sl], alpha=wd)
                    m = mf[sl].float()
                    v = vf[sl].float()
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    # Bias correction applied to the DENOMINATOR and the step
                    # size rather than to the stored moments, exactly as
                    # torch.optim.Adam does — the buffers stay uncorrected so
                    # that resuming from a checkpoint is well defined.
                    denom = v.sqrt().div_(math.sqrt(bias2)).add_(eps)
                    pf[sl].addcdiv_(m, denom, value=-lr / bias1)
                    mf[sl] = round_bf16(m, generator=gen)
                    vf[sl] = round_bf16(v, generator=gen)
        return loss
