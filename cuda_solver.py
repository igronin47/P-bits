"""
CUDA-accelerated p-bit solver for p-circuits.

Drop-in replacement for ``p_kit.solver.csd_solver.CaSuDaSolver`` from the
IBM/p-kit framework.  Extends that framework in three ways that are relevant
to the accompanying paper:

1. GPU-parallel updates.  Every p-bit is updated by a dedicated CUDA thread,
   so one kernel launch advances the whole circuit by one cycle.  On the
   Jetson Orin Nano (1024 Ampere CUDA cores) this exposes two-orders-of
   magnitude headroom over the single-threaded Python loop in the original
   IBM code.
2. Three simulated-annealing variants in one kernel:
     * ``sequential`` - classical Gibbs sampling (one p-bit per step).
       Guaranteed to converge to the exact Boltzmann distribution; kept for
       correctness testing and for compatibility with small IBM/p-kit gates.
     * ``psa``        - parallel p-bit simulated annealing (Camsari/Datta).
     * ``tapsa``      - time-averaged pSA (Onizawa & Hanyu, 2024).
     * ``spsa``       - stalled pSA (Onizawa & Hanyu, 2024).
3. Device-variability modelling (timing, intensity, offset) drawn from
   per-p-bit normal distributions, matching Eq. (6)-(7) of Onizawa & Hanyu,
   "GPU-accelerated simulated annealing based on p-bits with real-world
   device-variability modeling" (2026).

All paths also run on CPU (NumPy) and on GPU without a CUDA toolchain
(CuPy), so the same code is usable for unit tests on a laptop.

Core update equation (per cycle t, per p-bit i):

    I_i(t)   = i0(t) * ( h_i + sum_j J_ij * sigma_j(t) )
    I_i'(t)  = lambda_i * ( I_i(t) + delta_i )           # intensity + offset
    sigma_i(t+1) = sgn( r_i(t) + tanh( I_i'(t) ) )        # r in U(-1, +1)

Timing variability: each p-bit updates only on cycles that satisfy
    t % (nu_i + 1) == 0
which reproduces the ``count_device mod nu[i] == 0`` gate in Listing 1 of the
reference paper.
"""

from __future__ import annotations

import time
import numpy as np


# ---------------------------------------------------------------------------
# Backend selection (auto-detected, can be overridden via constructor arg)
# ---------------------------------------------------------------------------
_BACKEND = "numpy"
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    _BACKEND = "cuda"
except Exception:  # pragma: no cover
    try:
        import cupy as cp  # noqa: F401
        _BACKEND = "cupy"
    except Exception:
        _BACKEND = "numpy"


# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------
# Design notes:
# * One thread == one p-bit update for the parallel modes (psa/tapsa/spsa).
#   This is exactly the pattern in Listing 1 of Onizawa & Hanyu (2026).
# * For ``sequential`` mode the host picks which p-bit to update and launches
#   a single-thread kernel; this is only used for correctness validation on
#   small circuits, not for the benchmark path.
# * tanhf + curand_uniform compile to fast hardware instructions on Ampere
#   so the inner loop over J is the only serious memory-bound step.  With
#   block size 32 (== warp size) the J row is read coalesced.

_CUDA_SRC = r"""
#include <curand_kernel.h>

extern "C" {

/* ---------- one-time RNG initialisation ------------------------------- */
__global__ void init_rng(curandState *state, unsigned long long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &state[idx]);
}

/* ---------- parallel update (pSA / TApSA / SpSA) ---------------------- *
 * mode == 0 : pSA   (plain)
 * mode == 1 : TApSA (time-averaged local field; host supplies I_avg in `tavg`)
 * mode == 2 : SpSA  (with probability p_stall, keep previous I_i)
 *
 *   n             : number of p-bits
 *   J, h, sigma   : coupling matrix, bias, spin vector
 *   lam, del, nu  : per-pbit variability arrays
 *   tavg          : length-n buffer holding the running time-average of I
 *                   (only read/written when mode == 1)
 *   Iprev         : length-n buffer with I_i(t-1) for SpSA (mode == 2)
 *   i0            : current pseudo-inverse temperature
 *   step          : global cycle index (for timing variability)
 *   p_stall       : SpSA stall probability (mode == 2 only)
 *   state         : per-pbit RNG state
 */
__global__ void pbit_update(
        int           n,
        const float  *__restrict__ J,
        const float  *__restrict__ h,
              float  *              sigma,
        const float  *__restrict__ lam,
        const float  *__restrict__ del,
        const int    *__restrict__ nu,
              float  *              tavg,
              float  *              Iprev,
        float         i0,
        int           step,
        int           mode,
        float         p_stall,
        float         tavg_alpha,   /* 1/alpha for TApSA; unused otherwise */
        curandState  *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* Timing variability: gate the update. */
    int nu_i = nu[i];
    if (nu_i > 0 && (step % (nu_i + 1)) != 0) return;

    /* --- compute local field I_i = h_i + sum_j J_ij * sigma_j --------- */
    float field = h[i];
    for (int j = 0; j < n; ++j) {
        field += J[i * n + j] * sigma[j];
    }
    float I_i = i0 * field;

    /* --- algorithm-specific modification of I_i ----------------------- */
    if (mode == 1) {                       /* TApSA */
        /* exponential moving average approximating the finite window used
           in Eq. (4b).  tavg_alpha = 1/alpha. */
        float avg = tavg[i] + tavg_alpha * (I_i - tavg[i]);
        tavg[i] = avg;
        I_i = avg;
    } else if (mode == 2) {                /* SpSA */
        float u = curand_uniform(&state[i]);
        if (u < p_stall) {
            I_i = Iprev[i];                /* stall */
        } else {
            Iprev[i] = I_i;                /* use new value, remember it */
        }
    } else if (mode == 3) {   /* CaSuDa retention dynamics */

    float s_prob = expf(
        -1.0f * expf(-1.0f * sigma[i] * I_i)
    );

    float rand_val =
        curand_uniform(&state[i]);

    float new_sigma =
        sigma[i] *
        (
            (s_prob - rand_val) > 0.0f
            ? 1.0f
            : -1.0f
        );

    sigma[i] =
        (new_sigma == 0.0f)
        ? sigma[i]
        : new_sigma;

    return;
}

    /* --- intensity + offset variability + stochastic update ----------- */
    float arg = lam[i] * (I_i + del[i]);
    float r   = 2.0f * curand_uniform(&state[i]) - 1.0f;
    float v   = r + tanhf(arg);
    sigma[i]  = (v > 0.0f) ?  1.0f : -1.0f;
}

/* ---------- sequential (Gibbs) update -------------------------------- *
 * Used only for correctness testing.  `which` is the index of the single
 * p-bit that should be updated this call.  No variability is applied so the
 * resulting chain has the exact Boltzmann distribution of (J, h).
 */
__global__ void pbit_update_one(
        int           n,
        const float  *__restrict__ J,
        const float  *__restrict__ h,
              float  *              sigma,
        float         i0,
        int           which,
        curandState  *state)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int i = which;
    float field = h[i];
    for (int j = 0; j < n; ++j) field += J[i * n + j] * sigma[j];
    float r = 2.0f * curand_uniform(&state[i]) - 1.0f;
    sigma[i] = (r + tanhf(i0 * field) > 0.0f) ? 1.0f : -1.0f;
}

} /* extern "C" */
"""


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------
class CudaSolver:
    """
    GPU-accelerated p-bit sampler / simulated annealer.

    Parameters
    ----------
    Nt : int
        Number of cycles (time steps).
    dt : float
        Time step, kept for API parity with the original CaSuDaSolver.
    i0 : float
        Pseudo-inverse temperature (final value if ``anneal=True``).
    update_mode : {"sequential", "psa", "tapsa", "spsa"}
        Which update rule the kernel applies.  ``sequential`` is single-p-bit
        Gibbs and is the only mode with guaranteed Boltzmann convergence.
    anneal : bool
        Linear ramp of i0 from ``i0_min`` to ``i0`` when True.
    i0_min : float
        Starting i0 for the anneal schedule.
    alpha : int
        Time-averaging window for TApSA (default 4, as in Onizawa 2026).
    p_stall : float
        Stall probability for SpSA (default 0.5).
    sigma_nu, sigma_lambda, sigma_delta : float
        Std-dev of the three device-variability distributions (timing,
        intensity, offset).  All zero => ideal-device pSA.
    backend : {"cuda", "cupy", "numpy", None}
        Execution backend; None auto-selects the fastest available.
    block_size : int
        CUDA threads per block (32 is a good default on Orin Nano).
    record_samples : bool
        If False, skip copying sigma back to the host every cycle - huge
        speedup when only the final energy / final state is needed.
    seed : int or None
    """

    _VALID_MODES = ("sequential", "psa", "tapsa", "spsa", "casuda")

    def __init__(self, Nt=10000, dt=0.1667, i0=0.8,
                 update_mode="psa",
                 anneal=False, i0_min=0.01,
                 alpha=4, p_stall=0.5,
                 sigma_nu=0.0, sigma_lambda=0.0, sigma_delta=0.0,
                 backend=None, block_size=32,
                 record_samples=True, seed=None):
        if update_mode not in self._VALID_MODES:
            raise ValueError(
                f"update_mode must be one of {self._VALID_MODES}, got {update_mode!r}.")
        self.Nt = int(Nt)
        self.dt = float(dt)
        self.i0 = float(i0)
        self.update_mode = update_mode
        self.anneal = bool(anneal)
        self.i0_min = float(i0_min)
        self.alpha = int(alpha)
        self.p_stall = float(p_stall)
        self.sigma_nu = float(sigma_nu)
        self.sigma_lambda = float(sigma_lambda)
        self.sigma_delta = float(sigma_delta)
        self.block_size = int(block_size)
        self.record_samples = bool(record_samples)
        self.seed = seed

        self.backend = backend or _BACKEND
        if self.backend not in ("cuda", "cupy", "numpy"):
            raise ValueError(f"Unknown backend: {self.backend}")
        self.last_time = None

        if self.backend == "cuda":
            mod = SourceModule(_CUDA_SRC, no_extern_c=True,
                               options=["-use_fast_math"])
            self._k_update = mod.get_function("pbit_update")
            self._k_update_one = mod.get_function("pbit_update_one")
            self._k_init = mod.get_function("init_rng")

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _extract_matrices(circuit):
        if hasattr(circuit, "J") and hasattr(circuit, "h"):
            J = np.asarray(circuit.J, dtype=np.float32)
            h = np.asarray(circuit.h, dtype=np.float32).reshape(-1)
        elif isinstance(circuit, (tuple, list)) and len(circuit) == 2:
            J = np.asarray(circuit[0], dtype=np.float32)
            h = np.asarray(circuit[1], dtype=np.float32).reshape(-1)
        else:
            raise TypeError(
                "Circuit must expose J and h attributes or be a (J, h) tuple.")
        n = h.shape[0]
        if J.shape != (n, n):
            raise ValueError(f"J must be {n}x{n}, got {J.shape}.")
        return J, h, n

    def _draw_variability(self, n, rng):
        lam = rng.normal(1.0, self.sigma_lambda, size=n).astype(np.float32)
        lam = np.clip(lam, 1e-3, None)
        dlt = rng.normal(0.0, self.sigma_delta, size=n).astype(np.float32)
        nu_f = np.abs(rng.normal(0.0, self.sigma_nu, size=n))
        nu = np.clip(np.round(nu_f), 0, None).astype(np.int32)
        return lam, dlt, nu

    def _i0_schedule(self, t):
        if not self.anneal:
            return self.i0
        return self.i0_min + (self.i0 - self.i0_min) * (t / max(1, self.Nt - 1))

    # ---- entry point ----------------------------------------------------
    def solve(self, circuit):
        """
        Returns
        -------
        energies : (Nt,) float32 - H = -sigma^T J sigma - h^T sigma per cycle.
        samples  : (Nt, n) float32 - spin states per cycle, or an empty array
                   if ``record_samples=False``.
        """
        J, h, n = self._extract_matrices(circuit)
        rng = np.random.default_rng(self.seed)
        lam, dlt, nu = self._draw_variability(n, rng)

        t0 = time.perf_counter()
        if self.backend == "cuda":
            energies, samples = self._solve_cuda(J, h, n, lam, dlt, nu, rng)
        elif self.backend == "cupy":
            energies, samples = self._solve_cupy(J, h, n, lam, dlt, nu, rng)
        else:
            energies, samples = self._solve_numpy(J, h, n, lam, dlt, nu, rng)
        self.last_time = time.perf_counter() - t0
        return energies, samples

    # ==================================================================
    # Backend 1: raw CUDA
    # ==================================================================
    def _solve_cuda(self, J, h, n, lam, dlt, nu, rng):
        block = (self.block_size, 1, 1)
        grid = ((n + self.block_size - 1) // self.block_size, 1, 1)

        sigma0 = rng.choice([-1.0, 1.0], size=n).astype(np.float32)

        J_gpu = gpuarray.to_gpu(J)
        h_gpu = gpuarray.to_gpu(h)
        sigma_gpu = gpuarray.to_gpu(sigma0)
        lam_gpu = gpuarray.to_gpu(lam)
        dlt_gpu = gpuarray.to_gpu(dlt)
        nu_gpu = gpuarray.to_gpu(nu)
        tavg_gpu = gpuarray.zeros(n, dtype=np.float32)
        iprev_gpu = gpuarray.zeros(n, dtype=np.float32)

        state_gpu = cuda.mem_alloc(64 * n)     # generous: 48B per curandState
        seed = np.uint64(rng.integers(0, 2**63 - 1))
        self._k_init(state_gpu, seed, np.int32(n), block=block, grid=grid)

        mode = {
            "psa": 0,
            "tapsa": 1,
            "spsa": 2,
            "casuda": 3,
            "sequential": -1
        }[self.update_mode]
        tavg_alpha = np.float32(1.0 / max(1, self.alpha))

        if self.record_samples:
            samples = np.empty((self.Nt, n), dtype=np.float32)
        else:
            samples = np.empty((0, n), dtype=np.float32)
        energies = np.empty(self.Nt, dtype=np.float32)

        for t in range(self.Nt):
            i0_t = np.float32(self._i0_schedule(t))

            if mode == -1:
                # Sequential Gibbs: pick one site at random and launch the
                # single-thread kernel.
                which = int(rng.integers(0, n))
                self._k_update_one(
                    np.int32(n), J_gpu, h_gpu, sigma_gpu,
                    i0_t, np.int32(which), state_gpu,
                    block=(1, 1, 1), grid=(1, 1, 1))
            else:
                self._k_update(
                    np.int32(n), J_gpu, h_gpu, sigma_gpu,
                    lam_gpu, dlt_gpu, nu_gpu,
                    tavg_gpu, iprev_gpu,
                    i0_t, np.int32(t), np.int32(mode),
                    np.float32(self.p_stall), tavg_alpha,
                    state_gpu, block=block, grid=grid)

            s = sigma_gpu.get()
            if self.record_samples:
                samples[t] = s
            energies[t] = -s @ J @ s - h @ s

        return energies, samples

    # ==================================================================
    # Backend 2: CuPy
    # ==================================================================
    def _solve_cupy(self, J, h, n, lam, dlt, nu, rng):
        import cupy as cp
        J_d = cp.asarray(J)
        h_d = cp.asarray(h)
        lam_d = cp.asarray(lam)
        dlt_d = cp.asarray(dlt)
        nu_d = cp.asarray(nu)
        sigma = cp.asarray(rng.choice([-1.0, 1.0], size=n).astype(np.float32))
        tavg = cp.zeros(n, dtype=cp.float32)
        Iprev = cp.zeros(n, dtype=cp.float32)

        if self.record_samples:
            samples = np.empty((self.Nt, n), dtype=np.float32)
        else:
            samples = np.empty((0, n), dtype=np.float32)
        energies = np.empty(self.Nt, dtype=np.float32)

        for t in range(self.Nt):
            i0_t = float(self._i0_schedule(t))
            if self.update_mode == "sequential":
                i = int(rng.integers(0, n))
                field = float(h_d[i] + cp.dot(J_d[i], sigma))
                r = 2.0 * np.random.random() - 1.0
                sigma[i] = 1.0 if (r + np.tanh(i0_t * field)) > 0.0 else -1.0

            else:

                I = i0_t * (h_d + J_d @ sigma)

                # -------------------------------------------------
                # TApSA
                # -------------------------------------------------

                if self.update_mode == "tapsa":

                    tavg += (
                                    1.0 / self.alpha
                            ) * (I - tavg)

                    I = tavg

                # -------------------------------------------------
                # SpSA
                # -------------------------------------------------

                elif self.update_mode == "spsa":

                    u = cp.random.random(
                        n,
                        dtype=cp.float32
                    )

                    stall = u < self.p_stall

                    I = cp.where(
                        stall,
                        Iprev,
                        I
                    )

                    Iprev = cp.where(
                        stall,
                        Iprev,
                        I
                    )

                # -------------------------------------------------
                # CaSuDa
                # -------------------------------------------------

                elif self.update_mode == "casuda":

                    s_prob = cp.exp(
                        -1.0 * self.dt *
                        cp.exp(-1.0 * sigma * I)
                    )

                    rand_vals = cp.random.random(
                        n,
                        dtype=cp.float32
                    )

                    new_sigma = sigma * cp.sign(
                        s_prob - rand_vals
                    )

                    sigma = cp.where(
                        new_sigma == 0,
                        sigma,
                        new_sigma
                    )

                    energy_gpu = (
                            -sigma @ J_d @ sigma
                            - h_d @ sigma
                    )

                    energies[t] = float(
                        cp.asnumpy(energy_gpu)
                    )

                    if self.record_samples:
                        samples[t] = cp.asnumpy(sigma)

                    continue

                # -------------------------------------------------
                # Standard pSA
                # -------------------------------------------------

                arg = lam_d * (I + dlt_d)

                r = (
                        2.0 *
                        cp.random.random(
                            n,
                            dtype=cp.float32
                        )
                        - 1.0
                )

                new_sigma = cp.where(
                    r + cp.tanh(arg) > 0.0,
                    1.0,
                    -1.0
                ).astype(cp.float32)

                t_mask = (
                        (nu_d == 0)
                        |
                        ((t % (nu_d + 1)) == 0)
                )

                sigma = cp.where(
                    t_mask,
                    new_sigma,
                    sigma
                )
            # Compute energy fully on GPU
            energy_gpu = -sigma @ J_d @ sigma - h_d @ sigma
            energies[t] = float(cp.asnumpy(energy_gpu))

            # Only transfer samples if needed
            if self.record_samples:
                samples[t] = cp.asnumpy(sigma)

        return energies, samples

    # ==================================================================
    # Backend 3: NumPy reference
    # ==================================================================
    def _solve_numpy(self, J, h, n, lam, dlt, nu, rng):
        sigma = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        tavg = np.zeros(n, dtype=np.float32)
        Iprev = np.zeros(n, dtype=np.float32)

        if self.record_samples:
            samples = np.empty((self.Nt, n), dtype=np.float32)
        else:
            samples = np.empty((0, n), dtype=np.float32)
        energies = np.empty(self.Nt, dtype=np.float32)

        for t in range(self.Nt):
            i0_t = self._i0_schedule(t)
            if self.update_mode == "sequential":
                i = int(rng.integers(0, n))
                field = h[i] + J[i] @ sigma
                r = rng.uniform(-1.0, 1.0)
                sigma[i] = 1.0 if (r + np.tanh(i0_t * field)) > 0.0 else -1.0
            else:
                I = i0_t * (h + J @ sigma)
                # -----------------------------------------
                # TApSA
                # -----------------------------------------

                if self.update_mode == "tapsa":

                    tavg += (
                                    1.0 / self.alpha
                            ) * (I - tavg)

                    I = tavg.copy()

                # -----------------------------------------
                # SpSA
                # -----------------------------------------

                elif self.update_mode == "spsa":

                    u = rng.uniform(
                        0.0,
                        1.0,
                        size=n
                    )

                    stall = u < self.p_stall

                    I_new = np.where(
                        stall,
                        Iprev,
                        I
                    )

                    Iprev = np.where(
                        stall,
                        Iprev,
                        I
                    )

                    I = I_new

                # -----------------------------------------
                # CaSuDa Retention Dynamics
                # -----------------------------------------

                elif self.update_mode == "casuda":

                    s_prob = np.exp(
                        -1.0 * self.dt *
                        np.exp(-1.0 * sigma * I)
                    )

                    rand_vals = rng.uniform(
                        0.0,
                        1.0,
                        size=n
                    )

                    new_sigma = sigma * np.sign(
                        s_prob - rand_vals
                    )

                    sigma = np.where(
                        new_sigma == 0,
                        sigma,
                        new_sigma
                    )

                    if self.record_samples:
                        samples[t] = sigma

                    energies[t] = (
                            -sigma @ J @ sigma
                            - h @ sigma
                    )

                    continue

                # -----------------------------------------
                # Standard pSA
                # -----------------------------------------

                arg = lam * (I + dlt)

                r = rng.uniform(
                    -1.0,
                    1.0,
                    size=n
                ).astype(np.float32)

                new_sigma = np.where(
                    r + np.tanh(arg) > 0.0,
                    1.0,
                    -1.0
                ).astype(np.float32)

                t_mask = (
                        (nu == 0)
                        |
                        ((t % (nu + 1)) == 0)
                )

                sigma = np.where(
                    t_mask,
                    new_sigma,
                    sigma
                )

            if self.record_samples:
                samples[t] = sigma
            energies[t] = -sigma @ J @ sigma - h @ sigma

        return energies, samples


# ---------------------------------------------------------------------------
# Backwards-compatible alias.  Makes the drop-in replacement literal:
#
#     from p_kit.solver.cuda_solver import CaSuDaSolver
#     solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)
#     _, output = solver.solve(c)                  # unchanged
# ---------------------------------------------------------------------------
class CaSuDaSolver(CudaSolver):
    """Alias matching the original IBM/p-kit class name."""
    pass
