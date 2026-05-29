# SBI POC — Working Notes

A living document covering the design, implementation decisions, and practical
gotchas encountered while building this SBI proof-of-concept. Intended as a
companion to `sbi_poc_brief.md` and `sbi_demo.ipynb`.

---

## Overview

We are using **Neural Posterior Estimation (NPE-C)** via the `sbi` package to
infer the 21cm signal power spectrum P_s(k) from a simulated intensity-mapping
data cube. The data model is identical to the Gibbs sampler in
`../Gibbs_2026/Gibbs_update_2.ipynb`, enabling a direct comparison of the two
approaches on the same observation.

The key idea is: instead of sampling the posterior analytically (as Gibbs does),
we train a normalising flow on (θ, x) pairs produced by the simulator. Once
trained, the flow gives the posterior p(θ | x_obs) in a single forward pass —
no iterative sampling required.

---

## Data Model

```
d(x, ν) = U_f(ν) · a(x) + U_s · s(x) + n(x, ν)
```

| Symbol | Description |
|--------|-------------|
| `s` | 21cm signal, Gaussian random field with power spectrum P_s(k) |
| `a(x)` | Foreground PCA amplitudes per pixel, drawn from N(0, C_f) |
| `U_f` | Frequency basis (n_modes × n_freq PCA eigenvectors) |
| `U_s` | FFT operator — `rfftn` / `irfftn` with `norm='ortho'` |
| `n` | White noise, N(0, σ²I), σ = 0.165 K |

**θ** is `log10 P_s` per radial k-bin (14 bins). Foreground parameters are
nuisance variables marginalised implicitly by the simulator.

---

## File Structure

| File | Role |
|------|------|
| `simulator.py` | `Simulator` class — draws one (θ, x) pair per call |
| `sbi_utils.py` | Summary statistics, prior, normaliser |
| `linear_system.py` | Copy of `Gibbs_2026/Define_Linear_Equation_PCA_update_2.py` |
| `gibbs_utils.py` | Copy of `Gibbs_2026/gibbs_utils_update_2.py` |
| `sbi_demo.ipynb` | End-to-end notebook: PPC → training → inference → SBC |
| `data/` | Shared data files (signal, foreground, noise, freqs, PCA evecs) |
| `outputs/` | Saved training arrays (`thetas_train.npy`, `xs_train.npy`) |

---

## Simulator Design

### Signal generation (`_draw_signal`)

Rather than using fastbox's cosmological pipeline, we generate the GRF directly
by colouring white noise in rfft space:

```python
w   = np.random.normal(0, 1, shape)
s_k = rfftn(w, norm='ortho') * sqrt(P_s_3d)
s_x = irfftn(s_k, norm='ortho')
```

With this convention, `E[|s_k|²] = P_s(k)` per mode — the same definition used
by the Gibbs signal covariance sampler, so θ is directly comparable.

### Foreground generation (`_draw_foreground`)

Each simulation draws a fresh foreground covariance from an inverse-Wishart:

```
C_f ~ IW(a_true.T @ a_true, n_pix)
a(x) ~ N(0, C_f)  for each pixel x
d_fg = Uf(U_f, a, transpose=False)
```

`a_true` is computed once in `setup()` by projecting the single expensive
foreground simulation onto the PCA basis.  Passing `a_true` (not `C_f_hat`)
directly to `foreground_covariance_sampler` is correct — the function computes
`Ψ = a.T @ a` internally and draws from IW(Ψ, n_pix).

**Important:** pass mean-subtracted amplitudes to the IW sampler (Gibbs
convention). Passing raw amplitudes without mean-subtraction can produce a
singular or near-singular Ψ.

### k-bin convention

The `binner()` function in `gibbs_utils.py` uses **pixel-unit** k-magnitudes
(integer mode indices, no 2π/L factor). The same bins are used in both the
signal GRF generation (`_broadcast_Ps`) and the summary statistic
(`compute_signal_bandpowers`), so θ and x are on a consistent scale.  To
convert bin centres to physical Mpc⁻¹:

```python
k_phys = k_pix_centre * 2 * pi / L    # L = box_dims[0] in Mpc
```

---

## Summary Statistics

The summary statistic vector `x` (length 24) is:

```
x = [ band_powers (14) | fg_cov_upper_triangle (10) ]
```

### Band powers — FG-subtracted residual

We measure band powers of the **foreground-subtracted residual**, not the raw
data:

```python
a     = d_2d @ U_f.T          # project onto FG basis
d_res = d - (a @ U_f)         # remove reconstructed foreground
bp    = mean(|FFT(d_res)|²) per k-bin
```

**Why:** The raw data in foreground-dominated bins (k-bins 0–10) has band powers
~10⁵, completely swamping the 21cm signal (~10⁻¹ to 10⁻²). The NPE trained on
raw band powers learns the association "high band power → low P_s", producing a
systematic 1–2 dex underestimate.  After FG subtraction, the residual is
signal + noise, and the band powers are directly informative about P_s across
all bins. Bins 11–13 were recovered correctly even without this fix (they are
signal-dominated), which confirmed the diagnosis.

### Foreground covariance — projected scatter matrix

We use the **upper triangle of the scatter matrix in the PCA-projected space**:

```python
a    = d_2d @ U_f.T            # (n_pix, n_modes)
cov  = a.T @ a / n_pix         # (n_modes, n_modes)
vec  = cov[triu_indices]       # 10 elements for n_modes=4
```

**Why not the full 128×128 frequency covariance?** The full upper triangle is
8256 elements — far too high-dimensional for the default MAF given n_params=14.
More importantly, the foreground model draws C_f in the n_modes-dimensional
space; the projected scatter matrix is exactly the sufficient statistic for
the IW posterior on C_f.  The remaining 124 frequency dimensions mostly carry
signal + noise, which is already captured by the band powers.

---

## Practical Notes & Gotchas

### `construct_Uf` is degenerate at 629–949 MHz

`construct_Uf(n_modes, shape, freqs)` builds power-law frequency modes centred
at `ref_freq = 130 MHz`. At the actual data frequencies (629–949 MHz),
`log10(freqs/130)` only varies from 0.68 to 0.86, making all basis vectors
nearly collinear. The resulting scatter matrix Ψ has condition number ~10¹⁶
and a slightly negative eigenvalue, causing the Cholesky in the IW sampler to
fail.

**Fix:** load the precomputed PCA eigenvectors from
`data/fg_evecs.npy` (condition number 1.0) — exactly what
`Gibbs_update_2.ipynb` uses at runtime anyway. `construct_Uf` is kept in
`linear_system.py` for completeness but is not used by the simulator.

### NPE posterior sampling: 0% rejection acceptance

`posterior.sample()` applies a prior-support rejection step — any flow sample
landing outside the `BoxUniform` bounds is discarded.  With tight per-bin
priors and a flow that can extrapolate slightly beyond them, this gives near-0%
acceptance.

Attempting to sample directly from `density_estimator` hits a version
inconsistency: the keyword argument for the conditioning variable differs across
`sbi` releases (`context=`, `x=`, or positional only), making that path
fragile.  The robust fix is to address the root cause: rebuild the posterior
with bounds wide enough that the rejection gate never triggers.  The trained
flow is unchanged.

```python
from sbi.utils import BoxUniform
flat_prior = BoxUniform(
    low =torch.full((n_kbins,), -20.0),
    high=torch.full((n_kbins,),  20.0),
)
posterior_sample = inference.build_posterior(density_estimator, prior=flat_prior)
samples = posterior_sample.sample((N_SAMPLES,), x=x_tensor).numpy()
```

### `sample_with='mcmc'` gave poor results

MCMC posterior sampling for NPE works by using the flow as a likelihood
approximation and running MCMC in θ-space. With a 14-dimensional parameter
space and a flow that has not fully converged (e.g. after training on
FG-dominated summary stats), the MCMC chain mixes poorly. Direct sampling from
the density estimator is faster and more reliable for a well-trained NPE.
