# SBI POC — Implementation Brief

## Context

This is a proof-of-concept for simulation-based inference (SBI) applied to 21cm intensity mapping, designed to be directly comparable to the Gibbs sampler in the IM_Gibbs repository (https://github.com/GeoffMurphy/IM_Gibbs). Keep the scope, structure, and data model as close to that repo as possible.

---

## Data Model

Same as IM_Gibbs:

```
d(x, ν) = U_f(ν) · a(x) + U_s · s(x) + n(x, ν)
```

- `s`: 21cm signal, Gaussian random field with power spectrum P_s(k)
- `a(x)`: foreground frequency amplitudes per pixel, drawn from N(0, C_f)
- `U_f`: frequency basis (power-law modes), same as `construct_Uf()` in `linear_system.py`
- `U_s`: FFT operator, same as `Us()` in `linear_system.py`
- `n`: white noise with known covariance N

---

## Parameters θ

For this POC, θ is **P_s only** — the signal power spectrum values per k-bin (same binning as in IM_Gibbs). Foreground parameters are treated as nuisance and marginalised implicitly through the simulator.

Use log-uniform priors over each P_s bin, with bounds informed by the fiducial power spectrum used in IM_Gibbs.

---

## Simulator

The simulator must produce one `(θ, x)` pair per call, where `x` is the summary statistic (see below). Each call should:

1. **Draw signal**: use `fastbox` to generate a Gaussian random field with power spectrum P_s(k) matching θ. Apply `Us()` to go to real space.

2. **Draw foreground realisation**:
   - From the single expensive foreground simulation, estimate the frequency-frequency covariance:
     `C_f_hat = (1/n_pix) * Σ_x a(x) a(x)^T`
   - Sample a new `C_f` from an inverse-Wishart distribution with scale matrix `Ψ = C_f_hat * n_pix` and `ν = n_pix` degrees of freedom. This mirrors `foreground_covariance_sampler()` in `gibbs_utils.py`.
   - Draw new foreground amplitudes `a(x) ~ N(0, C_f)` for each pixel.
   - Apply `U_f` to get the frequency-space foreground map.

3. **Draw noise**: sample `n ~ N(0, N)` fresh for each simulation.

4. **Form data cube**: `d = foreground + signal + noise`

5. **Compute and return summary statistic** (see below).

---

## Summary Statistics

Two statistics concatenated into a single vector `x`:

1. **Signal band powers**: empirical power spectrum of `d` per k-bin (same bins as θ). Use the k-bin machinery from `gibbs_utils.py` (`k_vecs`, `binner`).

2. **Foreground frequency covariance**: upper triangle of the sample frequency-frequency covariance of `d` across spatial pixels, flattened to a vector.

Normalise both statistics (zero mean, unit variance) using statistics computed over a small set of prior-predictive simulations before training.

---

## SBI Algorithm

Use **NPE-C** (Neural Posterior Estimation) via the `sbi` Python package:

```python
from sbi.inference import NPE
from sbi.utils import BoxUniform
```

- Train on O(20k) simulations to start; increase if posteriors are poorly calibrated.
- Use the default masked autoregressive flow (MAF) posterior estimator — no need to tune architecture for a POC.
- Training simulations can be generated in parallel (the simulator is embarrassingly parallel).

---

## File Structure

Mirror IM_Gibbs as closely as possible:

```
sbi_demo.ipynb        # Main notebook (mirrors gibbs_demo.ipynb)
simulator.py          # Simulator wrapper: draw (θ, x) pairs
sbi_utils.py          # Summary statistic functions, prior definition, normalisation
linear_system.py      # Copy/reuse from IM_Gibbs unchanged
gibbs_utils.py        # Copy/reuse from IM_Gibbs (C_f sampler, k-bin tools)
```

---

## Validation

1. **Compare to Gibbs**: run both on the same simulated observation (fix a θ_true, generate one `d`). Overlay the P_s posterior from SBI with the Gibbs chain marginals. They should agree within noise.

2. **Simulation-based calibration (SBC)**: use `sbi`'s built-in SBC tools to check that the posterior is well-calibrated across the prior. Flag any k-bins where coverage is off.

3. **Prior predictive check**: plot a handful of simulated data cubes and their summary statistics to sanity-check the simulator before training.

---

## Dependencies

- `sbi` (Cranmer/Lueckmann et al.)
- `fastbox` (signal simulation)
- `numpy`, `scipy` (covariance sampling, FFTs)
- Existing `linear_system.py` and `gibbs_utils.py` from IM_Gibbs

---

## Out of Scope for This POC

- Learned data compression (use hand-crafted summary statistics only)
- C_f as part of θ (foreground parameters are nuisance only)
- Systematics, beam, or mask effects
- Sequential SBI (SNPE) — use amortised NPE for simplicity
- Large data cubes (keep the same small cube size as IM_Gibbs)
