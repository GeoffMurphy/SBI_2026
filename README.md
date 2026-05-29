# SBI_2026

A proof-of-concept applying **simulation-based inference (SBI)** to 21cm intensity mapping. The goal is to infer the 21cm signal power spectrum P_s(k) using Neural Posterior Estimation (NPE-C), and to compare directly with the Gibbs sampler in [IM_Gibbs](https://github.com/GeoffMurphy/IM_Gibbs).

## Method

The data model is:

```
d(x, ν) = U_f(ν) · a(x) + U_s · s(x) + n(x, ν)
```

The parameter of interest **θ** is `log10 P_s` per radial k-bin (14 bins). Foreground amplitudes are treated as nuisance variables and marginalised implicitly through the simulator.

A normalising flow is trained on (θ, x) pairs from the simulator, where the summary statistic **x** concatenates the empirical band powers of the data cube with the upper triangle of its frequency-frequency covariance. Once trained, the posterior p(θ | x_obs) is recovered in a single forward pass.

## Structure

| File | Description |
|------|-------------|
| `sbi_demo.ipynb` | Main notebook — training, inference, and comparison with Gibbs |
| `simulator.py` | Draws (θ, x) pairs for training |
| `sbi_utils.py` | Summary statistics, prior definition, normalisation |
| `linear_system.py` | Linear system operators (U_f, U_s) |
| `gibbs_utils.py` | Shared utilities — k-bin tools, foreground covariance sampler |
| `fastbox/` | Signal simulation package |

## Data

The `data/` directory contains simulation inputs (foreground maps, HI signal cubes, frequency arrays) as `.npy` files. These are **not included in the repository** due to file size (~500 MB). They are required to run the simulator and notebook.

## Dependencies

- [`sbi`](https://github.com/sbi-dev/sbi)
- `fastbox` (included as a subpackage)
- `numpy`, `scipy`, `torch`
