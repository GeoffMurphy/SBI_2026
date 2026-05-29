"""
SBI utilities: prior definition, summary statistics, and normalisation.

Summary statistic x is the concatenation of:
  1. signal_bandpowers  : empirical P(k) per k-bin from d  — shape (n_kbins,)
  2. fg_freq_covariance : upper triangle of freq-freq cov of d — shape (n_freq*(n_freq+1)//2,)

Both are normalised to zero mean / unit variance using statistics fitted over
a small set of prior-predictive simulations before training.
"""

import numpy as np
from gibbs_utils import binner


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def make_prior(log_Ps_min, log_Ps_max, n_kbins):
    """
    Independent log-uniform prior over each P_s k-bin.

    Parameters
    ----------
    log_Ps_min : float or array-like, shape (n_kbins,)
        Lower bound(s) in log10 P_s.
    log_Ps_max : float or array-like, shape (n_kbins,)
        Upper bound(s) in log10 P_s.
    n_kbins : int

    Returns
    -------
    prior : sbi.utils.BoxUniform
    """
    import torch
    from sbi.utils import BoxUniform

    low  = torch.full((n_kbins,), float(log_Ps_min)) if np.isscalar(log_Ps_min) \
           else torch.tensor(log_Ps_min, dtype=torch.float32)
    high = torch.full((n_kbins,), float(log_Ps_max)) if np.isscalar(log_Ps_max) \
           else torch.tensor(log_Ps_max, dtype=torch.float32)
    return BoxUniform(low=low, high=high)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_signal_bandpowers(d, n_kbins, cube_len, U_f):
    """
    Empirical power spectrum of the foreground-subtracted data per radial k-bin.

    Subtracts the U_f-projected foreground from d before measuring band powers,
    so the residual contains signal + noise + any FG power outside the n_modes
    subspace.  This removes the dominant FG contribution that otherwise swamps
    the signal in low-k bins.

    Parameters
    ----------
    d        : np.ndarray, shape (nx, ny, n_freq)
    n_kbins  : int
    cube_len : int — side length of the cube (assumed cubic, use nx)
    U_f      : np.ndarray, shape (n_modes, n_freq)

    Returns
    -------
    band_powers : np.ndarray, shape (n_kbins,)
        Mean |d_res_k|² per bin.
    """
    nx, ny, n_freq = d.shape
    d_2d  = d.reshape(nx * ny, n_freq)
    a     = d_2d @ U_f.T                      # (n_pix, n_modes) — FG amplitudes
    d_fg  = (a @ U_f).reshape(d.shape)        # reconstructed foreground
    d_res = d - d_fg                           # signal + noise residual

    d_k = np.fft.fftn(d_res - d_res.mean(), norm='ortho').flatten()
    binned_d, _, _ = binner(d_k, n_kbins, cube_len)
    return np.array([np.mean(np.abs(b) ** 2) for b in binned_d])


def compute_fg_frequency_covariance(d, U_f):
    """
    Upper triangle of the scatter matrix of d projected onto the foreground basis.

    Steps:
      1. Reshape d to (n_pix, n_freq)
      2. Project: a = d_2d @ U_f.T  →  (n_pix, n_modes)
      3. Scatter: C = a.T @ a / n_pix  →  (n_modes, n_modes)
      4. Return upper triangle  →  n_modes*(n_modes+1)//2 elements

    This is the sufficient statistic for the IW posterior on C_f, matching
    what the simulator draws from internally.

    Parameters
    ----------
    d   : np.ndarray, shape (nx, ny, n_freq)
    U_f : np.ndarray, shape (n_modes, n_freq)  — foreground PCA basis

    Returns
    -------
    cov_vec : np.ndarray, shape (n_modes*(n_modes+1)//2,)
    """
    nx, ny, n_freq = d.shape
    n_modes = U_f.shape[0]
    a    = d.reshape(nx * ny, n_freq) @ U_f.T        # (n_pix, n_modes)
    cov  = (a.T @ a) / (nx * ny)                     # (n_modes, n_modes)
    return cov[np.triu_indices(n_modes)]


def compute_summary_stats(d, binner_args, U_f):
    """
    Concatenate signal band powers and projected fg covariance into one vector.

    Total length: n_kbins + n_modes*(n_modes+1)//2

    Parameters
    ----------
    d           : np.ndarray, shape (nx, ny, n_freq)
    binner_args : dict with keys 'n_kbins' and 'cube_len'
    U_f         : np.ndarray, shape (n_modes, n_freq)

    Returns
    -------
    x : np.ndarray, shape (n_kbins + n_modes*(n_modes+1)//2,)
    """
    bp  = compute_signal_bandpowers(d, **binner_args, U_f=U_f)
    cov = compute_fg_frequency_covariance(d, U_f)
    return np.concatenate([bp, cov])


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class Normaliser:
    """
    Zero-mean / unit-variance normalisation fitted from prior-predictive sims.

    Usage:
        norm = Normaliser()
        X_norm = norm.fit_transform(X_prior_predictive)   # fit on ~500 sims
        X_train_norm = norm.transform(X_train)
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        """
        Estimate mean and std from a batch of summary statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_sims, n_summary)
        """
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # Guard against zero-variance features (e.g. constant bins)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """
        Apply (X - mean) / std.

        Parameters
        ----------
        X : np.ndarray, shape (..., n_summary)

        Returns
        -------
        X_norm : np.ndarray, same shape as X
        """
        assert self.mean_ is not None, "Call fit() first."
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
