"""
Simulator for SBI POC — 21cm intensity mapping.

Data model:  d(x, ν) = U_f(ν) · a(x) + U_s · s(x) + n(x, ν)

  s    : 21cm signal GRF with power spectrum P_s(k)            [shape^3 real cube]
  a(x) : foreground PCA amplitudes per pixel, drawn from N(0, C_f) [n_pix × n_modes]
  U_f  : frequency basis (power-law modes), from construct_Uf()
  U_s  : FFT operator, Us()
  n    : white noise, N(0, σ²I)

θ is log10 P_s per k-bin (length n_kbins).
x is the concatenated summary statistic computed by sbi_utils.compute_summary_stats().
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from linear_system import construct_Uf, Us, Uf
from gibbs_utils import foreground_covariance_sampler, k_vecs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    """
    Return a dict of simulation parameters matching Gibbs_update_2.ipynb.

    All paths are relative to the SBI_2026 project root.
    """
    return dict(
        shape       = (128, 128, 128),       # (nx, ny, nfreq)
        n_modes     = 4,                     # foreground PCA modes
        n_kbins     = 14,                    # radial k-bins for P_s
        noise_std   = 0.165,                 # K, i.i.d. Gaussian noise
        box_dims    = (1e3/0.678,) * 3,      # comoving Mpc per side (~1478 Mpc)
        data_dir    = os.path.join(os.path.dirname(__file__), 'data'),
        fg_file     = 'T_FG_nopol.npy',
        signal_file = 'T_HI_z=0.39_NoRSD.npy',
        freqs_file  = 'freqs.npy',
        evecs_file  = 'fg_evecs.npy',    # PCA eigenvectors, shape (n_evecs, n_freq)
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    One call returns (theta, x): draw θ from outside, get back summary stats x.

    Usage:
        sim = Simulator(load_config())
        sim.setup()                          # loads data, builds C_f_hat and U_f
        x = sim(theta)                       # theta: (n_kbins,) log10 P_s per bin
    """

    def __init__(self, config):
        self.cfg = config
        self.shape    = config['shape']
        self.n_modes  = config['n_modes']
        self.n_kbins  = config['n_kbins']
        self.noise_std = config['noise_std']

        # Set in setup()
        self.U_f      = None   # (n_modes, n_freq) frequency basis
        self.a_true   = None   # (n_pix, n_modes)  projected FG amplitudes → C_f prior
        self.freqs    = None   # (n_freq,) frequency array
        self._ready   = False

    def setup(self):
        """
        One-time expensive setup:
          - loads data files
          - builds U_f from construct_Uf()
          - projects the true FG cube onto the frequency basis to get a_true
          - C_f_hat = a_true.T @ a_true / n_pix  (stored implicitly via a_true)
        """
        cfg = self.cfg
        data_dir = cfg['data_dir']
        shape    = self.shape

        # Frequency array — slice to match n_freq = shape[2]
        freqs_full = np.load(os.path.join(data_dir, cfg['freqs_file']))
        self.freqs = freqs_full[:shape[2]]

        # Frequency basis: first n_modes PCA eigenvectors, shape (n_modes, n_freq).
        # construct_Uf() produces a power-law basis that becomes degenerate at the
        # actual frequency range (629-949 MHz), so we use the precomputed PCA evecs
        # from the Gibbs repo instead — exactly as Gibbs_update_2.ipynb does at runtime.
        evecs_all = np.load(os.path.join(data_dir, cfg['evecs_file']))
        self.U_f = evecs_all[:self.n_modes, :shape[2]].real   # (n_modes, n_freq)

        # Load single expensive foreground simulation: (nx, ny, n_freq)
        fg_raw = np.load(os.path.join(data_dir, cfg['fg_file']))
        fg_cube = fg_raw[:shape[0], :shape[1], :shape[2]]

        # Project onto frequency basis to get per-pixel amplitudes: (n_pix, n_modes)
        n_pix = shape[0] * shape[1]
        fg_2d = fg_cube.reshape(n_pix, shape[2])    # (n_pix, n_freq)
        self.a_true = fg_2d @ self.U_f.T            # (n_pix, n_modes)

        self._ready = True

    def __call__(self, theta):
        """
        Draw one simulation and return the summary statistic x.

        Parameters
        ----------
        theta : array-like, shape (n_kbins,)
            log10 P_s per radial k-bin.

        Returns
        -------
        x : np.ndarray, shape (n_summary,)
            Concatenated summary statistic (see sbi_utils.compute_summary_stats).
        """
        assert self._ready, "Call setup() before simulating."

        from sbi_utils import compute_summary_stats

        signal     = self._draw_signal(theta)
        foreground = self._draw_foreground()
        noise      = self._draw_noise()
        d          = self._form_data(signal, foreground, noise)

        binner_args = dict(n_kbins=self.n_kbins, cube_len=self.shape[0])
        return compute_summary_stats(d, binner_args, self.U_f)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_signal(self, log_Ps):
        """
        Generate a 21cm signal cube in real space.

        Colours white noise in rfft space: each mode k gets variance P_s(k_bin).
        Convention matches Us / the Gibbs signal covariance sampler — P_s(k) is
        the expected value of |s_k|² where s_k = rfftn(s_x, norm='ortho')[k].

        Parameters
        ----------
        log_Ps : array-like, shape (n_kbins,)

        Returns
        -------
        signal_cube : np.ndarray, shape self.shape
        """
        P_s = 10.0 ** np.asarray(log_Ps)           # (n_kbins,) linear units
        k_rfft  = self._rfft_kmag()                  # (N, N, N//2+1) pixel-unit |k|
        P_s_3d  = self._broadcast_Ps(P_s, k_rfft)   # (N, N, N//2+1)

        # Colour white noise: s_k = rfftn(w) * sqrt(P_s) → E[|s_k|²] = P_s
        w   = np.random.normal(0.0, 1.0, self.shape)
        s_k = Us(w, transpose=True) * np.sqrt(P_s_3d)
        s_k[0, 0, 0] = 0.0                           # zero DC mode
        return Us(s_k, transpose=False)              # irfftn → real space

    def _draw_foreground(self):
        """
        Draw a foreground realisation.

        Samples C_f ~ IW(a_true.T @ a_true, n_pix), then draws per-pixel
        amplitudes a(x) ~ N(0, C_f), then applies U_f to get the frequency map.

        Returns
        -------
        fg_cube : np.ndarray, shape self.shape
        """
        n_pix = self.shape[0] * self.shape[1]

        # C_f ~ IW(Psi = a_centred.T @ a_centred, nu = n_pix)
        # Mean-subtract before passing to the IW sampler (Gibbs convention)
        a_centred = self.a_true - self.a_true.mean(axis=0)
        C_f = foreground_covariance_sampler(a_centred)     # (n_modes, n_modes)

        # a(x) ~ N(0, C_f) independently per pixel
        a_new = np.random.multivariate_normal(
            mean=np.zeros(self.n_modes), cov=C_f, size=n_pix
        )                                                   # (n_pix, n_modes)

        # U_f @ a: (n_pix, n_modes) @ (n_modes, n_freq) → flatten → reshape
        fg_flat = Uf(self.U_f, a_new, transpose=False)     # (n_pix * n_freq,)
        return fg_flat.reshape(self.shape)

    def _draw_noise(self):
        """
        Draw i.i.d. Gaussian noise cube: n ~ N(0, σ²I).

        Returns
        -------
        noise_cube : np.ndarray, shape self.shape
        """
        return np.random.normal(0.0, self.noise_std, self.shape)

    def _form_data(self, signal, foreground, noise):
        """
        d = signal + foreground + noise

        Returns
        -------
        d : np.ndarray, shape self.shape
        """
        return signal + foreground + noise

    # ------------------------------------------------------------------
    # Signal generation helpers
    # ------------------------------------------------------------------

    def _rfft_kmag(self):
        """
        Pixel-unit |k| array for the rfft output of a cubic shape[0]^3 cube.
        Returns shape (N, N, N//2+1).  Consistent with k_vecs() convention.
        """
        N  = self.shape[0]
        NN = (N * np.fft.fftfreq(N, 1.0)).astype(int)   # integer mode indices
        KX = np.zeros((N, N, N // 2 + 1))
        KY = np.zeros((N, N, N // 2 + 1))
        for i in NN:
            KX[i, :, :] = i
            KY[:, i, :] = i
        KZ = np.arange(N // 2 + 1, dtype=float)[None, None, :]
        return np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    def _broadcast_Ps(self, P_s, k_rfft):
        """
        Map per-bin P_s values onto the rfft k-grid using the same linear
        k-bins that binner() produces from k_vecs().

        Parameters
        ----------
        P_s    : (n_kbins,) linear power values
        k_rfft : (N, N, N//2+1) pixel-unit k magnitudes

        Returns
        -------
        P_s_3d : (N, N, N//2+1)
        """
        k_all  = k_vecs(self.shape[0])                      # full FFT k-magnitudes
        k_min, k_max = k_all.min(), k_all.max()
        kbins  = np.linspace(k_min, k_max, self.n_kbins + 1)

        P_s_3d = np.zeros_like(k_rfft)
        for b in range(self.n_kbins):
            mask = (k_rfft >= kbins[b]) & (k_rfft < kbins[b + 1])
            P_s_3d[mask] = P_s[b]
        P_s_3d[k_rfft >= kbins[-1]] = P_s[-1]  # catch any corner modes beyond kmax
        return P_s_3d
