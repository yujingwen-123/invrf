from __future__ import annotations

import numpy as np


def _safe_arrays(obs: np.ndarray, syn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(obs) & np.isfinite(syn)
    o = obs[mask]
    s = syn[mask]
    if o.size == 0:
        raise ValueError("No finite samples available for misfit computation.")
    return o, s


def cc_misfit(obs: np.ndarray, syn: np.ndarray) -> float:
    o, s = _safe_arrays(obs, syn)
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return 1.0
    cc = np.corrcoef(o, s)[0, 1]
    if not np.isfinite(cc):
        return 1.0
    cc = np.clip(cc, -1.0, 1.0)
    return float((1.0 - cc) / 2.0)


def nrmse_misfit(obs: np.ndarray, syn: np.ndarray) -> float:
    o, s = _safe_arrays(obs, syn)
    rmse = np.sqrt(np.mean((o - s) ** 2))
    scale = np.sqrt(np.mean(o**2))
    if scale <= 0 or not np.isfinite(scale):
        scale = np.max(np.abs(o))
    if scale <= 0 or not np.isfinite(scale):
        return float(rmse)
    return float(rmse / scale)


def combined_misfit(obs: np.ndarray, syn: np.ndarray, w_cc: float, w_nrmse: float) -> float:
    return float(w_cc) * cc_misfit(obs, syn) + float(w_nrmse) * nrmse_misfit(obs, syn)


def chi_square_misfit(obs: np.ndarray, syn: np.ndarray, sigma: np.ndarray, sigma_floor: float = 1.0e-3) -> float:
    o, s = _safe_arrays(obs, syn)
    sig = np.asarray(sigma, dtype=float)[np.isfinite(obs) & np.isfinite(syn)]
    sig = np.where(np.isfinite(sig) & (sig > sigma_floor), sig, sigma_floor)
    r = (o - s) / sig
    return float(np.mean(r**2))
