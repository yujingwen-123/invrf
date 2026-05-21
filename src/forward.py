from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from .model import params_to_velocity_model

try:
    from rfsed.synrf import synrf  # type: ignore
except Exception:  # pragma: no cover
    synrf = None


def _baseline_correct(time: np.ndarray, data: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    mask = (time >= tmin) & (time <= tmax)
    if np.any(mask):
        return data - np.nanmean(data[mask])
    return data


def _normalize_like_observed(time: np.ndarray, data: np.ndarray, cfg: Dict) -> np.ndarray:
    data_cfg = cfg["data"]
    mode = str(data_cfg.get("normalize", "none")).lower()
    if mode == "none":
        return data
    mask = (time >= float(data_cfg.get("norm_tmin", -1.0))) & (time <= float(data_cfg.get("norm_tmax", 8.0)))
    if mode == "maxabs":
        scale = np.nanmax(np.abs(data[mask])) if np.any(mask) else np.nanmax(np.abs(data))
    elif mode == "rms":
        scale = np.sqrt(np.nanmean(data[mask] ** 2)) if np.any(mask) else np.sqrt(np.nanmean(data**2))
    else:
        scale = 1.0
    if np.isfinite(scale) and scale > 0:
        return data / scale
    return data


def synthetic_rf_for_rayp(params: np.ndarray, rayp: float, target_time: np.ndarray, cfg: Dict) -> np.ndarray:
    if synrf is None:
        raise ImportError("rfsed.synrf is required for forward modelling but is not available.")
    vm = params_to_velocity_model(params, cfg)
    dt = float(cfg["data"]["dt"])
    time_start = float(target_time[0])
    npts = len(target_time)
    preonset = max(0.0, -time_start)

    try:
        with warnings.catch_warnings():
            # rfsed can emit RuntimeWarning for non-physical combinations
            # (e.g., invalid sqrt/divide in propagator terms). Treat these as
            # invalid forward models and skip them cleanly.
            warnings.simplefilter("error", RuntimeWarning)
            synth = synrf(
                vm.interfaces_km,
                vm.vp_km_s,
                vm.vs_km_s,
                vm.rho_g_cm3,
                float(rayp),
                dt=dt,
                npts=npts,
                ipha=1,
            )
            synth.run_fwd()
            if bool(cfg["data"].get("apply_filter", False)):
                synth.filter(
                    freqmin=float(cfg["data"]["freqmin"]),
                    freqmax=float(cfg["data"]["freqmax"]),
                    order=int(cfg["data"].get("filter_order", 2)),
                    zerophase=bool(cfg["data"].get("zerophase", True)),
                )
            rf = synth.run_deconvolution(
                pre_filt=[float(cfg["data"].get("freqmin", 0.05)), float(cfg["data"].get("freqmax", 1.25))],
                preonset=preonset,
                gaussian=float(cfg["data"].get("gaussian", 1.25)),
            )
    except RuntimeWarning:
        return np.full_like(target_time, np.nan, dtype=float)
    tr = rf[0]
    syn_time = np.arange(tr.stats.npts, dtype=float) * tr.stats.delta - preonset
    syn_data = np.asarray(tr.data, dtype=float)
    interp = np.interp(target_time, syn_time, syn_data, left=np.nan, right=np.nan)
    if bool(cfg["data"].get("baseline_correct", True)):
        interp = _baseline_correct(
            target_time,
            interp,
            float(cfg["data"].get("baseline_tmin", -2.0)),
            float(cfg["data"].get("baseline_tmax", -0.2)),
        )
    interp = _normalize_like_observed(target_time, interp, cfg)
    return interp
