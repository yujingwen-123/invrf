from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .forward import synthetic_rf_for_rayp
from .misfit import chi_square_misfit
from .model import PARAMETER_NAMES, parameter_bounds, params_to_dict, params_to_geometry, validate_params
from .pbin import PBin


_GLOBAL_OBJECTIVE = None


def _worker_init(obj):
    global _GLOBAL_OBJECTIVE
    _GLOBAL_OBJECTIVE = obj


def _worker_eval(params: np.ndarray) -> tuple[float, list[np.ndarray]]:
    global _GLOBAL_OBJECTIVE
    return _GLOBAL_OBJECTIVE.evaluate_single(params)


@dataclass
class SearchResult:
    results: pd.DataFrame
    models: np.ndarray
    misfits: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def _crust_travel_time_avg_vpvs(params: np.ndarray) -> float:
    pd = params_to_dict(params)
    geom = params_to_geometry(params, cfg={})
    h = np.array([geom.h_sed1, geom.h_sed2, geom.h_uc, geom.h_lc], dtype=float)
    vs = np.array([pd["Vs_sed1"], pd["Vs_sed2"], pd["Vs_uc"], pd["Vs_lc"]], dtype=float)
    vpvs = np.array([pd["VpVs_sed"], pd["VpVs_sed"], pd["VpVs_uc"], pd["VpVs_lc"]], dtype=float)
    vp = vs * vpvs
    # travel-time-average crustal Vp/Vs = Ts/Tp
    tp = np.sum(h / vp)
    ts = np.sum(h / vs)
    if tp <= 0 or ts <= 0 or (not np.isfinite(tp)) or (not np.isfinite(ts)):
        return float("nan")
    return float(ts / tp)


class JointRFObjective:
    def __init__(self, bins: List[PBin], cfg: Dict, time: np.ndarray):
        self.bins = bins
        self.cfg = cfg
        self.time = time
        self.parallel_enabled = False
        self.n_workers = 1
        self.chunksize = 1
        self.start_method = str(cfg["parallel"].get("start_method", "spawn"))
        self._pool = None

    def setup_parallel(self) -> None:
        pcfg = self.cfg.get("parallel", {})
        enabled = bool(pcfg.get("enabled", True))
        if not enabled:
            self.parallel_enabled = False
            return
        n_workers = int(pcfg.get("n_workers", 0))
        if n_workers <= 0:
            n_workers = max((mp.cpu_count() or 2) - 1, 1)
        self.n_workers = max(1, n_workers)
        self.chunksize = max(1, int(pcfg.get("chunksize", 4)))
        ctx = mp.get_context(self.start_method)
        maxtasks = int(pcfg.get("maxtasksperchild", 0)) or None
        self._pool = ctx.Pool(processes=self.n_workers, initializer=_worker_init, initargs=(self,), maxtasksperchild=maxtasks)
        self.parallel_enabled = True

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self.parallel_enabled = False

    def prior_penalty(self, params: np.ndarray) -> float:
        pri = self.cfg["priors"]
        pd = params_to_dict(params)
        penalty = 0.0
        # Gaussian-style weak priors for sediment, whole crust, and mantle.
        if "H_sed_center" in pri and "H_sed_sigma" in pri and float(pri["H_sed_sigma"]) > 0:
            z = (pd["H_sed"] - float(pri["H_sed_center"])) / float(pri["H_sed_sigma"])
            penalty += z**2

        if "K_sed_center" in pri and "K_sed_sigma" in pri and float(pri["K_sed_sigma"]) > 0:
            z = (pd["VpVs_sed"] - float(pri["K_sed_center"])) / float(pri["K_sed_sigma"])
            penalty += z**2

        # Whole-crust prior: travel-time-average Vp/Vs across sediments + crust to Moho.
        if "H_crust_center" in pri and "H_crust_sigma" in pri and float(pri["H_crust_sigma"]) > 0:
            z = (pd["H_moho"] - float(pri["H_crust_center"])) / float(pri["H_crust_sigma"])
            penalty += z**2

        if "K_crust_center" in pri and "K_crust_sigma" in pri and float(pri["K_crust_sigma"]) > 0:
            k_crust = _crust_travel_time_avg_vpvs(params)
            if np.isfinite(k_crust):
                z = (k_crust - float(pri["K_crust_center"])) / float(pri["K_crust_sigma"])
                penalty += z**2

        # Mantle constraints.
        for name, center_key, sigma_key in [
            ("Vs_mantle", "Vs_mantle_center", "Vs_mantle_sigma"),
            ("VpVs_mantle", "VpVs_mantle_center", "VpVs_mantle_sigma"),
        ]:
            if center_key in pri and sigma_key in pri and float(pri[sigma_key]) > 0:
                z = (pd[name] - float(pri[center_key])) / float(pri[sigma_key])
                penalty += z**2

        return float(self.cfg["weights"].get("prior", 1.0)) * penalty

    def evaluate_single(self, params: np.ndarray) -> tuple[float, list[np.ndarray]]:
        valid, hard_penalty = validate_params(params, self.cfg)
        if not valid:
            return float(hard_penalty), []
        total = self.prior_penalty(params) + float(hard_penalty)
        synthetics: list[np.ndarray] = []
        try:
            for b in self.bins:
                syn = synthetic_rf_for_rayp(params, b.p_center, self.time, self.cfg)
                synthetics.append(syn)
                total += chi_square_misfit(b.stack, syn, b.std) * float(b.n_events)
        except Exception:
            return 1.0e9, []
        return float(total), synthetics

    def evaluate_many(self, param_matrix: np.ndarray) -> np.ndarray:
        if self.parallel_enabled and self._pool is not None:
            vals = self._pool.map(_worker_eval, [np.asarray(p, dtype=float) for p in param_matrix], chunksize=self.chunksize)
            return np.array([v[0] for v in vals], dtype=float)
        return np.array([self.evaluate_single(np.asarray(p, dtype=float))[0] for p in param_matrix], dtype=float)


def _propose_uniform(n: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lower, upper, size=(n, len(lower)))


def _propose_local(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    nd = len(lower)
    elite_spread = np.std(parents, axis=0, ddof=1) if len(parents) > 1 else (upper - lower) * 0.1
    elite_spread = np.where(np.isfinite(elite_spread) & (elite_spread > 0), elite_spread, (upper - lower) * 0.1)
    scale = np.maximum(elite_spread * 0.7, (upper - lower) * 0.03)
    idx = rng.integers(0, len(parents), size=n)
    base = parents[idx]
    props = base + rng.normal(0.0, scale, size=(n, nd))
    return np.clip(props, lower, upper)


def run_na(cfg: Dict, objective: JointRFObjective) -> SearchResult:
    lower, upper = parameter_bounds(cfg)
    n_initial = int(cfg["na"].get("n_initial", 5000))
    n_samples = int(cfg["na"].get("n_samples", 150))
    n_resample = int(cfg["na"].get("n_resample", 40))
    n_iterations = int(cfg["na"].get("n_iterations", 30))
    seed = int(cfg["na"].get("seed", 20260422))
    rng = np.random.default_rng(seed)

    models = _propose_uniform(n_initial, lower, upper, rng)
    misfits = objective.evaluate_many(models)

    for _ in range(n_iterations):
        order = np.argsort(misfits)
        elite_n = max(5, min(n_resample, len(order)))
        parents = models[order[:elite_n]]
        proposal = _propose_local(parents, lower, upper, n_samples, rng)
        prop_misfit = objective.evaluate_many(proposal)
        models = np.vstack([models, proposal])
        misfits = np.concatenate([misfits, prop_misfit])

    order = np.argsort(misfits)
    models = models[order]
    misfits = misfits[order]
    df = pd.DataFrame(models, columns=PARAMETER_NAMES)
    df["misfit"] = misfits
    return SearchResult(results=df, models=models, misfits=misfits, lower_bounds=lower, upper_bounds=upper)
