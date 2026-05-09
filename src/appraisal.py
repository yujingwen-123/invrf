from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from typing import Any, Dict, Tuple
import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from .inversion import SearchResult


@dataclass
class AppraisalResult:
    samples: np.ndarray
    mean: np.ndarray
    sample_mean_error: np.ndarray
    covariance: np.ndarray
    sample_covariance_error: np.ndarray
    log_ppd: np.ndarray
    temperature: float
    n_resample: int
    n_walkers: int
    bounds: Tuple[Tuple[float, float], ...]


class MCIntegrals:
    def __init__(self, nd: int, save: bool = True):
        self.nd = int(nd)
        self.save = bool(save)
        self.samples = [] if save else None
        self.count = 0

    def accumulate(self, x):
        if isinstance(x, MCIntegrals):
            self.count += x.count
            if self.samples is not None and x.samples is not None:
                self.samples.extend(x.samples)
            return
        arr = np.asarray(x, dtype=float).copy()
        self.count += 1
        if self.samples is not None:
            self.samples.append(arr)

    def _as_array(self) -> np.ndarray:
        if self.samples is None or len(self.samples) == 0:
            raise RuntimeError("No samples stored in accumulator.")
        return np.vstack(self.samples)

    def mean(self) -> np.ndarray:
        return np.mean(self._as_array(), axis=0)

    def sample_mean_error(self) -> np.ndarray:
        arr = self._as_array()
        if arr.shape[0] < 2:
            return np.full(arr.shape[1], np.nan)
        return np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

    def covariance(self) -> np.ndarray:
        arr = self._as_array()
        if arr.shape[0] < 2:
            return np.full((arr.shape[1], arr.shape[1]), np.nan)
        return np.cov(arr, rowvar=False, ddof=1)

    def sample_covariance_error(self) -> np.ndarray:
        arr = self._as_array()
        n, nd = arr.shape
        if n < 3:
            return np.full((nd, nd), np.nan)
        mu = np.mean(arr, axis=0)
        outer = np.einsum("ni,nj->nij", arr - mu, arr - mu)
        return np.std(outer, axis=0, ddof=1) / np.sqrt(n)


class NAAppraiserLite:
    """
    Standalone NAII-style appraisal using the same manual-input route as the
    uploaded NAAppraiser code: initial_ensemble + log_ppd + bounds.
    """

    def __init__(
        self,
        n_resample: int,
        n_walkers: int = 1,
        initial_ensemble: NDArray | None = None,
        log_ppd: NDArray | None = None,
        bounds: Tuple[Tuple[float, float], ...] | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ):
        if initial_ensemble is None or log_ppd is None or bounds is None:
            raise ValueError("initial_ensemble, log_ppd and bounds must all be provided.")
        self.initial_ensemble = np.asarray(initial_ensemble, dtype=float)
        self.log_ppd = np.asarray(log_ppd, dtype=float)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.nd = len(self.bounds)
        self.lower = np.array([b[0] for b in self.bounds], dtype=float)
        self.upper = np.array([b[1] for b in self.bounds], dtype=float)
        self.Cm = 1.0 / (self.upper - self.lower) ** 2
        self.verbose = bool(verbose)
        self.Ne = len(self.initial_ensemble)
        self.j = max(int(n_walkers), 1)
        self.nr = max(int(n_resample) // self.j, 1)
        ss = np.random.SeedSequence(seed)
        self.rngs = [np.random.default_rng(s) for s in ss.spawn(self.j)]

    def run(self, save: bool = True, start_fraction: float = 0.5) -> None:
        if self.j == 1:
            accumulator = self._run_serial(save)
        else:
            if start_fraction < 0 or start_fraction > 1:
                raise ValueError("start_fraction must be between 0 and 1")
            accumulator = self._run_parallel(save, start_fraction)
        self.mean = accumulator.mean()
        self.sample_mean_error = accumulator.sample_mean_error()
        self.covariance = accumulator.covariance()
        self.sample_covariance_error = accumulator.sample_covariance_error()
        self.samples = np.stack(accumulator.samples) if save and accumulator.samples is not None else None

    def _run_serial(self, save: bool = True) -> MCIntegrals:
        start = int(np.argmax(self.log_ppd))
        accumulator = MCIntegrals(self.nd, save)
        for x in self._random_walk_through_parameter_space(self.rngs[0], start):
            accumulator.accumulate(x)
        return accumulator

    def _run_parallel(self, save: bool = True, start_fraction: float = 0.5) -> MCIntegrals:
        n_jobs = min(self.j, cpu_count() or self.j)
        int_threshold = max(1, int(self.Ne * start_fraction))
        start_pool = np.argpartition(self.log_ppd, -int_threshold)[-int_threshold:]
        start_points = np.random.choice(start_pool, self.j, replace=self.j > len(start_pool))
        start_points[0] = int(np.argmax(self.log_ppd))
        accumulators = [MCIntegrals(self.nd, save) for _ in range(self.j)]
        with Parallel(n_jobs=n_jobs) as parallel:
            accumulators = parallel(
                delayed(self._appraise)(acc, rng, int(start))
                for acc, rng, start in zip(accumulators, self.rngs, start_points)
            )
        accumulator = MCIntegrals(self.nd, save)
        for acc in accumulators:
            accumulator.accumulate(acc)
        return accumulator

    def _appraise(self, accumulator: MCIntegrals, rng: np.random.Generator, start_k: int = 0):
        for x in self._random_walk_through_parameter_space(rng, start_k):
            accumulator.accumulate(x)
        return accumulator

    def _random_walk_through_parameter_space(self, rng: np.random.Generator, start_k: int = 0):
        xA = self.initial_ensemble[start_k].copy()
        for _ in tqdm(range(self.nr), desc="NAII - Random Walk", disable=not self.verbose):
            for i in range(self.nd):
                intersections, cells = self._axis_intersections(i, xA)
                xpi = self._random_step(i, intersections, cells, rng)
                xA[i] = xpi
            yield xA.copy()

    def _axis_intersections(self, axis: int, xA: NDArray) -> Tuple[NDArray, NDArray]:
        d = (xA - self.initial_ensemble) ** 2
        d2 = np.sum(d * self.Cm, axis=1)
        k = int(np.argmin(d2))
        dk2 = np.sum(np.delete(d, axis, 1) * np.delete(self.Cm, axis), axis=1)
        down_intersections, down_cells = self._get_axis_intersections(axis, k, dk2, down=True)
        down_intersections = down_intersections[::-1]
        down_cells = down_cells[::-1]
        up_intersections, up_cells = self._get_axis_intersections(axis, k, dk2, up=True)
        return np.array(down_intersections + up_intersections), np.array(down_cells + [k] + up_cells)

    def _random_step(self, axis: int, intersections: NDArray, cells: NDArray, rng: np.random.Generator) -> float:
        while True:
            xpi = float(rng.uniform(self.lower[axis], self.upper[axis]))
            k = self._identify_cell(xpi, intersections, cells)
            r = float(rng.uniform(0, 1))
            log_pxpi = self.log_ppd[k]
            log_pmax = np.max(self.log_ppd[cells])
            if np.log(r) < log_pxpi - log_pmax:
                return xpi

    def _get_axis_intersections(self, axis: int, k: int, di2: NDArray, down: bool = False, up: bool = False):
        assert not (down and up)
        intersections = []
        cells = []
        vk = self.initial_ensemble[k]
        vki = vk[axis]
        vji = self.initial_ensemble[:, axis]
        a = di2[k] - di2
        b = self.Cm[axis] * (vki - vji)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            xji = 0.5 * (vki + vji + a / b)
        if down:
            mask = (vki <= vji) | ~np.isfinite(xji)
            closest = np.argmax
        else:
            mask = (vki >= vji) | ~np.isfinite(xji)
            closest = np.argmin
        xji = np.ma.array(xji, mask=mask)
        if xji.count() > 0:
            k_new = int(closest(xji))
            intersections += [float(xji[k_new])]
            cells += [k_new]
            new_intersections, new_cells = self._get_axis_intersections(axis, k_new, di2, down, up)
            return intersections + new_intersections, cells + new_cells
        return intersections, cells

    def _identify_cell(self, xp: float, intersections: NDArray, cells: NDArray) -> int:
        closest_intersection = int(np.argmin(np.abs(intersections - xp)))
        cell_id = closest_intersection if xp < intersections[closest_intersection] else closest_intersection + 1
        return int(cells[cell_id])


def compute_log_ppd(misfits: NDArray, cfg: Dict[str, Any]) -> tuple[np.ndarray, float]:
    misfits = np.asarray(misfits, dtype=float)
    base = misfits - np.nanmin(misfits)
    app = cfg["appraisal"]
    mode = str(app.get("temperature_mode", "best_iqr")).lower()
    min_temp = float(app.get("temperature_min", 1.0e-3))
    if mode == "fixed":
        temperature = float(app["temperature"])
    else:
        frac = float(app.get("temperature_top_fraction", 0.10))
        n_top = max(5, int(np.ceil(len(base) * frac)))
        top = np.sort(base)[:n_top]
        q25, q75 = np.percentile(top, [25, 75])
        temperature = max(q75 - q25, min_temp)
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = max(np.std(top), min_temp)
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = max(np.std(base), min_temp)
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = 1.0
    return -base / temperature, float(temperature)


def run_appraisal_from_search(search: SearchResult, cfg: Dict[str, Any]) -> AppraisalResult:
    log_ppd, temperature = compute_log_ppd(search.misfits, cfg)
    bounds = tuple(zip(search.lower_bounds, search.upper_bounds))
    app = cfg["appraisal"]
    walker_count = int(app.get("n_walkers", 8))
    seed = app.get("seed", None)
    if seed is not None:
        seed = int(seed)
    naii = NAAppraiserLite(
        int(app["n_resample"]),
        walker_count,
        search.models,
        log_ppd,
        bounds,
        bool(app.get("verbose", True)),
        seed,
    )
    naii.run(save=True, start_fraction=float(app.get("start_fraction", 0.3)))
    return AppraisalResult(
        np.asarray(naii.samples, dtype=float),
        np.asarray(naii.mean, dtype=float),
        np.asarray(naii.sample_mean_error, dtype=float),
        np.asarray(naii.covariance, dtype=float),
        np.asarray(naii.sample_covariance_error, dtype=float),
        log_ppd,
        float(temperature),
        int(app["n_resample"]),
        walker_count,
        bounds,
    )
