from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .rf_io import LoadedRF


@dataclass
class PBin:
    p_left: float
    p_right: float
    p_center: float
    indices: np.ndarray
    stack: np.ndarray
    std: np.ndarray
    n_events: int
    event_ids: list[str]


def _bootstrap_stack_std(
    traces: np.ndarray,
    stack_method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_events, n_samples = traces.shape
    if n_events <= 1:
        return np.zeros(n_samples, dtype=float)

    boot = np.empty((n_bootstrap, n_samples), dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_events, size=n_events)
        sampled = traces[idx]
        if stack_method == "median":
            boot[i] = np.nanmedian(sampled, axis=0)
        else:
            boot[i] = np.nanmean(sampled, axis=0)

    std = np.nanstd(boot, axis=0, ddof=1)
    std = np.where(np.isfinite(std), std, 0.0)
    return std


def build_pbin_stacks(loaded: List[LoadedRF], cfg: Dict) -> List[PBin]:
    if not loaded:
        return []
    p = np.array([x.record.rayp for x in loaded], dtype=float)
    width = float(cfg["pbin"]["width"])
    step = float(cfg["pbin"]["step"])
    min_events = int(cfg["pbin"].get("min_events", 8))
    stack_method = str(cfg["pbin"].get("stack_method", "mean")).lower()
    pmin = np.min(p)
    pmax = np.max(p)
    n_bootstrap = int(cfg["pbin"].get("bootstrap_n", 400))
    bootstrap_seed = int(cfg["pbin"].get("bootstrap_seed", 20260521))
    rng = np.random.default_rng(bootstrap_seed)

    bins: List[PBin] = []
    left = pmin
    while left <= pmax + 1.0e-12:
        right = left + width
        idx = np.where((p >= left) & (p < right if right < pmax + width else p <= right))[0]
        if idx.size >= min_events:
            traces = np.vstack([loaded[i].data for i in idx])
            if stack_method == "median":
                stack = np.nanmedian(traces, axis=0)
            else:
                stack = np.nanmean(traces, axis=0)
            std = _bootstrap_stack_std(traces, stack_method, n_bootstrap=max(20, n_bootstrap), rng=rng)
            bins.append(
                PBin(
                    p_left=float(left),
                    p_right=float(right),
                    p_center=float(np.nanmedian(p[idx])),
                    indices=idx.copy(),
                    stack=stack,
                    std=std,
                    n_events=int(idx.size),
                    event_ids=[loaded[i].record.event_id for i in idx],
                )
            )
        left += step
    return bins
