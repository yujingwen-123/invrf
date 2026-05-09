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
            std = np.nanstd(traces, axis=0, ddof=1) if idx.size > 1 else np.zeros(traces.shape[1])
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
