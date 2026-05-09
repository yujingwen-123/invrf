from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import obspy

from .finallist import EventRecord


@dataclass
class LoadedRF:
    record: EventRecord
    filepath: str
    time: np.ndarray
    data: np.ndarray


def _trace_time_axis(tr: obspy.Trace, time_reference: str, preonset_fallback: float) -> np.ndarray:
    npts = tr.stats.npts
    dt = tr.stats.delta
    sac = getattr(tr.stats, "sac", None)
    tref = None
    if sac is not None:
        for key in [time_reference, "b", "a", "t0"]:
            if key is None:
                continue
            val = getattr(sac, key, None)
            if val is not None:
                try:
                    tref = float(val)
                    break
                except Exception:
                    pass
    if tref is None:
        tref = -float(preonset_fallback)
    return tref + np.arange(npts, dtype=float) * dt


def _baseline_correct(time: np.ndarray, data: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    mask = (time >= tmin) & (time <= tmax)
    if np.any(mask):
        return data - np.nanmean(data[mask])
    return data


def _normalize_trace(time: np.ndarray, data: np.ndarray, mode: str, norm_tmin: float, norm_tmax: float) -> np.ndarray:
    mode = mode.lower()
    if mode == "none":
        return data
    if mode == "maxabs":
        mask = (time >= norm_tmin) & (time <= norm_tmax)
        scale = np.nanmax(np.abs(data[mask])) if np.any(mask) else np.nanmax(np.abs(data))
        if np.isfinite(scale) and scale > 0:
            return data / scale
    elif mode == "rms":
        mask = (time >= norm_tmin) & (time <= norm_tmax)
        scale = np.sqrt(np.nanmean(data[mask] ** 2)) if np.any(mask) else np.sqrt(np.nanmean(data**2))
        if np.isfinite(scale) and scale > 0:
            return data / scale
    return data


def load_rf_for_record(record: EventRecord, cfg: Dict, target_time: np.ndarray) -> LoadedRF | None:
    station = cfg["station"]
    rf_path = station["rf_pattern"].format(rf_dir=station["rf_dir"], event_id=record.event_id)
    filepath = Path(rf_path).expanduser().resolve()
    if not filepath.exists():
        return None

    st = obspy.read(str(filepath))
    tr = st[0].copy()
    data_cfg = cfg["data"]

    if bool(data_cfg.get("remove_mean", False)):
        tr.detrend("demean")
    if bool(data_cfg.get("remove_trend", False)):
        tr.detrend("linear")
    if bool(data_cfg.get("apply_taper", True)):
        tr.taper(max_percentage=0.05, type="cosine")
    if bool(data_cfg.get("apply_filter", False)):
        tr.filter(
            "bandpass",
            freqmin=float(data_cfg["freqmin"]),
            freqmax=float(data_cfg["freqmax"]),
            corners=int(data_cfg.get("filter_order", 2)),
            zerophase=bool(data_cfg.get("zerophase", True)),
        )

    time = _trace_time_axis(tr, str(data_cfg.get("time_reference", "b")), float(data_cfg.get("preonset_fallback", 10.0)))
    interp = np.interp(target_time, time, np.asarray(tr.data, dtype=float), left=np.nan, right=np.nan)
    if np.any(~np.isfinite(interp)):
        return None

    if bool(data_cfg.get("baseline_correct", True)):
        interp = _baseline_correct(
            target_time,
            interp,
            float(data_cfg.get("baseline_tmin", -2.0)),
            float(data_cfg.get("baseline_tmax", -0.2)),
        )

    interp = _normalize_trace(
        target_time,
        interp,
        str(data_cfg.get("normalize", "none")),
        float(data_cfg.get("norm_tmin", -1.0)),
        float(data_cfg.get("norm_tmax", 8.0)),
    )
    return LoadedRF(record=record, filepath=str(filepath), time=target_time, data=interp)


def load_station_rfs(records: Iterable[EventRecord], cfg: Dict) -> Tuple[np.ndarray, List[LoadedRF], List[EventRecord]]:
    dt = float(cfg["data"]["dt"])
    t0 = float(cfg["data"]["time_start"])
    t1 = float(cfg["data"]["time_end"])
    target_time = np.arange(t0, t1 + 0.5 * dt, dt)
    loaded: List[LoadedRF] = []
    missing: List[EventRecord] = []
    for rec in records:
        item = load_rf_for_record(rec, cfg, target_time)
        if item is None:
            missing.append(rec)
        else:
            loaded.append(item)
    return target_time, loaded, missing
