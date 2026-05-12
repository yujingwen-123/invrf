from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict


def _load_tomllib_module():
    module = importlib.util.find_spec("tomllib")
    if module is not None:
        return importlib.import_module("tomllib")
    fallback = importlib.util.find_spec("tomli")
    if fallback is None:
        raise ModuleNotFoundError(
            "Neither 'tomllib' (Python 3.11+) nor 'tomli' is available. "
            "Install tomli when using Python < 3.11."
        )
    return importlib.import_module("tomli")


tomllib = _load_tomllib_module()


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_config(path: str) -> Dict:
    cfg_path = _expand_path(path)
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    station_cfg = cfg.get("station")
    if not isinstance(station_cfg, dict):
        raise KeyError("Missing required [station] section in config TOML.")

    cfg["_config_path"] = str(cfg_path)
    for key in ("finallist", "rf_dir", "out_dir"):
        if key in station_cfg:
            station_cfg[key] = str(_expand_path(station_cfg[key]))
    return cfg


def ensure_output_dirs(cfg: Dict) -> Dict[str, Path]:
    root = _expand_path(cfg["station"]["out_dir"])
    out = {
        "root": root,
        "csv": root / "csv",
        "figures": root / "figures",
        "pbins": root / "pbins",
        "meta": root / "meta",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out
