from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import tomllib  # py311+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_config(path: str) -> Dict:
    cfg_path = _expand_path(path)
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_config_path"] = str(cfg_path)
    for key in ("finallist", "rf_dir", "out_dir"):
        if key in cfg["station"]:
            cfg["station"][key] = str(_expand_path(cfg["station"][key]))
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
