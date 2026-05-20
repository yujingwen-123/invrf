#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.model import get_parameter_names, row_to_params
from src.postprocess import plot_velocity_posterior_density


def _auto_find_file(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / "csv" / name
        if p.exists():
            return p
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def _load_samples(samples_path: Path, cfg: dict) -> np.ndarray:
    df = pd.read_csv(samples_path)
    param_names = get_parameter_names(cfg)
    missing = [c for c in param_names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required sample columns in {samples_path}: {missing}")
    return df[param_names].to_numpy(dtype=float)


def _load_best_model(best_path: Path, cfg: dict) -> np.ndarray:
    df = pd.read_csv(best_path)
    if df.empty:
        raise ValueError(f"Best-model csv is empty: {best_path}")
    row = df.nsmallest(1, "misfit").iloc[0] if "misfit" in df.columns else df.iloc[0]
    arr = row_to_params(row)
    expected = len(get_parameter_names(cfg))
    if len(arr) != expected:
        arr = np.asarray([float(row[name]) for name in get_parameter_names(cfg)], dtype=float)
    return arr


def _load_reference_model(ref_path: Path, cfg: dict) -> np.ndarray:
    df = pd.read_csv(ref_path)
    if df.empty:
        raise ValueError(f"Reference-model csv is empty: {ref_path}")
    row = df.iloc[0]
    param_names = get_parameter_names(cfg)
    missing = [name for name in param_names if name not in row.index]
    if missing:
        raise KeyError(f"Missing required reference-model columns in {ref_path}: {missing}")
    return np.asarray([float(row[name]) for name in param_names], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot posterior Vs PDF from NAII results.")
    parser.add_argument("result_dir", help="Result root dir containing csv/ and figures/")
    parser.add_argument("--config", required=True, help="Path to station config TOML used by inversion")
    parser.add_argument("--samples", default=None, help="Posterior sample csv path (default: auto-find naii_samples_valid.csv)")
    parser.add_argument("--best", default=None, help="Optional best/search csv path for plotting best model")
    parser.add_argument("--ref-model", default=None, help="Optional reference/true-model csv path for plotting real model")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <result_dir>/figures/posterior_velocity_density_from_naii.png)")
    parser.add_argument("--n-draw", type=int, default=1500, help="Max number of posterior samples to draw")
    parser.add_argument(
        "--blocky",
        action="store_true",
        help="Disable sediment/crust control-point interpolation for plotting (force layered/blocky Vs profiles).",
    )
    args = parser.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    cfg = load_config(args.config)

    samples_path = Path(args.samples).expanduser().resolve() if args.samples else _auto_find_file(
        root,
        ["naii_samples_valid.csv", "naii_samples.csv", "appraisal_samples_valid.csv", "appraisal_samples.csv"],
    )
    if samples_path is None:
        raise FileNotFoundError("No posterior sample csv found under result_dir/csv/.")

    samples = _load_samples(samples_path, cfg)

    best_model = None
    if args.best:
        best_model = _load_best_model(Path(args.best).expanduser().resolve(), cfg)
    else:
        best_path = _auto_find_file(root, ["search_results.csv", "naii_search_results.csv"])
        if best_path is not None:
            try:
                best_model = _load_best_model(best_path, cfg)
            except Exception:
                best_model = None
    reference_model = None
    if args.ref_model:
        reference_model = _load_reference_model(Path(args.ref_model).expanduser().resolve(), cfg)

    out_png = Path(args.out).expanduser().resolve() if args.out else (root / "figures" / "posterior_velocity_density_from_naii.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_cfg = copy.deepcopy(cfg)
    if args.blocky:
        plot_cfg.setdefault("model", {})["sed_control_dz_km"] = 0.0
        plot_cfg.setdefault("model", {})["crust_control_dz_km"] = 0.0

    plot_velocity_posterior_density(
        appraisal_samples=samples,
        cfg=plot_cfg,
        out_png=out_png,
        n_draw=max(1, int(args.n_draw)),
        best_model=best_model,
        reference_model=reference_model,
    )

    print(f"[INFO] Samples: {samples_path}")
    print(f"[INFO] Output : {out_png}")
    print(
        "[INFO] Plot mode: blocky (no interpolation)" if args.blocky else "[INFO] Plot mode: interpolated (uses *_control_dz_km)"
    )
    if best_model is not None:
        print("[INFO] Best model overlay: enabled")
    if reference_model is not None:
        print("[INFO] Reference model overlay: enabled")


if __name__ == "__main__":
    main()
