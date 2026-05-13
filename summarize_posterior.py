#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _detect_mode(df: pd.DataFrame) -> str:
    if {"H_sed1", "H_sed2", "H_c1", "H_c2", "H_c3"}.issubset(df.columns):
        return "classic_nainvrf"
    if {"H_sed", "H_uc", "H_moho"}.issubset(df.columns):
        return "legacy"
    raise ValueError("Cannot detect parameterization mode from sample columns.")


def _derived_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    mode = _detect_mode(df)
    out: dict[str, np.ndarray] = {}

    if mode == "classic_nainvrf":
        _require_cols(df, ["H_sed1", "H_sed2", "VpVs_sed1", "VpVs_sed2", "H_c1", "H_c2", "H_c3", "VpVs_c1", "VpVs_c2", "VpVs_c3"])
        h_sed = df["H_sed1"].to_numpy() + df["H_sed2"].to_numpy()
        h_crust_no_sed = df["H_c1"].to_numpy() + df["H_c2"].to_numpy() + df["H_c3"].to_numpy()
        h_crust_total = h_sed + h_crust_no_sed

        k_sed = 0.5 * (df["VpVs_sed1"].to_numpy() + df["VpVs_sed2"].to_numpy())
        k_crust_total = (
            df["VpVs_sed1"].to_numpy() * df["H_sed1"].to_numpy()
            + df["VpVs_sed2"].to_numpy() * df["H_sed2"].to_numpy()
            + df["VpVs_c1"].to_numpy() * df["H_c1"].to_numpy()
            + df["VpVs_c2"].to_numpy() * df["H_c2"].to_numpy()
            + df["VpVs_c3"].to_numpy() * df["H_c3"].to_numpy()
        ) / np.maximum(h_crust_total, 1e-6)
        k_crust_no_sed = (
            df["VpVs_c1"].to_numpy() * df["H_c1"].to_numpy()
            + df["VpVs_c2"].to_numpy() * df["H_c2"].to_numpy()
            + df["VpVs_c3"].to_numpy() * df["H_c3"].to_numpy()
        ) / np.maximum(h_crust_no_sed, 1e-6)
    else:
        _require_cols(df, ["H_sed", "VpVs_sed", "H_uc", "H_moho", "VpVs_uc", "VpVs_lc"])
        h_sed = df["H_sed"].to_numpy()
        h_crust_total = df["H_moho"].to_numpy()
        h_crust_no_sed = np.maximum(df["H_moho"].to_numpy() - df["H_sed"].to_numpy(), 0.0)

        k_sed = df["VpVs_sed"].to_numpy()
        h_uc = df["H_uc"].to_numpy()
        h_lc = np.maximum(df["H_moho"].to_numpy() - df["H_sed"].to_numpy() - df["H_uc"].to_numpy(), 0.0)
        k_crust_total = (
            df["VpVs_sed"].to_numpy() * df["H_sed"].to_numpy()
            + df["VpVs_uc"].to_numpy() * h_uc
            + df["VpVs_lc"].to_numpy() * h_lc
        ) / np.maximum(h_crust_total, 1e-6)
        k_crust_no_sed = (
            df["VpVs_uc"].to_numpy() * h_uc + df["VpVs_lc"].to_numpy() * h_lc
        ) / np.maximum(h_uc + h_lc, 1e-6)

    out["H_sed"] = h_sed
    out["K_sed"] = k_sed
    out["H_crust_total"] = h_crust_total
    out["K_crust_total"] = k_crust_total
    out["H_crust_no_sed"] = h_crust_no_sed
    out["K_crust_no_sed"] = k_crust_no_sed
    return out


def _summary_table(derived: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, arr in derived.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            mean = np.nan
            std = np.nan
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
        rows.append({"parameter": key, "posterior_mean": mean, "posterior_std": std})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize posterior-derived sediment/crust parameters with uncertainties.")
    parser.add_argument("result_dir", help="Result root dir containing csv/")
    parser.add_argument("--samples", default=None, help="Optional sample csv; default auto-finds naii_samples_valid.csv")
    parser.add_argument("--out", default=None, help="Output CSV path; default: <result_dir>/csv/derived_summary.csv")
    args = parser.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    samples_path = Path(args.samples).expanduser().resolve() if args.samples else _auto_find_file(
        root,
        ["naii_samples_valid.csv", "naii_samples.csv", "appraisal_samples_valid.csv", "appraisal_samples.csv"],
    )
    if samples_path is None:
        raise FileNotFoundError("No posterior samples found under result_dir/csv/")

    df = pd.read_csv(samples_path)
    derived = _derived_series(df)
    summary = _summary_table(derived)

    outpath = Path(args.out).expanduser().resolve() if args.out else (root / "csv" / "derived_summary.csv")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outpath, index=False)

    print(f"[INFO] Samples : {samples_path}")
    print(f"[INFO] Output  : {outpath}")


if __name__ == "__main__":
    main()
