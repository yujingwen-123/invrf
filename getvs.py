#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auto_find(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / "csv" / name
        if p.exists():
            return p
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def load_numeric_samples(samples_path: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_path)
    num = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in num.columns if c.lower() in {"misfit", "objective", "log_ppd"}]
    if drop_cols:
        num = num.drop(columns=drop_cols)
    if num.shape[1] == 0:
        raise RuntimeError(f"No numeric parameter columns in {samples_path}")
    return num


def build_profiles(samples_df: pd.DataFrame, cfg, dz: float, zmax_extra: float):
    from src.model import model_to_depth_grid

    zmax = float(cfg["bounds"]["H_moho"][1]) + float(zmax_extra)
    z = np.arange(0.0, zmax + dz, dz)

    valid_profiles = []
    for row in samples_df.to_numpy(dtype=float):
        try:
            prof = model_to_depth_grid(row, cfg, z)
            if np.all(np.isfinite(prof)):
                valid_profiles.append(np.asarray(prof, dtype=float))
        except Exception:
            continue

    if len(valid_profiles) == 0:
        raise RuntimeError("No valid profiles could be generated from the sample CSV.")

    return z, np.vstack(valid_profiles)


def mode_vs_from_density(profiles: np.ndarray, n_vbins: int = 140) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        vs_mode   : most probable Vs at each depth from histogram peak
        vs_mean   : mean Vs at each depth
        vs_median : median Vs at each depth
        vs_std    : std Vs at each depth
    """
    vs_min = float(np.nanmin(profiles))
    vs_max = float(np.nanmax(profiles))
    if not np.isfinite(vs_min) or not np.isfinite(vs_max) or vs_max <= vs_min:
        raise RuntimeError("Invalid Vs range while building density.")

    vs_edges = np.linspace(vs_min, vs_max, n_vbins + 1)
    vs_centers = 0.5 * (vs_edges[:-1] + vs_edges[1:])

    vs_mode = np.empty(profiles.shape[1], dtype=float)
    for i in range(profiles.shape[1]):
        hist, _ = np.histogram(profiles[:, i], bins=vs_edges, density=False)
        k = int(np.argmax(hist))
        vs_mode[i] = vs_centers[k]

    vs_mean = np.mean(profiles, axis=0)
    vs_median = np.median(profiles, axis=0)
    vs_std = np.std(profiles, axis=0)
    return vs_mode, vs_mean, vs_median, vs_std


def main():
    parser = argparse.ArgumentParser(
        description="Save a 2-column depth-Vs TXT from posterior velocity-density results."
    )
    parser.add_argument("result_dir", help="Project output root containing csv/ and figures/")
    parser.add_argument("--config", required=True, help="Project config TOML used for this station")
    parser.add_argument("--samples", default=None,
                        help="Posterior sample CSV. Default: csv/naii_samples_valid.csv or csv/naii_samples.csv")
    parser.add_argument("--dz", type=float, default=0.2, help="Depth grid spacing in km")
    parser.add_argument("--zmax-extra", type=float, default=15.0, help="Extra depth below upper H_moho bound")
    parser.add_argument("--n-vbins", type=int, default=140, help="Number of velocity bins in the density image")
    parser.add_argument("--method", choices=["mode", "mean", "median"], default="mode",
                        help="Which Vs profile to export to the 2-column TXT")
    parser.add_argument("--out", default=None,
                        help="Output TXT path. Default: result_dir/csv/posterior_vs_<method>.txt")
    parser.add_argument("--save-full-csv", action="store_true",
                        help="Also save a CSV with depth, mode, mean, median, std")
    args = parser.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    from src.config import load_config

    cfg = load_config(args.config)

    samples_path = Path(args.samples).expanduser().resolve() if args.samples else auto_find(
        root, ["naii_samples_valid.csv", "naii_samples.csv", "appraisal_samples_valid.csv", "appraisal_samples.csv"]
    )
    if samples_path is None:
        raise FileNotFoundError("Could not find appraisal sample CSV under result_dir/csv/")

    samples_df = load_numeric_samples(samples_path)
    z, profiles = build_profiles(samples_df, cfg, dz=args.dz, zmax_extra=args.zmax_extra)
    vs_mode, vs_mean, vs_median, vs_std = mode_vs_from_density(profiles, n_vbins=args.n_vbins)

    if args.method == "mode":
        vs_out = vs_mode
    elif args.method == "mean":
        vs_out = vs_mean
    else:
        vs_out = vs_median

    out_txt = Path(args.out).expanduser().resolve() if args.out else (root / "csv" / f"posterior_vs_{args.method}.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out_txt,
        np.column_stack([z, vs_out]),
        fmt="%.6f",
        header="depth_km vs_km_s",
        comments=""
    )

    if args.save_full_csv:
        out_csv = out_txt.with_suffix(".csv")
        pd.DataFrame({
            "depth_km": z,
            "vs_mode": vs_mode,
            "vs_mean": vs_mean,
            "vs_median": vs_median,
            "vs_std": vs_std,
        }).to_csv(out_csv, index=False)
        print(f"[INFO] Saved full CSV: {out_csv}")

    print(f"[INFO] Samples CSV : {samples_path}")
    print(f"[INFO] Valid profiles: {profiles.shape[0]}")
    print(f"[INFO] Method     : {args.method}")
    print(f"[INFO] Saved TXT  : {out_txt}")


if __name__ == "__main__":
    main()
