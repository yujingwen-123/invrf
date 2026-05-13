#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.model import get_parameter_names


DEFAULT_PRIORITY = [
    "H_sed", "sed_ratio1", "Vs_sed1", "Vs_sed2", "VpVs_sed",
    "H_uc", "Vs_uc", "VpVs_uc", "H_moho", "Vs_lc", "VpVs_lc",
    "Vs_mantle", "VpVs_mantle",
]


def _auto_find_file(root: Path, candidates: Sequence[str]) -> Path | None:
    for name in candidates:
        p = root / "csv" / name
        if p.exists():
            return p
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def _load_samples(samples_path: Path) -> pd.DataFrame:
    if samples_path.suffix.lower() == ".csv":
        df = pd.read_csv(samples_path)
    elif samples_path.suffix.lower() == ".npy":
        arr = np.load(samples_path)
        if arr.ndim != 2:
            raise ValueError(f"{samples_path} must contain a 2D array")
        df = pd.DataFrame(arr)
    else:
        raise ValueError(f"Unsupported sample file type: {samples_path.suffix}")

    # keep only numeric columns and drop obvious non-parameter cols
    num = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in num.columns if c.lower() in {"misfit", "objective", "log_ppd"}]
    if drop_cols:
        num = num.drop(columns=drop_cols)
    if num.shape[1] == 0:
        raise ValueError(f"No numeric parameter columns found in {samples_path}")
    return num


def _load_best_from_posterior(samples_path: Path, params: Sequence[str]) -> np.ndarray | None:
    if not samples_path.exists():
        return None
    df = pd.read_csv(samples_path)
    if "misfit" in df.columns:
        row = df.loc[df["misfit"].idxmin()]
    else:
        row = df.iloc[0]
    values = []
    for p in params:
        if p not in row.index:
            return None
        values.append(float(row[p]))
    return np.asarray(values, dtype=float)


def _load_summary_means(summary_path: Path | None, params: Sequence[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if summary_path is None or not summary_path.exists():
        return None, None
    df = pd.read_csv(summary_path)
    if not {"parameter"}.issubset(df.columns):
        return None, None

    mean_col = None
    std_col = None
    for c in ["posterior_mean", "top_mean", "mean"]:
        if c in df.columns:
            mean_col = c
            break
    for c in ["posterior_std", "top_std", "std"]:
        if c in df.columns:
            std_col = c
            break

    if mean_col is None:
        return None, None

    mapping = df.set_index("parameter")
    means = []
    stds = []
    for p in params:
        if p not in mapping.index:
            return None, None
        means.append(float(mapping.loc[p, mean_col]))
        stds.append(float(mapping.loc[p, std_col]) if std_col is not None else np.nan)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _choose_params(df: pd.DataFrame, params_arg: Sequence[str] | None, max_params: int | None) -> list[str]:
    cols = list(df.columns)

    if params_arg:
        missing = [p for p in params_arg if p not in cols]
        if missing:
            raise ValueError(f"Parameters not found in samples: {missing}")
        selected = list(params_arg)
    else:
        preferred = [p for p in DEFAULT_PRIORITY if p in cols]
        remaining = [c for c in cols if c not in preferred]
        selected = preferred + remaining

    if max_params is not None:
        selected = selected[:max_params]
    return selected


def _ranges_from_config(cfg: dict, params: Sequence[str]) -> list[tuple[float, float]]:
    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    bounds = cfg.get("bounds_classic_nainvrf", {}) if mode == "classic_nainvrf" else cfg.get("bounds", {})
    ranges: list[tuple[float, float]] = []
    for p in params:
        if p in bounds and len(bounds[p]) >= 2:
            lo, hi = float(bounds[p][0]), float(bounds[p][1])
            if lo == hi:
                lo -= 0.5; hi += 0.5
            ranges.append((lo, hi))
        else:
            raise KeyError(f"Missing bounds for parameter '{p}' in config")
    return ranges


def plot_appraisal_corner(
    samples_df: pd.DataFrame,
    params: Sequence[str],
    best: np.ndarray | None,
    posterior_mean: np.ndarray | None,
    posterior_median: np.ndarray | None,
    outpath: Path,
    bins: int = 40,
    max_scatter_points: int = 12000,
    scatter_alpha: float = 0.05,
    scatter_size: float = 2.0,
    figsize_per_dim: float = 1.5,
    hist_lw: float = 1.0,
    title: str | None = None,
    ranges: Sequence[tuple[float, float]] | None = None,
) -> None:
    arr = samples_df.loc[:, params].to_numpy(dtype=float)

    # finite rows only
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        raise RuntimeError("No finite appraisal samples remain after filtering.")

    if arr.shape[0] > max_scatter_points:
        rng = np.random.default_rng(20260422)
        idx = rng.choice(arr.shape[0], size=max_scatter_points, replace=False)
        arr_plot = arr[idx]
    else:
        arr_plot = arr

    n = len(params)
    if ranges is None:
        raise ValueError("ranges must be provided from config bounds.")
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_dim * n, figsize_per_dim * n), squeeze=False)

    # style
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                x = arr[:, j]
                ax.hist(x, bins=bins, histtype="step", color="k", linewidth=hist_lw)

                if best is not None:
                    ax.axvline(best[j], color="blue", ls="--", lw=1.2)
                if posterior_mean is not None:
                    ax.axvline(posterior_mean[j], color="red", ls="--", lw=1.2)
                if posterior_median is not None:
                    ax.axvline(posterior_median[j], color="green", ls="--", lw=1.2)

                ax.set_xlim(ranges[j])
                ax.set_yticks([])
            else:
                x = arr_plot[:, j]
                y = arr_plot[:, i]
                ax.scatter(x, y, s=scatter_size, c="k", alpha=scatter_alpha, linewidths=0, rasterized=True)

                if best is not None:
                    ax.plot(best[j], best[i], marker="o", ms=3.2, color="blue", linestyle="None")
                if posterior_mean is not None:
                    ax.plot(posterior_mean[j], posterior_mean[i], marker="o", ms=3.2, color="red", linestyle="None")
                if posterior_median is not None:
                    ax.plot(posterior_median[j], posterior_median[i], marker="o", ms=3.2, color="green", linestyle="None")

                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])

            if i == n - 1:
                ax.set_xlabel(params[j], fontsize=9)
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(params[i], fontsize=9)
            else:
                if i != j:
                    ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=8, length=2.5, pad=1.5)

    # compact legend using invisible axis
    handles = []
    labels = []
    if best is not None:
        handles.append(plt.Line2D([0], [0], color="blue", ls="--", marker="o", ms=5, lw=1.2))
        labels.append("Best model")
    if posterior_mean is not None:
        handles.append(plt.Line2D([0], [0], color="red", ls="--", marker="o", ms=5, lw=1.2))
        labels.append("Posterior mean")
    if posterior_median is not None:
        handles.append(plt.Line2D([0], [0], color="green", ls="--", marker="o", ms=5, lw=1.2))
        labels.append("Posterior median")

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=True, bbox_to_anchor=(0.5, 0.995))

    if title:
        fig.suptitle(title, fontsize=12, y=0.998)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.06, top=0.93, wspace=0.03, hspace=0.03)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corner-style appraisal parameter plot for the current RF joint inversion project."
    )
    parser.add_argument(
        "result_dir",
        help="Project output root directory (the one containing csv/ and figures/)."
    )
    parser.add_argument("--config", required=True, help="Path to station TOML config used for fixed parameter bounds.")
    parser.add_argument(
        "--samples",
        default=None,
        help="Override appraisal sample file. Defaults to csv/naii_samples_valid.csv or csv/naii_samples.csv."
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Override appraisal summary file. Defaults to csv/appraisal_summary.csv."
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Parameter names to plot, in order. If omitted, auto-detect from samples."
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=None,
        help="Limit number of parameters plotted from the auto-detected list."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins on diagonal."
    )
    parser.add_argument(
        "--max-scatter",
        type=int,
        default=12000,
        help="Maximum number of scatter points after downsampling."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to figures/posterior_corner.png under result_dir."
    )
    args = parser.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    samples_path = Path(args.samples).expanduser().resolve() if args.samples else _auto_find_file(
        root, ["naii_samples_valid.csv", "naii_samples.csv", "appraisal_samples_valid.csv", "appraisal_samples.csv"]
    )
    if samples_path is None:
        raise FileNotFoundError("Could not find appraisal sample file under result_dir/csv/")

    summary_path = Path(args.summary).expanduser().resolve() if args.summary else _auto_find_file(
        root, ["appraisal_summary.csv", "posterior_summary.csv"]
    )

    cfg = load_config(args.config)
    samples_df = _load_samples(samples_path)
    params = _choose_params(samples_df, args.params or get_parameter_names(cfg), args.max_params)
    ranges = _ranges_from_config(cfg, params)

    posterior_mean, _ = _load_summary_means(summary_path, params)
    if posterior_mean is None:
        posterior_mean = samples_df.loc[:, params].mean().to_numpy(dtype=float)

    posterior_median = samples_df.loc[:, params].median().to_numpy(dtype=float)
    best = _load_best_from_posterior(samples_path, params)

    outpath = Path(args.out).expanduser().resolve() if args.out else (root / "figures" / "posterior_corner.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plot_appraisal_corner(
        samples_df=samples_df,
        params=params,
        best=best,
        posterior_mean=posterior_mean,
        posterior_median=posterior_median,
        outpath=outpath,
        bins=args.bins,
        max_scatter_points=args.max_scatter,
        title="Posterior appraisal parameter corner plot",
        ranges=ranges,
    )

    print(f"[INFO] Samples file : {samples_path}")
    print(f"[INFO] Summary file : {summary_path if summary_path else 'None'}")
    print(f"[INFO] Params       : {params}")
    print(f"[INFO] Output       : {outpath}")


if __name__ == "__main__":
    main()
