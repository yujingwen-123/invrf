from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config, ensure_output_dirs
from src.finallist import parse_finallist
from src.rf_io import load_station_rfs
from src.pbin import build_pbin_stacks

try:
    from scipy.stats import normaltest
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _normality_pvalue(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.nan
    if HAS_SCIPY:
        return float(normaltest(x, nan_policy="omit").pvalue)
    # Fallback heuristic: not a true p-value.
    m = np.mean(x)
    s = np.std(x)
    if s <= 0:
        return np.nan
    z = (x - m) / s
    skew = np.mean(z**3)
    kurt = np.mean(z**4) - 3.0
    score = abs(skew) + abs(kurt)
    return float(np.exp(-score))


def evaluate_bin_gaussianity(traces: np.ndarray, stack: np.ndarray, time: np.ndarray, p_threshold: float) -> pd.DataFrame:
    # error samples at each time: event_i(t) - stack(t)
    errors = traces - stack[None, :]
    rows = []
    for it, t in enumerate(time):
        e = errors[:, it]
        e = e[np.isfinite(e)]
        if e.size < 8:
            rows.append((it, float(t), int(e.size), np.nan, np.nan, np.nan, np.nan, False))
            continue
        mu = float(np.mean(e))
        sigma = float(np.std(e, ddof=1)) if e.size > 1 else np.nan
        if np.isfinite(sigma) and sigma > 0:
            z = (e - mu) / sigma
            skew = float(np.mean(z**3))
            kurt_excess = float(np.mean(z**4) - 3.0)
        else:
            skew = np.nan
            kurt_excess = np.nan
        p = _normality_pvalue(e)
        is_gaussian = bool(np.isfinite(p) and p >= p_threshold)
        rows.append((it, float(t), int(e.size), mu, sigma, skew, kurt_excess, is_gaussian if np.isfinite(p) else False, p))
    return pd.DataFrame(rows, columns=["time_index", "time_s", "n", "mu", "sigma", "skew", "kurtosis_excess", "is_gaussian", "p_value"])


def plot_bin_result(df: pd.DataFrame, bin_id: int, out_png: Path, p_threshold: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = df["time_s"].to_numpy()

    axes[0].plot(x, df["p_value"], lw=1.2)
    axes[0].axhline(p_threshold, color="r", ls="--", lw=1.0, label=f"p={p_threshold}")
    axes[0].set_ylabel("normality p-value")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, df["skew"], lw=1.2, label="skew")
    axes[1].plot(x, df["kurtosis_excess"], lw=1.2, label="kurtosis excess")
    axes[1].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("shape moments")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    g = df["is_gaussian"].astype(float).to_numpy()
    axes[2].plot(x, g, lw=1.2)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["No", "Yes"])
    axes[2].set_ylabel("Gaussian?")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Bin {bin_id}: error Gaussianity vs time")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether per-bin RF stack errors are Gaussian at each time sample.")
    parser.add_argument("config", help="Path to station TOML")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="Normality p-value threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = ensure_output_dirs(cfg)
    records = parse_finallist(cfg["station"]["finallist"])
    time, loaded, _ = load_station_rfs(records, cfg)
    bins = build_pbin_stacks(loaded, cfg)

    if not bins:
        raise RuntimeError("No usable bins.")

    out_dir = out["figures"] / "bin_gaussianity"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out["csv"]

    summary_rows = []
    for i, b in enumerate(bins):
        traces = np.vstack([loaded[idx].data for idx in b.indices])
        df = evaluate_bin_gaussianity(traces, b.stack, time, args.p_threshold)
        df.to_csv(csv_dir / f"bin_{i:02d}_gaussianity.csv", index=False)
        plot_bin_result(df, i, out_dir / f"bin_{i:02d}_gaussianity.png", args.p_threshold)

        valid = df["p_value"].notna()
        frac = float((df.loc[valid, "p_value"] >= args.p_threshold).mean()) if valid.any() else np.nan
        summary_rows.append({
            "bin_id": i,
            "p_center": b.p_center,
            "n_events": b.n_events,
            "fraction_gaussian_time": frac,
            "n_valid_time_samples": int(valid.sum()),
        })

    pd.DataFrame(summary_rows).to_csv(csv_dir / "bin_gaussianity_summary.csv", index=False)
    print(f"[INFO] Wrote per-bin gaussianity CSVs to: {csv_dir}")
    print(f"[INFO] Wrote per-bin gaussianity plots to: {out_dir}")


if __name__ == "__main__":
    main()
