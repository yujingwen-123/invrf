from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import pandas as pd

from .forward import synthetic_rf_for_rayp
from .model import get_parameter_names, model_to_depth_grid, params_to_dict, params_to_geometry, row_to_params
from .pbin import PBin

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def save_station_metadata(path: Path, cfg: Dict, n_loaded: int, n_missing: int, n_bins: int) -> None:
    payload = {
        "station": cfg["station"]["name"],
        "config": cfg.get("_config_path"),
        "n_loaded": int(n_loaded),
        "n_missing": int(n_missing),
        "n_bins": int(n_bins),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pbin_table(bins: List[PBin], path: Path) -> None:
    rows = []
    for b in bins:
        rows.append(
            {
                "p_left": b.p_left,
                "p_right": b.p_right,
                "p_center": b.p_center,
                "n_events": b.n_events,
                "event_ids": ",".join(b.event_ids),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_pbin_waveforms(bins: List[PBin], time: np.ndarray, path: Path) -> None:
    payload = {"time": time}
    for i, b in enumerate(bins):
        payload[f"stack_{i}"] = b.stack
        payload[f"std_{i}"] = b.std
        payload[f"p_{i}"] = np.array([b.p_center])
    np.savez(path, **payload)


def save_search_results(search_df: pd.DataFrame, path: Path) -> None:
    search_df.to_csv(path, index=False)


def save_appraisal_samples(samples: np.ndarray, path: Path, cfg: Dict) -> None:
    pd.DataFrame(samples, columns=get_parameter_names(cfg)).to_csv(path, index=False)


def summarize_search(search_df: pd.DataFrame, cfg: Dict, top_fraction: float = 0.10) -> pd.DataFrame:
    n_top = max(5, int(np.ceil(len(search_df) * top_fraction)))
    top = search_df.nsmallest(n_top, "misfit")
    rows = []
    for name in get_parameter_names(cfg):
        rows.append(
            {
                "parameter": name,
                "best": float(search_df.iloc[0][name]),
                "top_mean": float(top[name].mean()),
                "top_std": float(top[name].std(ddof=1)) if len(top) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_appraisal(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, cfg: Dict) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov)) if np.ndim(cov) == 2 else np.full(len(mean), np.nan)
    return pd.DataFrame({"parameter": get_parameter_names(cfg), "posterior_mean": mean, "posterior_std": std})


def plot_pbin_stacks(bins: List[PBin], time: np.ndarray, out_png: Path) -> None:
    if not bins:
        return
    fig_h = max(4.0, 1.25 * len(bins))
    fig, axes = plt.subplots(len(bins), 1, figsize=(9, fig_h), sharex=True)
    if len(bins) == 1:
        axes = [axes]
    for ax, b in zip(axes, bins):
        ax.plot(time, b.stack, color="black", lw=1.6)
        if np.any(np.isfinite(b.std)):
            ax.fill_between(time, b.stack - b.std, b.stack + b.std, color="0.7", alpha=0.22)
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.text(0.01, 0.93, f"p={b.p_center:.4f}, N={b.n_events}", transform=ax.transAxes, ha="left", va="top")
    axes[-1].set_xlabel("Time (s)")
    axes[len(axes) // 2].set_ylabel("Amplitude")
    fig.suptitle("Observed p-bin RF stacks", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_best_fit_synthetics(best_params: np.ndarray, bins: List[PBin], time: np.ndarray, cfg: Dict) -> List[np.ndarray]:
    return [synthetic_rf_for_rayp(best_params, b.p_center, time, cfg) for b in bins]


def plot_best_fit_bins(bins: List[PBin], syns: List[np.ndarray], time: np.ndarray, out_png: Path) -> None:
    if not bins:
        return
    fig_h = max(4.0, 1.25 * len(bins))
    fig, axes = plt.subplots(len(bins), 1, figsize=(10, fig_h), sharex=True)
    if len(bins) == 1:
        axes = [axes]
    for ax, b, syn in zip(axes, bins, syns):
        ax.plot(time, b.stack, color="black", lw=1.5, label="Observed")
        if np.any(np.isfinite(b.std)):
            ax.fill_between(time, b.stack - b.std, b.stack + b.std, color="0.8", alpha=0.25)
        ax.plot(time, syn, color="crimson", lw=1.4, ls="--", label="Synthetic")
        ax.axvline(0.0, color="0.65", lw=0.8, ls="--")
        ax.text(0.01, 0.93, f"p={b.p_center:.4f}, N={b.n_events}", transform=ax.transAxes, ha="left", va="top")
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Time (s)")
    axes[len(axes) // 2].set_ylabel("Amplitude")
    fig.suptitle("Best-fit joint inversion: per-bin RF comparison", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _tradeoff_pairs(cfg: Dict | None = None) -> list[tuple[str, str]]:
    mode = str((cfg or {}).get("model", {}).get("parameterization", "legacy")).lower()
    if mode == "classic_nainvrf":
        return [
            ("H_sed", "K_sed"),
            ("H_c1", "VpVs_c1"),
            ("Vs_c3", "Vs_mantle"),
            ("VpVs_c3", "VpVs_mantle"),
        ]
    return [
        ("H_sed", "VpVs_sed"),
        ("H_moho", "VpVs_uc"),
        ("Vs_lc", "Vs_mantle"),
        ("VpVs_lc", "VpVs_mantle"),
    ]


def plot_search_tradeoffs(search_results: pd.DataFrame, out_png: Path, top_fraction: float = 0.10, cfg: Dict | None = None) -> None:
    pairs = _tradeoff_pairs(cfg)
    n_top = max(10, int(np.ceil(len(search_results) * top_fraction)))
    top = search_results.nsmallest(n_top, "misfit")
    best = search_results.iloc[0]
    fig, axes = plt.subplots(1, len(pairs), figsize=(15.5, 3.8))
    for ax, (xcol, ycol) in zip(axes, pairs):
        ax.scatter(search_results[xcol], search_results[ycol], s=8, c="0.82", alpha=0.55, edgecolors="none")
        ax.scatter(top[xcol], top[ycol], s=12, c="black", alpha=0.65, edgecolors="none")
        ax.scatter([best[xcol]], [best[ycol]], s=65, facecolors="none", edgecolors="crimson", linewidths=1.4)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
    fig.suptitle("Search-stage trade-offs (diagnostic only)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_tradeoffs(search_results: pd.DataFrame, appraisal_samples: np.ndarray, appraisal_mean: np.ndarray, out_png: Path, cfg: Dict | None = None) -> None:
    pairs = _tradeoff_pairs(cfg)
    best = row_to_params(search_results.iloc[0])
    best_d = params_to_dict(best, cfg)
    mean_d = params_to_dict(appraisal_mean, cfg)
    dfp = pd.DataFrame(appraisal_samples, columns=search_results.columns[:-1])
    fig, axes = plt.subplots(1, len(pairs), figsize=(15.5, 3.8))
    for ax, (xcol, ycol) in zip(axes, pairs):
        ax.scatter(dfp[xcol], dfp[ycol], s=5, c="black", alpha=0.08, edgecolors="none")
        ax.scatter([best_d[xcol]], [best_d[ycol]], s=70, facecolors="none", edgecolors="forestgreen", linewidths=1.5, label="Best")
        ax.scatter([mean_d[xcol]], [mean_d[ycol]], s=56, c="crimson", marker="x", linewidths=1.7, label="Posterior mean")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Posterior trade-offs (formal appraisal result)", y=1.06)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _velocity_step_arrays(params: np.ndarray, cfg: Dict, mantle_extra: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    g = params_to_geometry(params, cfg)
    pdict = params_to_dict(params, cfg)
    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    if mode == "classic_nainvrf":
        depths = np.array([
            0.0,
            g.h_sed1,
            g.h_sed1 + g.h_sed2,
            g.h_sed1 + g.h_sed2 + pdict["H_c1"],
            g.h_sed1 + g.h_sed2 + pdict["H_c1"] + pdict["H_c2"],
            g.H_moho,
            g.H_moho + mantle_extra,
        ])
        vs = np.array([
            pdict["Vs_sed1"],
            pdict["Vs_sed2"],
            pdict["Vs_c1"],
            pdict["Vs_c2"],
            pdict["Vs_c3"],
            pdict["Vs_mantle"],
        ])
    else:
        depths = np.array([0.0, g.h_sed1, g.h_sed1 + g.h_sed2, g.h_sed1 + g.h_sed2 + g.h_uc, g.H_moho, g.H_moho + mantle_extra])
        vs = np.array([pdict["Vs_sed1"], pdict["Vs_sed2"], pdict["Vs_uc"], pdict["Vs_lc"], pdict["Vs_mantle"]])
    x = [vs[0], vs[0]]
    y = [depths[0], depths[1]]
    for i in range(1, len(vs)):
        x.extend([vs[i - 1], vs[i], vs[i]])
        y.extend([depths[i], depths[i], depths[i + 1]])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def plot_search_velocity_family(search_results: pd.DataFrame, cfg: Dict, out_png: Path, top_fraction: float = 0.10) -> None:
    n_top = max(10, int(np.ceil(len(search_results) * top_fraction)))
    top = search_results.nsmallest(n_top, "misfit")
    best = row_to_params(search_results.iloc[0])
    fig, ax = plt.subplots(figsize=(5.4, 7.0))
    mantle_extra = float(cfg["model"].get("mantle_extra_depth", 20.0))
    for _, row in top.iterrows():
        x, y = _velocity_step_arrays(row_to_params(row), cfg, mantle_extra)
        ax.plot(x, y, color="0.55", lw=0.8, alpha=0.20, zorder=1)
    x, y = _velocity_step_arrays(best, cfg, mantle_extra)
    ax.plot(x, y, color="forestgreen", lw=2.0, ls="--", zorder=3, label="Best search model")
    ax.set_ylim(float(cfg["model"].get("plot_depth_max", 60.0)), 0.0)
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Search-stage velocity family")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pso_diagnostics(trace: Dict | None, out_png: Path) -> None:
    if not trace or str(trace.get("method", "")).lower() != "pso":
        return
    best = np.asarray(trace.get("best_misfit_history", []), dtype=float)
    mean = np.asarray(trace.get("mean_misfit_history", []), dtype=float)
    if best.size == 0:
        return

    it = np.arange(best.size)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axes[0].plot(it, best, color="crimson", lw=1.8, label="Global best misfit")
    if mean.size == best.size:
        axes[0].plot(it, mean, color="0.25", lw=1.3, ls="--", label="Swarm mean misfit")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Misfit")
    axes[0].set_title("PSO convergence")
    axes[0].legend(frameon=True)

    improve = np.diff(best)
    if improve.size > 0:
        axes[1].plot(np.arange(1, best.size), improve, color="navy", lw=1.4)
        axes[1].axhline(0.0, color="0.4", lw=0.9, ls="--")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Δ(best misfit)")
        axes[1].set_title("PSO per-iteration improvement")
    else:
        axes[1].axis("off")

    fig.suptitle("PSO search diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _centers_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need at least two centers to compute edges.")
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def plot_velocity_posterior_density(
    search_results: pd.DataFrame | None = None,
    appraisal_samples: np.ndarray | None = None,
    appraisal_mean: np.ndarray | None = None,
    cfg: Dict | None = None,
    out_png: Path | None = None,
    n_draw: int = 1500,
    *,
    # backward-compatible aliases
    samples: np.ndarray | None = None,
    best_model: np.ndarray | None = None,
    outpath: Path | None = None,
    **_ignored,
) -> None:
    """Robust posterior Vs density plot.

    Accepts both the current run_station.py call signature and older variants.
    """
    if appraisal_samples is None and samples is not None:
        appraisal_samples = samples
    if out_png is None and outpath is not None:
        out_png = outpath
    if best_model is None and search_results is not None and len(search_results) > 0:
        best_model = row_to_params(search_results.iloc[0])

    if appraisal_samples is None:
        raise ValueError("plot_velocity_posterior_density requires appraisal_samples or samples.")
    if cfg is None:
        raise ValueError("plot_velocity_posterior_density requires cfg.")
    if out_png is None:
        raise ValueError("plot_velocity_posterior_density requires out_png or outpath.")

    zmax = float(cfg["model"].get("plot_depth_max", 60.0))
    dz = float(cfg["model"].get("plot_depth_dz", 0.25))
    n_vs_bins = int(cfg["model"].get("plot_vs_bins", 140))
    z = np.arange(0.0, zmax + 0.5 * dz, dz)

    all_samples = np.asarray(appraisal_samples, dtype=float)
    rng = np.random.default_rng(20260422)
    if len(all_samples) > n_draw:
        draw_idx = rng.choice(len(all_samples), size=n_draw, replace=False)
        draw = all_samples[draw_idx]
    else:
        draw = all_samples

    valid_params = []
    valid_profiles = []
    for s in draw:
        try:
            prof = model_to_depth_grid(s, cfg, z)
            if np.all(np.isfinite(prof)):
                valid_params.append(np.asarray(s, dtype=float))
                valid_profiles.append(np.asarray(prof, dtype=float))
        except Exception:
            continue

    if len(valid_profiles) == 0:
        raise RuntimeError("No valid posterior samples available for velocity density plot.")

    profiles = np.vstack(valid_profiles)
    mean_profile = np.mean(profiles, axis=0)
    median_profile = np.median(profiles, axis=0)

    vs_min = float(np.nanpercentile(profiles, 0.5)) - 0.1
    vs_max = float(np.nanpercentile(profiles, 99.5)) + 0.1
    if not np.isfinite(vs_min) or not np.isfinite(vs_max) or vs_max <= vs_min:
        raise RuntimeError("Invalid Vs range for posterior density plot.")

    vs_edges = np.linspace(vs_min, vs_max, n_vs_bins + 1)
    z_edges = _centers_to_edges(z)

    H = np.zeros((len(z), n_vs_bins), dtype=float)
    for iz in range(len(z)):
        hist, _ = np.histogram(profiles[:, iz], bins=vs_edges)
        H[iz, :] = hist

    row_sum = H.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    H = H / row_sum

    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    pcm = ax.pcolormesh(vs_edges, z_edges, H, shading="auto", cmap="Greys")

    n_cloud = min(800, len(profiles))
    if n_cloud > 0:
        idx = np.linspace(0, len(profiles) - 1, n_cloud, dtype=int)
        for i in idx:
            ax.plot(profiles[i], z, color="k", alpha=0.01, lw=0.5, zorder=2)

    ax.plot(mean_profile, z, color="crimson", lw=1.8, ls="--", label="Posterior mean profile", zorder=3)
    ax.plot(median_profile, z, color="dodgerblue", lw=1.7, label="Posterior median profile", zorder=3)

    if best_model is not None:
        try:
            best_profile = model_to_depth_grid(np.asarray(best_model, dtype=float), cfg, z)
            if np.all(np.isfinite(best_profile)):
                ax.plot(best_profile, z, color="forestgreen", lw=1.8, ls="--", label="Best model", zorder=4)
        except Exception:
            pass

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Relative posterior density")

    ax.set_ylim(zmax, 0.0)
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Posterior velocity density")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
