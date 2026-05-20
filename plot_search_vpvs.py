#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.model import model_to_depth_grid, params_to_velocity_model


def _auto_find_search_csv(result_dir: Path) -> Path:
    candidates = [
        result_dir / "csv" / "na_search_results.csv",
        result_dir / "search_results.csv",
        result_dir / "csv" / "models.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find search model CSV. Tried: " + ", ".join(str(p) for p in candidates))


def _to_vpvs_depth_grid(params: np.ndarray, cfg: dict, z: np.ndarray) -> np.ndarray:
    vm = params_to_velocity_model(params, cfg)
    bounds = np.concatenate([np.array([0.0]), vm.interfaces_km, np.array([np.inf])])
    vpvs = vm.vp_km_s / vm.vs_km_s
    out = np.empty_like(z, dtype=float)
    for i in range(len(vpvs)):
        mask = (z >= bounds[i]) & (z < bounds[i + 1])
        out[mask] = vpvs[i]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Use search model ensemble to plot Vs+Vp/Vs family with Vp/Vs density.")
    parser.add_argument("config", help="Station config TOML path")
    parser.add_argument("result_dir", help="Inversion result root directory")
    parser.add_argument("--search-csv", default=None, help="Optional explicit search results CSV path")
    parser.add_argument("--kbest", type=int, default=1000, help="Number of best search models used for density (default 1000)")
    parser.add_argument("--depth-max", type=float, default=None, help="Plot max depth (km), defaults to cfg model.plot_depth_max")
    parser.add_argument("--dz", type=float, default=None, help="Depth grid interval (km), defaults to cfg model.plot_depth_dz")
    parser.add_argument("--vpvs-bins", type=int, default=140, help="Number of vp/vs bins")
    parser.add_argument("--out", default=None, help="Output png path; default: <result_dir>/figures/search_vpvs_family.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = Path(args.result_dir).expanduser().resolve()
    search_csv = Path(args.search_csv).expanduser().resolve() if args.search_csv else _auto_find_search_csv(result_dir)
    out_png = Path(args.out).expanduser().resolve() if args.out else (result_dir / "figures" / "search_vpvs_family.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(search_csv)
    if "misfit" not in df.columns:
        raise KeyError(f"search CSV missing 'misfit' column: {search_csv}")

    df = df.sort_values("misfit", ascending=True).reset_index(drop=True)
    kbest = max(1, min(int(args.kbest), len(df)))
    top = df.iloc[:kbest]

    zmax = float(args.depth_max if args.depth_max is not None else cfg["model"].get("plot_depth_max", 70.0))
    dz = float(args.dz if args.dz is not None else cfg["model"].get("plot_depth_dz", 0.25))
    z = np.arange(0.0, zmax + 0.5 * dz, dz)

    vpvs_profiles = []
    vs_profiles = []
    for _, row in top.iterrows():
        params = row.drop(labels=["misfit"]).to_numpy(dtype=float)
        try:
            vpvs_prof = _to_vpvs_depth_grid(params, cfg, z)
            vs_prof = model_to_depth_grid(params, cfg, z)
            if np.all(np.isfinite(vpvs_prof)) and np.all(np.isfinite(vs_prof)):
                vpvs_profiles.append(vpvs_prof)
                vs_profiles.append(vs_prof)
        except Exception:
            continue

    if not vpvs_profiles:
        raise RuntimeError("No valid profiles generated from search models.")

    vpvs_profiles = np.vstack(vpvs_profiles)
    vs_profiles = np.vstack(vs_profiles)

    vpvs_min = float(np.nanpercentile(vpvs_profiles, 0.5)) - 0.05
    vpvs_max = float(np.nanpercentile(vpvs_profiles, 99.5)) + 0.05
    vpvs_edges = np.linspace(vpvs_min, vpvs_max, int(args.vpvs_bins) + 1)

    H = np.zeros((len(z), len(vpvs_edges) - 1), dtype=float)
    for iz in range(len(z)):
        hist, _ = np.histogram(vpvs_profiles[:, iz], bins=vpvs_edges)
        H[iz, :] = hist

    logH = np.log10(np.clip(H, 1.0, None))
    z_edges = np.concatenate([[z[0] - 0.5 * dz], 0.5 * (z[:-1] + z[1:]), [z[-1] + 0.5 * dz]])

    best_row = df.iloc[0]
    best_params = best_row.drop(labels=["misfit"]).to_numpy(dtype=float)
    ref_vs = model_to_depth_grid(best_params, cfg, z)
    ref_vpvs = _to_vpvs_depth_grid(best_params, cfg, z)

    fig, (ax_vs, ax_vpvs) = plt.subplots(1, 2, figsize=(11.5, 7.2), sharey=True, gridspec_kw={"width_ratios": [1, 1]})

    for i in np.linspace(0, len(vs_profiles) - 1, min(400, len(vs_profiles)), dtype=int):
        ax_vs.plot(vs_profiles[i], z, color="#cfd400", lw=0.8, alpha=0.25)
    ax_vs.plot(ref_vs, z, color="red", lw=1.8, label="Reference Vs model")
    ax_vs.set_xlabel("S VELOCITY (km/s)")
    ax_vs.set_ylabel("DEPTH (km)")
    ax_vs.set_ylim(zmax, 0.0)
    ax_vs.legend(loc="lower left", frameon=True)

    pcm = ax_vpvs.pcolormesh(vpvs_edges, z_edges, logH, shading="auto", cmap="summer")
    ax_vpvs.plot(ref_vpvs, z, color="red", lw=1.8, label="Reference Vp/Vs model")
    ax_vpvs.set_xlabel("Vp/Vs RATIO")
    ax_vpvs.legend(loc="lower left", frameon=True)

    cbar = fig.colorbar(pcm, ax=ax_vpvs, pad=0.02, fraction=0.05)
    cbar.set_label(f"log10(count), top {kbest} search models")

    fig.suptitle(f"MODEL: {cfg['station']['name']}   Misfit = {float(best_row['misfit']):.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] search_csv = {search_csv}")
    print(f"[INFO] kbest      = {kbest}")
    print(f"[INFO] output     = {out_png}")


if __name__ == "__main__":
    main()
