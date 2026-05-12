from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import multiprocessing as mp

import numpy as np

from src.appraisal import run_appraisal_from_search
from src.config import ensure_output_dirs, load_config
from src.finallist import parse_finallist
from src.inversion import JointRFObjective, run_na
from src.model import get_parameter_names, params_to_velocity_model, row_to_params
from src.pbin import build_pbin_stacks
"""
from src.postprocess import (
    build_best_fit_synthetics,
    plot_best_fit_bins,
    plot_pbin_stacks,
    plot_posterior_tradeoffs,
    plot_search_tradeoffs,
    plot_search_velocity_family,
    plot_velocity_posterior_density,
    save_appraisal_samples,
    save_pbin_table,
    save_pbin_waveforms,
    save_search_results,
    save_station_metadata,
    summarize_appraisal,
    summarize_search,
)
"""
from src.rf_io import load_station_rfs


def filter_valid_samples(samples: np.ndarray, cfg: dict) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    valid = []
    for s in samples:
        try:
            params_to_velocity_model(np.asarray(s, dtype=float), cfg)
            valid.append(np.asarray(s, dtype=float))
        except Exception:
            continue
    if len(valid) == 0:
        raise RuntimeError("All appraisal samples are invalid under the model constraints.")
    return np.vstack(valid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run joint inversion for one station using ray-parameter-bin RF stacks, then formal appraisal-stage posterior plotting."
    )
    parser.add_argument("config", help="Path to station TOML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from src.postprocess import (
        build_best_fit_synthetics,
        plot_best_fit_bins,
        plot_pbin_stacks,
        plot_posterior_tradeoffs,
        plot_search_tradeoffs,
        plot_search_velocity_family,
        plot_velocity_posterior_density,
        save_appraisal_samples,
        save_pbin_table,
        save_pbin_waveforms,
        save_search_results,
        save_station_metadata,
        summarize_appraisal,
        summarize_search,
    )
    out = ensure_output_dirs(cfg)

    station = cfg["station"]["name"]
    print(f"[INFO] Station: {station}")
    print("[INFO] Reading finallist...")
    records = parse_finallist(cfg["station"]["finallist"])

    print("[INFO] Loading RFs...")
    time, loaded, missing = load_station_rfs(records, cfg)

    print("[INFO] Building p-bin stacks...")
    bins = build_pbin_stacks(loaded, cfg)
    if len(bins) == 0:
        raise RuntimeError("No usable p-bins were created. Relax p-bin settings or inspect RF loading.")

    save_station_metadata(
        out["meta"] / "station_metadata.json",
        cfg,
        n_loaded=len(loaded),
        n_missing=len(missing),
        n_bins=len(bins),
    )
    save_pbin_table(bins, out["csv"] / "pbin_table.csv")
    save_pbin_waveforms(bins, time, out["pbins"] / "pbin_stacks.npz")
    plot_pbin_stacks(bins, time, out["figures"] / "pbin_stacks.png")

    print("[INFO] Running joint search...")
    objective = JointRFObjective(bins=bins, cfg=cfg, time=time)
    objective.setup_parallel()
    if objective.parallel_enabled:
        print(
            f"[INFO] Parallel enabled: workers={objective.n_workers}, "
            f"chunksize={objective.chunksize}, start_method={objective.start_method}"
        )
    else:
        print("[INFO] Parallel disabled; serial evaluation will be used.")

    try:
        search = run_na(cfg, objective)
    finally:
        objective.close()

    save_search_results(search.results, out["csv"] / "na_search_results.csv")
    top_fraction = float(cfg["na"].get("top_fraction", 0.10))
    summarize_search(search.results, cfg, top_fraction=top_fraction).to_csv(out["csv"] / "search_summary.csv", index=False)

    best = search.results.iloc[0]
    best_params = row_to_params(best)
    syns = build_best_fit_synthetics(best_params, bins, time, cfg)
    plot_best_fit_bins(bins, syns, time, out["figures"] / "bestfit_bins.png")

    # Search-stage diagnostic figures only.
    plot_search_tradeoffs(search.results, out["figures"] / "search_tradeoffs.png", top_fraction=top_fraction)
    plot_search_velocity_family(search.results, cfg, out["figures"] / "search_velocity_family.png", top_fraction=top_fraction)

    appraisal = None
    posterior_samples = None
    posterior_mean = None
    posterior_cov = None
    if bool(cfg["appraisal"].get("enabled", True)):
        print("[INFO] Running NAII-style appraisal...")
        appraisal = run_appraisal_from_search(search, cfg)
        save_appraisal_samples(appraisal.samples, out["csv"] / "naii_samples_raw.csv", cfg)

        posterior_samples = filter_valid_samples(appraisal.samples, cfg)
        n_total = len(appraisal.samples)
        n_valid = len(posterior_samples)
        print(f"[INFO] Appraisal valid samples: {n_valid}/{n_total} ({100.0 * n_valid / max(n_total, 1):.1f}%)")
        save_appraisal_samples(posterior_samples, out["csv"] / "naii_samples_valid.csv")

        posterior_mean = np.mean(posterior_samples, axis=0)
        if len(posterior_samples) > 1:
            posterior_cov = np.cov(posterior_samples, rowvar=False, ddof=1)
        else:
            posterior_cov = np.full((posterior_samples.shape[1], posterior_samples.shape[1]), np.nan)

        summarize_appraisal(posterior_samples, posterior_mean, posterior_cov, cfg).to_csv(
            out["csv"] / "appraisal_summary.csv", index=False
        )

        plot_posterior_tradeoffs(
            search_results=search.results,
            appraisal_samples=posterior_samples,
            appraisal_mean=posterior_mean,
            out_png=out["figures"] / "posterior_tradeoffs.png",
        )
        plot_velocity_posterior_density(
            search_results=search.results,
            appraisal_samples=posterior_samples,
            appraisal_mean=posterior_mean,
            cfg=cfg,
            out_png=out["figures"] / "posterior_velocity_density.png",
        )

    with (out["meta"] / "best_model.txt").open("w", encoding="utf-8") as f:
        f.write(f"Station: {station}\n")
        f.write(f"Search minimum misfit: {best['misfit']:.6f}\n")
        if appraisal is not None:
            f.write(f"Appraisal temperature: {appraisal.temperature:.6f}\n")
            f.write(f"Appraisal n_resample: {appraisal.n_resample}\n")
            f.write(f"Appraisal n_walkers: {appraisal.n_walkers}\n")
            f.write(f"Appraisal raw samples: {len(appraisal.samples)}\n")
            f.write(f"Appraisal valid samples: {len(posterior_samples)}\n")
        for name in get_parameter_names(cfg):
            f.write(f"{name}: {best[name]:.6f}\n")

    print("[INFO] Done.")
    print(f"[INFO] Output directory: {out['root']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
