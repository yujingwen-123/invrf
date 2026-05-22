#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def vs2vp_brocher(vs: np.ndarray) -> np.ndarray:
    return 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4


def _std_or_nan(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize station H_sed/H_moho and travel-time-average K from posterior samples")
    parser.add_argument("--root", default="./results", help="Result root directory containing station subfolders")
    parser.add_argument("--out", default="station_hsed_moho_k_summary.csv", help="Output summary file path")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, float | str]] = []

    for sta_dir in sorted(root.iterdir()):
        if not sta_dir.is_dir():
            continue

        stnm = sta_dir.name
        samp = sta_dir / "csv" / "naii_samples_valid.csv"
        if not samp.exists():
            print(f"[WARN] missing: {samp}")
            continue

        df = pd.read_csv(samp)

        need = [
            "H_sed", "sed_ratio1", "Vs_sed1", "Vs_sed2",
            "H_uc", "Vs_uc", "VpVs_uc",
            "H_moho", "Vs_lc", "VpVs_lc",
        ]
        if any(c not in df.columns for c in need):
            print(f"[WARN] missing required columns in {samp}")
            continue

        H_sed = df["H_sed"].to_numpy(float)
        sed_ratio1 = df["sed_ratio1"].to_numpy(float)
        Vs_sed1 = df["Vs_sed1"].to_numpy(float)
        Vs_sed2 = df["Vs_sed2"].to_numpy(float)

        H_uc = df["H_uc"].to_numpy(float)
        Vs_uc = df["Vs_uc"].to_numpy(float)
        K_uc = df["VpVs_uc"].to_numpy(float)

        H_moho = df["H_moho"].to_numpy(float)
        Vs_lc = df["Vs_lc"].to_numpy(float)
        K_lc = df["VpVs_lc"].to_numpy(float)

        h_sed1 = H_sed * sed_ratio1
        h_sed2 = H_sed * (1.0 - sed_ratio1)
        h_lc = H_moho - H_sed - H_uc

        # Sediment Vp/Vs is no longer an inversion parameter.
        # Use the same empirical Vs->Vp conversion as the forward model.
        Vp_sed1 = vs2vp_brocher(Vs_sed1)
        Vp_sed2 = vs2vp_brocher(Vs_sed2)

        mask = (
            np.isfinite(H_sed) & np.isfinite(H_uc) & np.isfinite(H_moho) &
            np.isfinite(h_sed1) & np.isfinite(h_sed2) & np.isfinite(h_lc) &
            np.isfinite(Vs_sed1) & np.isfinite(Vs_sed2) &
            np.isfinite(Vs_uc) & np.isfinite(K_uc) &
            np.isfinite(Vs_lc) & np.isfinite(K_lc) &
            np.isfinite(Vp_sed1) & np.isfinite(Vp_sed2) &
            (h_sed1 > 0) & (h_sed2 > 0) & (h_lc > 0) &
            (Vs_sed1 > 0) & (Vs_sed2 > 0) &
            (Vs_uc > 0) & (K_uc > 0) &
            (Vs_lc > 0) & (K_lc > 0) &
            (Vp_sed1 > 0) & (Vp_sed2 > 0)
        )

        if mask.sum() == 0:
            print(f"[WARN] no valid samples in {samp}")
            continue

        H_sed = H_sed[mask]
        H_moho = H_moho[mask]

        h_sed1 = h_sed1[mask]
        h_sed2 = h_sed2[mask]
        h_lc = h_lc[mask]
        H_uc = H_uc[mask]

        Vs_sed1 = Vs_sed1[mask]
        Vs_sed2 = Vs_sed2[mask]
        Vp_sed1 = Vp_sed1[mask]
        Vp_sed2 = Vp_sed2[mask]

        Vs_uc = Vs_uc[mask]
        K_uc = K_uc[mask]

        Vs_lc = Vs_lc[mask]
        K_lc = K_lc[mask]

        tS_sed = h_sed1 / Vs_sed1 + h_sed2 / Vs_sed2
        tP_sed = h_sed1 / Vp_sed1 + h_sed2 / Vp_sed2

        tS_crust = H_uc / Vs_uc + h_lc / Vs_lc
        tP_crust = H_uc / (K_uc * Vs_uc) + h_lc / (K_lc * Vs_lc)

        tS_full = tS_sed + tS_crust
        tP_full = tP_sed + tP_crust

        Ksed_tt = tS_sed / tP_sed
        # Crystalline crust only (upper+lower crust, excluding sediments).
        Kcrystalline_tt = tS_crust / tP_crust
        # Whole crust (sediment + crystalline crust), consistent with priors.K_crust_center.
        Kcrust_tt = tS_full / tP_full

        rows.append({
            "station": stnm,
            "H_sed_mean": float(np.mean(H_sed)),
            "H_sed_std": _std_or_nan(H_sed),
            "H_moho_mean": float(np.mean(H_moho)),
            "H_moho_std": _std_or_nan(H_moho),
            "K_full_mean": float(np.mean(Kcrust_tt)),
            "K_full_std": _std_or_nan(Kcrust_tt),
            "K_crust_mean": float(np.mean(Kcrust_tt)),
            "K_crust_std": _std_or_nan(Kcrust_tt),
            "K_crystalline_mean": float(np.mean(Kcrystalline_tt)),
            "K_crystalline_std": _std_or_nan(Kcrystalline_tt),
            "K_sed_mean": float(np.mean(Ksed_tt)),
            "K_sed_std": _std_or_nan(Ksed_tt),
        })

    out = pd.DataFrame(rows).sort_values("station")
    out.to_csv(args.out, sep=" ", index=False)
    print(out)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
