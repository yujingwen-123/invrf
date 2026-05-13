#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def read_station_lalo(path: str) -> pd.DataFrame:
    """读取台站经纬度: stnm stla stlo [stle]."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["stnm", "stla", "stlo", "stle"],
        engine="python",
    )
    df["stnm"] = df["stnm"].astype(str).str.strip()
    return df


def _read_summary(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, sep=r"\s+", engine="python")
    else:
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    df.columns = df.columns.str.strip()
    if "station" in df.columns and "stnm" not in df.columns:
        df = df.rename(columns={"station": "stnm"})
    if "stnm" not in df.columns:
        raise RuntimeError(f"summary_csv 没有 stnm/station 列，当前列名: {list(df.columns)}")
    df["stnm"] = df["stnm"].astype(str).str.strip()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="按台站经纬度导出整网参数结果文件")
    parser.add_argument("lalo_file", help="台站经纬度文件: stnm stla stlo [stle]")
    parser.add_argument("summary_csv", help="整网汇总 CSV/TXT（需含 stnm 和统计列）")
    parser.add_argument("--outdir", default="lalo_param_out", help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lalo = read_station_lalo(args.lalo_file)
    res = _read_summary(args.summary_csv)

    merged = pd.merge(lalo, res, on="stnm", how="inner")
    if merged.empty:
        raise RuntimeError("合并结果为空，请检查台站名是否一致。")

    need = [
        "H_sed_mean", "H_sed_std",
        "H_moho_mean", "H_moho_std",
        "K_full_mean", "K_full_std",
        "K_crust_mean", "K_crust_std",
        "K_sed_mean", "K_sed_std",
    ]
    missing = [c for c in need if c not in merged.columns]
    if missing:
        raise RuntimeError(f"summary_csv 缺少这些列: {missing}")

    merged["H_crust_mean"] = merged["H_moho_mean"] - merged["H_sed_mean"]
    merged["H_crust_std"] = np.sqrt(merged["H_moho_std"] ** 2 + merged["H_sed_std"] ** 2)

    # 1) 输出单参数文件（兼容你现有工作流）
    outputs = [
        ("whole", "H", "H_moho_mean", "H_moho_std"),
        ("whole", "K", "K_full_mean", "K_full_std"),
        ("crust", "H", "H_crust_mean", "H_crust_std"),
        ("crust", "K", "K_crust_mean", "K_crust_std"),
        ("sed", "H", "H_sed_mean", "H_sed_std"),
        ("sed", "K", "K_sed_mean", "K_sed_std"),
    ]
    for site, param_name, mean_col, std_col in outputs:
        out = merged[["stnm", "stlo", "stla", mean_col, std_col]].copy()
        out.columns = ["stnm", "stlo", "stla", param_name, f"std_{param_name}"]
        outfile = outdir / f"{site}_lalo_{param_name}.txt"
        out.to_csv(outfile, sep=" ", index=False, header=False, float_format="%.6f")
        print(f"[INFO] saved: {outfile}")

    # 2) 输出全网合并总表（一个文件包含全部参数）
    all_cols = [
        "stnm", "stlo", "stla",
        "H_sed_mean", "H_sed_std",
        "H_moho_mean", "H_moho_std",
        "H_crust_mean", "H_crust_std",
        "K_sed_mean", "K_sed_std",
        "K_full_mean", "K_full_std",
        "K_crust_mean", "K_crust_std",
    ]
    all_out = merged[all_cols].copy()
    all_file = outdir / "network_lalo_all_params.txt"
    all_out.to_csv(all_file, sep=" ", index=False, header=True, float_format="%.6f")
    print(f"[INFO] saved: {all_file}")


if __name__ == "__main__":
    main()
