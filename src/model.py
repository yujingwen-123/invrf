from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd

LEGACY_PARAMETER_NAMES = [
    "H_sed",
    "sed_ratio1",
    "Vs_sed1",
    "Vs_sed2",
    "VpVs_sed",
    "H_uc",
    "Vs_uc",
    "VpVs_uc",
    "H_moho",
    "Vs_lc",
    "VpVs_lc",
    "Vs_mantle",
    "VpVs_mantle",
]


CLASSIC_NAINVRF_PARAMETER_NAMES = [
    "H_sed", "Vs_sed0", "Vs_sedz", "K_sed",
    "H_c1", "H_c2", "H_c3", "Vs_c1", "Vs_c2", "Vs_c3", "VpVs_c1", "VpVs_c2", "VpVs_c3",
    "Vs_mantle", "VpVs_mantle",
]


def get_parameter_names(cfg: Dict | None = None) -> list[str]:
    mode = str((cfg or {}).get("model", {}).get("parameterization", "legacy")).lower()
    if mode == "classic_nainvrf":
        return CLASSIC_NAINVRF_PARAMETER_NAMES
    return LEGACY_PARAMETER_NAMES

PARAMETER_NAMES = LEGACY_PARAMETER_NAMES


@dataclass(frozen=True)
class LayerGeometry:
    h_sed1: float
    h_sed2: float
    h_uc: float
    h_lc: float
    H_moho: float


@dataclass(frozen=True)
class VelocityModel:
    interfaces_km: np.ndarray
    vp_km_s: np.ndarray
    vs_km_s: np.ndarray
    rho_g_cm3: np.ndarray
    geometry: LayerGeometry


def parameter_bounds(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    bounds = cfg["bounds_classic_nainvrf"] if mode == "classic_nainvrf" else cfg["bounds"]
    lower = np.array([float(bounds[name][0]) for name in get_parameter_names(cfg)], dtype=float)
    upper = np.array([float(bounds[name][1]) for name in get_parameter_names(cfg)], dtype=float)
    return lower, upper


def row_to_params(row: pd.Series | Dict) -> np.ndarray:
    names = [k for k in row.keys() if k != "misfit"]
    return np.array([float(row[name]) for name in names], dtype=float)


def params_to_dict(params: Iterable[float], cfg: Dict | None = None) -> Dict[str, float]:
    vals = list(map(float, params))
    names = get_parameter_names(cfg)
    return {name: vals[i] for i, name in enumerate(names)}


def vp2rho_brocher(vp: np.ndarray) -> np.ndarray:
    return (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )


def params_to_geometry(params: np.ndarray, cfg: Dict) -> LayerGeometry:
    p = params_to_dict(params, cfg)
    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    if mode == "classic_nainvrf":
        h_sed1 = 0.5 * p["H_sed"]; h_sed2 = p["H_sed"] - h_sed1
        h_uc = p["H_c1"] + p["H_c2"]
        h_lc = p["H_c3"]
        H_moho = p["H_sed"] + p["H_c1"] + p["H_c2"] + p["H_c3"]
    else:
        H_sed = p["H_sed"]
        sed_ratio1 = p["sed_ratio1"]
        h_sed1 = H_sed * sed_ratio1
        h_sed2 = H_sed - h_sed1
        h_uc = p["H_uc"]
        H_moho = p["H_moho"]
        h_lc = H_moho - H_sed - h_uc
    return LayerGeometry(
        h_sed1=float(h_sed1),
        h_sed2=float(h_sed2),
        h_uc=float(h_uc),
        h_lc=float(h_lc),
        H_moho=float(H_moho),
    )


def validate_params(params: np.ndarray, cfg: Dict) -> tuple[bool, float]:
    p = params_to_dict(params, cfg)
    geom = params_to_geometry(params, cfg)
    model_cfg = cfg["model"]
    min_lc = float(model_cfg.get("min_lower_crust_thickness", 5.0))
    allow_lvz = bool(model_cfg.get("allow_low_velocity_layer", False))
    penalty = 0.0

    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    if mode != "classic_nainvrf" and not (0.0 < p["sed_ratio1"] < 1.0):
        return False, 1.0e6
    if geom.h_lc < min_lc:
        return False, 1.0e6 + 100.0 * (min_lc - geom.h_lc) ** 2

    # Optional monotonic Vs ordering constraint:
    # - allow_low_velocity_layer = false (default): disallow LVZ by enforcing non-decreasing Vs with depth.
    # - allow_low_velocity_layer = true: allow velocity reversals (LVZ), only keep other geometric/physical checks.
    if not allow_lvz:
        if mode == "classic_nainvrf":
            if not (p["Vs_sed0"] <= p["Vs_sedz"] <= p["Vs_c1"] <= p["Vs_c2"] <= p["Vs_c3"] <= p["Vs_mantle"]):
                return False, 1.0e6
        else:
            if not (p["Vs_sed1"] <= p["Vs_sed2"] <= p["Vs_uc"] <= p["Vs_lc"] <= p["Vs_mantle"]):
                return False, 1.0e6
            if p["VpVs_sed"] < p["VpVs_uc"]:
                penalty += 25.0 * (p["VpVs_uc"] - p["VpVs_sed"]) ** 2

    return True, penalty



def _segment_grid(z0: float, z1: float, dz: float) -> np.ndarray:
    if z1 <= z0:
        return np.array([z0], dtype=float)
    if dz <= 0:
        return np.array([z0, z1], dtype=float)
    pts = [float(z0)]
    z = float(z0)
    while z + dz < z1:
        z += dz
        pts.append(float(z))
    if pts[-1] < z1:
        pts.append(float(z1))
    return np.array(pts, dtype=float)


def _build_interpolated_layers(
    depth_bounds: np.ndarray,
    vs_nodes: np.ndarray,
    vpvs_nodes: np.ndarray,
    sed_dz: float,
    crust_dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_sed = float(depth_bounds[2])
    h_moho = float(depth_bounds[-1])

    sed_grid = _segment_grid(0.0, h_sed, sed_dz)
    crust_grid = _segment_grid(h_sed, h_moho, crust_dz)

    z_grid = np.concatenate([sed_grid, crust_grid[1:]])
    z_grid = np.unique(np.clip(z_grid, 0.0, h_moho))
    if z_grid[-1] < h_moho:
        z_grid = np.append(z_grid, h_moho)

    z_nodes = np.asarray(depth_bounds[:-1], dtype=float)
    vs = np.interp(z_grid[:-1], z_nodes, vs_nodes)
    vpvs = np.interp(z_grid[:-1], z_nodes, vpvs_nodes)
    return z_grid[1:], vs, vpvs


def params_to_velocity_model(params: np.ndarray, cfg: Dict) -> VelocityModel:
    valid, penalty = validate_params(params, cfg)
    if not valid:
        raise ValueError(f"Invalid parameter set with penalty={penalty}")

    p = params_to_dict(params, cfg)
    geom = params_to_geometry(params, cfg)

    mode = str(cfg.get("model", {}).get("parameterization", "legacy")).lower()
    if mode == "classic_nainvrf":
        depth_bounds = np.array([0.0, geom.h_sed1, geom.h_sed1 + geom.h_sed2, geom.h_sed1 + geom.h_sed2 + p["H_c1"], geom.h_sed1 + geom.h_sed2 + p["H_c1"] + p["H_c2"], geom.H_moho], dtype=float)
        vs_nodes = np.array([p["Vs_sed0"], p["Vs_sedz"], p["Vs_c1"], p["Vs_c2"], p["Vs_c3"]], dtype=float)
        vpvs_nodes = np.array([p["K_sed"], p["K_sed"], p["VpVs_c1"], p["VpVs_c2"], p["VpVs_c3"]], dtype=float)
    else:
        depth_bounds = np.array([0.0, geom.h_sed1, geom.h_sed1 + geom.h_sed2, geom.h_sed1 + geom.h_sed2 + geom.h_uc, geom.H_moho], dtype=float)
        vs_nodes = np.array([p["Vs_sed1"], p["Vs_sed2"], p["Vs_uc"], p["Vs_lc"]], dtype=float)
        vpvs_nodes = np.array([p["VpVs_sed"], p["VpVs_sed"], p["VpVs_uc"], p["VpVs_lc"]], dtype=float)
    sed_dz = float(cfg.get("model", {}).get("sed_control_dz_km", 0.0))
    crust_dz = float(cfg.get("model", {}).get("crust_control_dz_km", 0.0))
    interfaces, vs_finite, vpvs_finite = _build_interpolated_layers(depth_bounds, vs_nodes, vpvs_nodes, sed_dz, crust_dz)

    vs = np.concatenate([vs_finite, np.array([p["Vs_mantle"]], dtype=float)])
    vpvs = np.concatenate([vpvs_finite, np.array([p["VpVs_mantle"]], dtype=float)])
    vp = vs * vpvs
    rho = vp2rho_brocher(vp)
    return VelocityModel(
        interfaces_km=interfaces,
        vp_km_s=vp,
        vs_km_s=vs,
        rho_g_cm3=rho,
        geometry=geom,
    )


def model_to_depth_grid(params: np.ndarray, cfg: Dict, z: np.ndarray) -> np.ndarray:
    vm = params_to_velocity_model(params, cfg)
    bounds = np.concatenate([np.array([0.0]), vm.interfaces_km, np.array([np.inf])])
    vs = vm.vs_km_s
    out = np.empty_like(z, dtype=float)
    for i in range(len(vs)):
        mask = (z >= bounds[i]) & (z < bounds[i + 1])
        out[mask] = vs[i]
    return out
