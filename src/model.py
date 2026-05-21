from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PARAMETER_NAMES = [
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
    bounds = cfg["bounds"]
    lower = np.array([float(bounds[name][0]) for name in PARAMETER_NAMES], dtype=float)
    upper = np.array([float(bounds[name][1]) for name in PARAMETER_NAMES], dtype=float)
    return lower, upper


def row_to_params(row: pd.Series | Dict) -> np.ndarray:
    return np.array([float(row[name]) for name in PARAMETER_NAMES], dtype=float)


def params_to_dict(params: Iterable[float]) -> Dict[str, float]:
    vals = list(map(float, params))
    return {name: vals[i] for i, name in enumerate(PARAMETER_NAMES)}


def vp2rho_brocher(vp: np.ndarray) -> np.ndarray:
    return (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )




def v2v(vp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical relationship in sediment layer to derive Vs and density from Vp."""
    rho = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    vs = 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
    return vs, rho

def params_to_geometry(params: np.ndarray, cfg: Dict) -> LayerGeometry:
    p = params_to_dict(params)
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
    p = params_to_dict(params)
    geom = params_to_geometry(params, cfg)
    model_cfg = cfg["model"]
    min_lc = float(model_cfg.get("min_lower_crust_thickness", 5.0))
    penalty = 0.0

    if not (0.0 < p["sed_ratio1"] < 1.0):
        return False, 1.0e6
    if geom.h_lc < min_lc:
        return False, 1.0e6 + 100.0 * (min_lc - geom.h_lc) ** 2

    # Weak monotonicity / physical ordering constraints.
    if not (p["Vs_sed1"] <= p["Vs_sed2"] <= p["Vs_uc"] <= p["Vs_lc"] <= p["Vs_mantle"]):
        return False, 1.0e6

    if p["VpVs_sed"] < p["VpVs_uc"]:
        penalty += 25.0 * (p["VpVs_uc"] - p["VpVs_sed"]) ** 2

    return True, penalty


def params_to_velocity_model(params: np.ndarray, cfg: Dict) -> VelocityModel:
    valid, penalty = validate_params(params, cfg)
    if not valid:
        raise ValueError(f"Invalid parameter set with penalty={penalty}")

    p = params_to_dict(params)
    geom = params_to_geometry(params, cfg)

    interfaces = np.array(
        [
            geom.h_sed1,
            geom.h_sed1 + geom.h_sed2,
            geom.h_sed1 + geom.h_sed2 + geom.h_uc,
            geom.H_moho,
        ],
        dtype=float,
    )
    vs = np.array(
        [
            p["Vs_sed1"],
            p["Vs_sed2"],
            p["Vs_uc"],
            p["Vs_lc"],
            p["Vs_mantle"],
        ],
        dtype=float,
    )
    vpvs = np.array(
        [
            p["VpVs_sed"],
            p["VpVs_sed"],
            p["VpVs_uc"],
            p["VpVs_lc"],
            p["VpVs_mantle"],
        ],
        dtype=float,
    )
    vp = vs * vpvs

    # Use empirical sediment relationship for the first two sediment layers.
    sed_vs, sed_rho = v2v(vp[:2])
    vs = vs.copy()
    rho = vp2rho_brocher(vp)
    vs[:2] = sed_vs
    rho[:2] = sed_rho
    return VelocityModel(
        interfaces_km=interfaces,
        vp_km_s=vp,
        vs_km_s=vs,
        rho_g_cm3=rho,
        geometry=geom,
    )


def model_to_depth_grid(params: np.ndarray, cfg: Dict, z: np.ndarray) -> np.ndarray:
    vm = params_to_velocity_model(params, cfg)
    g = vm.geometry
    bounds = np.array([0.0, g.h_sed1, g.h_sed1 + g.h_sed2, g.h_sed1 + g.h_sed2 + g.h_uc, g.H_moho, np.inf])
    vs = vm.vs_km_s
    out = np.empty_like(z, dtype=float)
    for i in range(len(vs)):
        mask = (z >= bounds[i]) & (z < bounds[i + 1])
        out[mask] = vs[i]
    out[z >= g.H_moho] = vs[-1]
    return out
