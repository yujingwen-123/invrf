from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from functools import partial

import numpy as np


def _obj_wrapper(func: Callable, args: tuple, kwargs: dict, x: np.ndarray) -> float:
    return float(func(x, *args, **kwargs))


def _is_feasible_wrapper(func: Callable, x: np.ndarray) -> bool:
    return bool(np.all(func(x) >= 0))


def _cons_none_wrapper(x: np.ndarray) -> np.ndarray:
    return np.array([0.0])


def pso(
    func: Callable,
    lb: list | np.ndarray,
    ub: list | np.ndarray,
    ieqcons: list | None = None,
    f_ieqcons: Callable | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    swarmsize: int = 100,
    omega: float = 0.5,
    phip: float = 0.5,
    phig: float = 0.5,
    maxiter: int = 100,
    minstep: float = 1e-8,
    minfunc: float = 1e-8,
    debug: bool = False,
    processes: int = 1,
    particle_output: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if kwargs is None:
        kwargs = {}
    if ieqcons is None:
        ieqcons = []
    if len(lb) != len(ub):
        raise ValueError("Lower- and upper-bounds must be the same length")
    if not callable(func):
        raise TypeError("Invalid function handle")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if not np.all(ub > lb):
        raise ValueError("All upper-bound values must be greater than lower-bound values")

    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    rng = np.random.default_rng(seed)

    obj = partial(_obj_wrapper, func, args, kwargs)
    if f_ieqcons is None and not ieqcons:
        cons = _cons_none_wrapper
    elif f_ieqcons is not None:
        cons = lambda x: np.asarray(f_ieqcons(x, *args, **kwargs), dtype=float)
    else:
        cons = lambda x: np.asarray([y(x, *args, **kwargs) for y in ieqcons], dtype=float)
    is_feasible = partial(_is_feasible_wrapper, cons)

    mp_pool = multiprocessing.Pool(processes) if processes > 1 else None
    swarm_size = int(swarmsize)
    num_dims = len(lb)

    x = lb + rng.random((swarm_size, num_dims)) * (ub - lb)
    v = np.zeros_like(x)
    p = np.zeros_like(x)
    fx = np.zeros(swarm_size)
    fs = np.zeros(swarm_size, dtype=bool)
    fp = np.ones(swarm_size) * np.inf

    if mp_pool is not None:
        fx = np.asarray(mp_pool.map(obj, x), dtype=float)
        fs = np.asarray(mp_pool.map(is_feasible, x), dtype=bool)
    else:
        for i in range(swarm_size):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])

    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    i_min = int(np.argmin(fp))
    if fp[i_min] < np.inf:
        g = p[i_min, :].copy()
        fg = float(fp[i_min])
    else:
        g = x[0, :].copy()
        fg = float(obj(g))

    v = rng.uniform(vlow, vhigh, size=(swarm_size, num_dims))

    it = 1
    while it <= maxiter:
        rp = rng.uniform(size=(swarm_size, num_dims))
        rg = rng.uniform(size=(swarm_size, num_dims))
        v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
        x += v
        x = np.clip(x, lb, ub)

        if mp_pool is not None:
            fx = np.asarray(mp_pool.map(obj, x), dtype=float)
            fs = np.asarray(mp_pool.map(is_feasible, x), dtype=bool)
        else:
            for i in range(swarm_size):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        i_min = int(np.argmin(fp))
        if fp[i_min] < fg:
            p_min = p[i_min, :].copy()
            stepsize = float(np.sqrt(np.sum((g - p_min) ** 2)))
            if np.abs(fg - fp[i_min]) <= minfunc or stepsize <= minstep:
                g = p_min
                fg = float(fp[i_min])
                break
            g = p_min
            fg = float(fp[i_min])
        if debug:
            print(f"Best after iteration {it}: {g} {fg}")
        it += 1

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()

    if particle_output:
        return g, fg, p, fp
    return g, fg
