"""Generate a generic ODE trajectory dataset for Mezzanine.

This writes a `.npz` consumed by `ode_time_origin_distill`.

Gold standard: SciPy `solve_ivp` with dense output sampled on a uniform grid.

By default we generate Lorenz trajectories; you can switch to Van der Pol.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Tuple

import numpy as np


def lorenz_rhs(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
    def f(t: float, x: np.ndarray) -> np.ndarray:
        dx = np.empty_like(x)
        dx[0] = sigma * (x[1] - x[0])
        dx[1] = x[0] * (rho - x[2]) - x[1]
        dx[2] = x[0] * x[1] - beta * x[2]
        return dx
    return f


def vdp_rhs(mu: float = 5.0):
    def f(t: float, x: np.ndarray) -> np.ndarray:
        # x = [q, p]
        q, p = x[0], x[1]
        dq = p
        dp = mu * (1.0 - q * q) * p - q
        return np.array([dq, dp], dtype=np.float64)
    return f


def solve_traj(rhs, x0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    try:
        from scipy.integrate import solve_ivp  # type: ignore
    except Exception:
        solve_ivp = None

    if solve_ivp is not None:
        sol = solve_ivp(
            rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            x0.astype(np.float64),
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return np.asarray(sol.y.T, dtype=np.float32)  # [T,D]

    # Fallback: fixed-step RK4 on the uniform grid (no SciPy dependency).
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.ndim != 1 or t_eval.shape[0] < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 time points")
    dt = float(t_eval[1] - t_eval[0])
    if not np.allclose(np.diff(t_eval), dt):
        raise ValueError("RK4 fallback requires a uniform time grid")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    x = np.empty((int(t_eval.shape[0]), int(x0.shape[0])), dtype=np.float64)
    x[0] = x0.astype(np.float64)
    for i in range(int(t_eval.shape[0]) - 1):
        t = float(t_eval[i])
        xi = x[i]
        k1 = rhs(t, xi)
        k2 = rhs(t + 0.5 * dt, xi + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, xi + 0.5 * dt * k2)
        k4 = rhs(t + dt, xi + dt * k3)
        x[i + 1] = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x.astype(np.float32)  # [T,D]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--system", type=str, default="lorenz", choices=["lorenz", "vdp"])
    ap.add_argument("--n_train_traj", type=int, default=819)
    ap.add_argument("--n_test_traj", type=int, default=205)
    ap.add_argument(
        "--n_traj",
        type=int,
        default=None,
        help="Alias: total trajectories; splits 80/20 into train/test.",
    )
    ap.add_argument("--T", type=int, default=201, help="Number of time points")
    ap.add_argument(
        "--t_max",
        type=float,
        default=None,
        help="Alias: maximum time (sets T so t_eval[-1] ~= t_max).",
    )
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError("--dt must be > 0")

    T_default = 201
    if args.t_max is not None:
        if int(args.T) != T_default:
            raise ValueError("Specify only one of --T or --t_max")
        t_max = float(args.t_max)
        if t_max <= 0.0:
            raise ValueError("--t_max must be > 0")
        # Use floor so we don't overshoot t_max, and include the initial point.
        T = int(np.floor(t_max / dt)) + 1
    else:
        T = int(args.T)
    if T < 2:
        raise ValueError("--T must be >= 2 (or choose a larger --t_max / smaller --dt)")

    # Keep the integration grid in float64 so the RK4 fallback sees a truly
    # uniform grid. Casting long grids to float32 can accumulate rounding error
    # and trip the uniform-step check.
    t_eval = np.arange(T, dtype=np.float64) * dt
    t_eval_f32 = t_eval.astype(np.float32)

    if args.system == "lorenz":
        rhs = lorenz_rhs()
        D = 3
        def sample_x0() -> np.ndarray:
            return rng.normal(loc=0.0, scale=5.0, size=(D,)).astype(np.float64)
    else:
        rhs = vdp_rhs()
        D = 2
        def sample_x0() -> np.ndarray:
            return rng.normal(loc=0.0, scale=2.0, size=(D,)).astype(np.float64)

    n_train_default = 819
    n_test_default = 205
    if args.n_traj is not None:
        if int(args.n_train_traj) != n_train_default or int(args.n_test_traj) != n_test_default:
            raise ValueError("Specify --n_traj or (--n_train_traj, --n_test_traj), not both")
        n_total = int(args.n_traj)
        if n_total < 0:
            raise ValueError("--n_traj must be >= 0")
        n_train = int(np.floor(0.8 * n_total))
        n_test = int(n_total - n_train)
    else:
        n_train = int(args.n_train_traj)
        n_test = int(args.n_test_traj)
    if n_train < 0 or n_test < 0:
        raise ValueError("--n_train_traj/--n_test_traj must be >= 0")

    train_x = np.zeros((n_train, T, D), dtype=np.float32)
    test_x = np.zeros((n_test, T, D), dtype=np.float32)

    for i in range(n_train):
        train_x[i] = solve_traj(rhs, sample_x0(), t_eval)
    for i in range(n_test):
        test_x[i] = solve_traj(rhs, sample_x0(), t_eval)

    np.savez(
        out,
        train_x=train_x,
        test_x=test_x,
        train_t=t_eval_f32,
        test_t=t_eval_f32,
        dt=np.array([dt], dtype=np.float32),
        source=str({"system": args.system, "dt": dt, "T": T}),
    )

    print(f"Wrote {out} with {n_train} train trajectories and {n_test} test trajectories.")


if __name__ == "__main__":
    main()
