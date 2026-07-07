"""SYSTEM-ID: fit a per-serve faithful ball model that reproduces each REAL trajectory to the hit
plane, so we can sample new physically-consistent serves.

Two models (chosen by ``--model``); both also fit the launch velocity v0 (the KF lock-on velocity is
unreliable due to serve-onset):

  * ``dragbounce`` (DEFAULT): v0 + per-serve air_drag_coeff + bounce_alpha_z + bounce_alpha_xy,
    Magnus OFF. Stays inside the KF's own model family (no assumed Magnus coefficient). Experiment 3
    showed this reproduces trajectories as well as spin.
  * ``spin``: v0 + spin, Magnus ON (paper coeff). Kept for comparison; NOT used for generation.

Fit target = KF-filtered positions from lock-on to the plane. ``--save`` fits all serves, applies a
reproduction-quality gate, and dumps the per-serve latents for distribution fitting + generation.

Usage:
  python serve_fit.py --data-dir /data/PPO-pingpong/data/0629_RL_traj --save fitted_serves.npz
  python serve_fit.py --data-dir ... --n 20 --model spin   # comparison only
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import least_squares

try:
    from .ball_physics import ball_config, default_config, first_downward_crossing, gen_config, rollout_positions
    from .recording_io import Serve, load_dir
except ImportError:
    from ball_physics import ball_config, default_config, first_downward_crossing, gen_config, rollout_positions
    from recording_io import Serve, load_dir

PLANE_X = -1.37
MM = 1000.0
# physical bounds for the drag/bounce latents (drag ~ base 1.05e-4; alpha_z restitution; alpha_xy retention)
DRAG_LO, DRAG_HI = 1e-5, 5e-4
AZ_LO, AZ_HI = 0.5, 1.0
AXY_LO, AXY_HI = 0.3, 1.0
SPIN_MAX, V0_MAX = 600.0, 20000.0


def _window(s: Serve):
    """Indices [i0, iend) from lock-on to just past the plane crossing."""
    vi = np.where(s.valid)[0]
    if vi.size < 8:
        return None
    i0 = int(vi[0])
    below = np.where(s.kf[i0:, 0] <= PLANE_X - 0.02)[0]
    iend = i0 + int(below[0]) + 2 if below.size else len(s.t)
    if iend - i0 < 8:
        return None
    return i0, min(iend, len(s.t))


def _rmse_cm(traj, obs):
    return float(np.sqrt(np.mean(np.sum((traj - obs) ** 2, axis=1)))) / 10.0


def _endpoint_cm(traj, obs, t):
    a = first_downward_crossing(traj[:, 0] / MM, traj[:, 1] / MM, traj[:, 2] / MM, t, PLANE_X)
    b = first_downward_crossing(obs[:, 0] / MM, obs[:, 1] / MM, obs[:, 2] / MM, t, PLANE_X)
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))) * 100


def fit_serve(s: Serve, model: str = "dragbounce", magnus_coeff: float = 0.003604):
    w = _window(s)
    if w is None:
        return None
    i0, iend = w
    t = s.t[i0:iend] - s.t[i0]
    obs = s.kf[i0:iend, 0:3] * MM
    pos0 = obs[0].copy()
    v0_kf = s.kf[i0, 3:6] * MM

    base = rollout_positions(pos0, v0_kf, 0.0, t, default_config(), spin_rad_s=None)
    base_rmse = _rmse_cm(base, obs)

    if model == "spin":
        gcfg = gen_config(magnus_coeff)
        def resid(p):
            return (rollout_positions(pos0, p[0:3], 0.0, t, gcfg, spin_rad_s=p[3:6]) - obs).ravel()
        lb = np.array([-V0_MAX] * 3 + [-SPIN_MAX] * 3)
        ub = np.array([V0_MAX] * 3 + [SPIN_MAX] * 3)
        x0 = np.concatenate([v0_kf, np.zeros(3)])
        res = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=20.0, max_nfev=80)
        v0_fit = res.x[0:3]
        fit = rollout_positions(pos0, v0_fit, 0.0, t, gcfg, spin_rad_s=res.x[3:6])
        extra = {"spin_rad_s": res.x[3:6]}
    else:  # dragbounce
        d0 = default_config().air_drag_coeff
        az0, axy0 = default_config().bounce_alpha_z, default_config().bounce_alpha_xy
        def resid(p):
            cfg = ball_config(p[3], p[4], p[5])
            return (rollout_positions(pos0, p[0:3], 0.0, t, cfg, spin_rad_s=None) - obs).ravel()
        lb = np.array([-V0_MAX] * 3 + [DRAG_LO, AZ_LO, AXY_LO])
        ub = np.array([V0_MAX] * 3 + [DRAG_HI, AZ_HI, AXY_HI])
        x0 = np.array([v0_kf[0], v0_kf[1], v0_kf[2], d0, az0, axy0])
        res = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=20.0, max_nfev=120)
        v0_fit = res.x[0:3]
        fit = rollout_positions(pos0, v0_fit, 0.0, t, ball_config(res.x[3], res.x[4], res.x[5]), spin_rad_s=None)
        extra = {"drag": float(res.x[3]), "alpha_z": float(res.x[4]), "alpha_xy": float(res.x[5])}

    cr = first_downward_crossing(obs[:, 0] / MM, obs[:, 1] / MM, obs[:, 2] / MM, t, PLANE_X)
    return {
        "serve_id": s.serve_id,
        "base_rmse_cm": base_rmse,
        "fit_rmse_cm": _rmse_cm(fit, obs),
        "base_end_cm": _endpoint_cm(base, obs, t),
        "fit_end_cm": _endpoint_cm(fit, obs, t),
        "pos0_mm": pos0,
        "v0_mm_s": v0_fit,
        "tau_lockon_s": float(cr[2]) if cr is not None else float(t[-1]),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--n", type=int, default=15, help="-1 = all serves")
    ap.add_argument("--model", choices=["dragbounce", "spin"], default="dragbounce")
    ap.add_argument("--magnus-coeff", type=float, default=0.003604)
    ap.add_argument("--save", default=None, help="npz of fitted latents (implies all serves)")
    ap.add_argument("--max-fit-rmse-cm", type=float, default=5.0, help="quality gate for saved serves")
    args = ap.parse_args()
    serves = load_dir(args.data_dir)
    if args.save:
        args.n = -1
    if args.n >= 0:
        serves = serves[: args.n]
    rows = [r for s in serves for r in [fit_serve(s, args.model, args.magnus_coeff)] if r is not None]
    if not rows:
        print("no serves fit")
        return
    br = np.array([r["base_rmse_cm"] for r in rows]); fr = np.array([r["fit_rmse_cm"] for r in rows])
    be = np.array([r["base_end_cm"] for r in rows]); fe = np.array([r["fit_end_cm"] for r in rows])
    print(f"\n=== per-serve fit  model={args.model}  (n={len(rows)}) ===")
    print(f"  full-path RMSE cm: baseline median={np.median(br):.2f} -> fit median={np.median(fr):.2f}")
    print(f"  endpoint@plane cm: baseline median={np.nanmedian(be):.2f} -> fit median={np.nanmedian(fe):.2f}")
    print(f"  fit_rmse<3cm: {(fr < 3).mean()*100:.0f}%")
    if args.model == "dragbounce":
        dr = np.array([r["drag"] for r in rows]); az = np.array([r["alpha_z"] for r in rows]); axy = np.array([r["alpha_xy"] for r in rows])
        base = default_config()
        print(f"  drag     median={np.median(dr):.2e} (KF {base.air_drag_coeff:.2e})")
        print(f"  alpha_z  median={np.median(az):.3f} (KF {base.bounce_alpha_z:.3f})")
        print(f"  alpha_xy median={np.median(axy):.3f} (KF {base.bounce_alpha_xy:.3f})")

    if args.save:
        keep = [r for r in rows if r["fit_rmse_cm"] < args.max_fit_rmse_cm]
        print(f"  quality gate fit_rmse<{args.max_fit_rmse_cm}cm: kept {len(keep)}/{len(rows)}")
        np.savez(
            args.save,
            model=args.model,
            pos0_mm=np.array([r["pos0_mm"] for r in keep]),
            v0_mm_s=np.array([r["v0_mm_s"] for r in keep]),
            drag=np.array([r["drag"] for r in keep]),
            alpha_z=np.array([r["alpha_z"] for r in keep]),
            alpha_xy=np.array([r["alpha_xy"] for r in keep]),
            tau_lockon_s=np.array([r["tau_lockon_s"] for r in keep]),
            fit_rmse_cm=np.array([r["fit_rmse_cm"] for r in keep]),
            serve_id=np.array([r["serve_id"] for r in keep]),
        )
        print(f"  saved -> {args.save}  ({len(keep)} serves)")


if __name__ == "__main__":
    main()
