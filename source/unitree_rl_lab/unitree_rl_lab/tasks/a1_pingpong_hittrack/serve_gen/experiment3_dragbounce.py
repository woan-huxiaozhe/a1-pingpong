"""EXPERIMENT 3 (discriminating test): is the real trajectory curvature SPIN, or just error in the
KF's drag/bounce parameters?

For the same serves, fit two 6-parameter models to reproduce the real KF-position trajectory
(both also fit the launch velocity v0, since the KF lock-on velocity is unreliable):

  A) v0(3) + spin(3)          -- Magnus ON (paper coeff), the model used so far.
  B) v0(3) + drag/bounce(3)   -- spin = 0, magnus OFF; free per-serve air_drag_coeff, bounce_alpha_z
                                 (vertical restitution) and bounce_alpha_xy (horizontal retention).
                                 These are ISOTROPIC/symmetric -- they can change speed, height and
                                 bounce energy but CANNOT create lateral (y) curvature.

Read-out:
  * If B reaches ~A (~2 cm) -> the deviation is explained by drag/bounce model error, NOT spin;
    we should perturb drag/bounce (self-consistent with the KF's own model family) and drop Magnus.
  * If B stays well above A -> symmetric drag/bounce cannot reproduce the trajectory; the curvature
    genuinely needs spin (a lateral/Magnus effect).

Usage: python experiment3_dragbounce.py --data-dir /data/PPO-pingpong/data/0629_RL_traj --n 20
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

try:
    from .ball_physics import default_config, gen_config, rollout_positions
    from .recording_io import load_dir
    from .serve_fit import _window
except ImportError:
    from ball_physics import default_config, gen_config, rollout_positions
    from recording_io import load_dir
    from serve_fit import _window

MM = 1000.0


def _obs(s):
    w = _window(s)
    if w is None:
        return None
    i0, iend = w
    t = s.t[i0:iend] - s.t[i0]
    obs = s.kf[i0:iend, 0:3] * MM
    v0_kf = s.kf[i0, 3:6] * MM
    return t, obs, obs[0].copy(), v0_kf


def fit_spin(t, obs, pos0, v0_kf):
    gcfg = gen_config(0.003604)
    def resid(p):
        return (rollout_positions(pos0, p[0:3], 0.0, t, gcfg, spin_rad_s=p[3:6]) - obs).ravel()
    lb = np.array([-20000] * 3 + [-600] * 3); ub = np.array([20000] * 3 + [600] * 3)
    r = least_squares(resid, np.concatenate([v0_kf, np.zeros(3)]), bounds=(lb, ub),
                      loss="soft_l1", f_scale=20.0, max_nfev=80)
    fit = rollout_positions(pos0, r.x[0:3], 0.0, t, gcfg, spin_rad_s=r.x[3:6])
    return float(np.sqrt(np.mean(np.sum((fit - obs) ** 2, axis=1)))) / 10.0  # cm


def fit_dragbounce(t, obs, pos0, v0_kf):
    base = default_config()
    d0 = base.air_drag_coeff
    def cfg_of(drag, az, axy):
        return replace(base, magnus_coeff=0.0, air_drag_coeff=float(drag),
                       bounce_alpha_z=float(az), bounce_alpha_xy=float(axy))
    def resid(p):
        cfg = cfg_of(p[3], p[4], p[5])
        return (rollout_positions(pos0, p[0:3], 0.0, t, cfg, spin_rad_s=None) - obs).ravel()
    # bounds: v0 generous; drag in [0, 5e-4]; alpha_z (restitution) [0.5,1.0]; alpha_xy [0.3,1.0]
    lb = np.array([-20000, -20000, -20000, 0.0, 0.5, 0.3])
    ub = np.array([20000, 20000, 20000, 5e-4, 1.0, 1.0])
    x0 = np.array([v0_kf[0], v0_kf[1], v0_kf[2], d0, base.bounce_alpha_z, base.bounce_alpha_xy])
    r = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=20.0, max_nfev=120)
    cfg = cfg_of(r.x[3], r.x[4], r.x[5])
    fit = rollout_positions(pos0, r.x[0:3], 0.0, t, cfg, spin_rad_s=None)
    rmse = float(np.sqrt(np.mean(np.sum((fit - obs) ** 2, axis=1)))) / 10.0
    return rmse, r.x[3], r.x[4], r.x[5]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    serves = load_dir(args.data_dir)[: args.n]
    A, B = [], []
    for s in serves:
        o = _obs(s)
        if o is None:
            continue
        t, obs, pos0, v0_kf = o
        A.append(fit_spin(t, obs, pos0, v0_kf))
        B.append(fit_dragbounce(t, obs, pos0, v0_kf)[0])
    A = np.array(A); B = np.array(B)
    print(f"\n=== EXPERIMENT 3: spin vs drag/bounce (n={len(A)}) full-path reproduction RMSE (cm) ===")
    print(f"  A) v0+spin            : median={np.median(A):5.2f}  mean={A.mean():5.2f}  p90={np.percentile(A,90):5.2f}")
    print(f"  B) v0+drag/bounce     : median={np.median(B):5.2f}  mean={B.mean():5.2f}  p90={np.percentile(B,90):5.2f}")
    print(f"  B within 1cm of A: {(B <= A + 1.0).mean()*100:.0f}%   B<=3cm: {(B<=3).mean()*100:.0f}%")
    verdict = "drag/bounce EXPLAINS it (spin not needed)" if np.median(B) <= np.median(A) + 1.5 \
        else "spin NEEDED (symmetric drag/bounce cannot reproduce the curvature)"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
