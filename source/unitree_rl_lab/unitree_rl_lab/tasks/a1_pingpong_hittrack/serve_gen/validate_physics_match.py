"""VALIDATION (sim2real-gap gate): does the vendored physics reproduce real ball flight?

For each real serve we take the KF state (position + velocity) at a chosen ``x0`` in the
[0.7, 1.4] m window, "release" it into the vendored physics rollout, integrate forward THROUGH the
table bounce to the hit plane x = -1.44 m, and compare the produced trajectory to the recorded one.

If the produced (y, z) at the hit-plane crossing matches the real crossing to within ~cm across
serves, the physics has no sim2real gap and synthetic serves rolled the same way are trustworthy
training data. This is the criterion the user set before any generation is believed.

Usage:
    python validate_physics_match.py --data-dir /data/PPO-pingpong/data/0629_RL_traj [--x0 1.3]
    python validate_physics_match.py --data-dir ... --sweep 1.4 1.2 1.0 0.8
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import first_downward_crossing, rollout_positions
    from .recording_io import Serve, load_dir
except ImportError:
    from ball_physics import first_downward_crossing, rollout_positions
    from recording_io import Serve, load_dir

HIT_PLANE_X = -1.44
X0_LO, X0_HI = 0.7, 1.4
MM = 1000.0


def _release_index(kf_x: np.ndarray, x0: float) -> int | None:
    """First index where the (descending) KF x-track enters at/below ``x0``; None if the serve
    never sits inside the [X0_LO, X0_HI] window there (held ball / too-late lock-on / no approach)."""
    below = np.where(kf_x <= x0)[0]
    if below.size == 0:
        return None
    i0 = int(below[0])
    if not (X0_LO <= kf_x[i0] <= X0_HI):
        return None
    return i0


def eval_serve(s: Serve, x0: float) -> dict | None:
    kf_x = s.kf[:, 0]
    if kf_x.max() < x0 or kf_x.min() > HIT_PLANE_X:  # must span x0 -> hit plane
        return None
    i0 = _release_index(kf_x, x0)
    if i0 is None or len(s.t) - i0 < 5:
        return None

    t = s.t[i0:] - s.t[i0]
    pos_mm = s.kf[i0, 0:3] * MM
    vel_mm = s.kf[i0, 3:6] * MM
    rolled_m = rollout_positions(pos_mm, vel_mm, 0.0, t) / MM

    real_kf = s.kf[i0:, 0:3]
    real_moc = s.mocap[i0:, 0:3]

    cross_roll = first_downward_crossing(rolled_m[:, 0], rolled_m[:, 1], rolled_m[:, 2], t, HIT_PLANE_X)
    cross_real = first_downward_crossing(real_kf[:, 0], real_kf[:, 1], real_kf[:, 2], t, HIT_PLANE_X)
    if cross_roll is None or cross_real is None:
        return None
    yz_roll = np.array(cross_roll[:2])
    yz_real = np.array(cross_real[:2])
    t_cross_real = cross_real[2]

    # path error over the window up to the real crossing (rolled shares timestamps with real)
    w = t <= t_cross_real
    dif_kf = rolled_m[w] - real_kf[w]
    dif_moc = rolled_m[w] - real_moc[w]
    return {
        "serve_id": s.serve_id,
        "n": int(w.sum()),
        "endpoint_y": float(abs(yz_roll[0] - yz_real[0])),
        "endpoint_z": float(abs(yz_roll[1] - yz_real[1])),
        "endpoint_yz": float(np.linalg.norm(yz_roll - yz_real)),
        "path_rmse_kf": float(np.sqrt(np.mean(np.sum(dif_kf ** 2, axis=1)))),
        "path_rmse_moc": float(np.sqrt(np.mean(np.sum(dif_moc ** 2, axis=1)))),
        "flight_s": float(t_cross_real),
    }


def run(data_dir: str, x0: float) -> None:
    serves = load_dir(data_dir)
    rows = [r for s in serves for r in [eval_serve(s, x0)] if r is not None]
    print(f"\n=== x0={x0:.2f} m release -> rollout to x={HIT_PLANE_X} m ===")
    print(f"serves total={len(serves)}  evaluated={len(rows)}")
    if not rows:
        print("  (no serves span the window; try a smaller x0)")
        return
    ey = np.array([r["endpoint_y"] for r in rows]) * 100  # cm
    ez = np.array([r["endpoint_z"] for r in rows]) * 100
    eyz = np.array([r["endpoint_yz"] for r in rows]) * 100
    prk = np.array([r["path_rmse_kf"] for r in rows]) * 100
    fl = np.array([r["flight_s"] for r in rows])

    def stat(name, a, unit):
        print(f"  {name:22s} mean={a.mean():6.2f}  median={np.median(a):6.2f}  "
              f"p90={np.percentile(a,90):6.2f}  max={a.max():6.2f}  {unit}")

    stat("endpoint |Δy|", ey, "cm")
    stat("endpoint |Δz|", ez, "cm")
    stat("endpoint |Δ(y,z)|", eyz, "cm")
    stat("path RMSE vs KF", prk, "cm")
    stat("flight time", fl, "s")
    # success-threshold context: bake/env SUCCESS_POS = 0.05 m = 5 cm
    frac_5cm = float((eyz <= 5.0).mean())
    frac_2cm = float((eyz <= 2.0).mean())
    print(f"  endpoint within 2cm: {frac_2cm*100:5.1f}%   within 5cm (SUCCESS_POS): {frac_5cm*100:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--x0", type=float, default=1.3, help="release x (m) inside [0.7,1.4]")
    ap.add_argument("--sweep", type=float, nargs="+", default=None,
                    help="evaluate several release x0 values instead of one")
    args = ap.parse_args()
    for x0 in (args.sweep if args.sweep else [args.x0]):
        run(args.data_dir, x0)


if __name__ == "__main__":
    main()
