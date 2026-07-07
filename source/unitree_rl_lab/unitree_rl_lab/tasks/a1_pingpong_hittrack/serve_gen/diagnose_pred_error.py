"""DIAGNOSTIC: the REAL system's forward-prediction error vs time-to-hit (tau).

Uses recorded columns ONLY (no synthetic physics). For each serve we take the realized hit-plane
crossing (from the smooth KF track at x=-1.37) as ground truth, then for every earlier frame measure
how far that frame's recorded ``KF_pred`` (y,z) sat from the eventual crossing, bucketed by
``tau = t_cross - t``. This curve is the target the generator's synthetic KF_pred error must
reproduce: if pure-physics generation makes KF_pred far more accurate than this, the training data
has a sim2real gap in the very reference the policy observes.

Usage: python diagnose_pred_error.py --data-dir /data/PPO-pingpong/data/0629_RL_traj
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import first_downward_crossing
    from .recording_io import Serve, load_dir
except ImportError:
    from ball_physics import first_downward_crossing
    from recording_io import Serve, load_dir

DEPLOY_PLANE_X = -1.37


def serve_pred_errors(s: Serve):
    cr = first_downward_crossing(s.kf[:, 0], s.kf[:, 1], s.kf[:, 2], s.t, DEPLOY_PLANE_X)
    if cr is None:
        return None
    yz_real = np.array(cr[:2])
    t_cross = cr[2]
    tau = t_cross - s.t
    m = (tau > 0.0) & s.valid  # frames before the crossing, KF-converged
    if m.sum() < 3:
        return None
    err = np.linalg.norm(s.kf_pred[m, 0:2] - yz_real[None, :], axis=1)  # m
    return tau[m], err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    args = ap.parse_args()
    serves = load_dir(args.data_dir)
    taus, errs = [], []
    for s in serves:
        r = serve_pred_errors(s)
        if r is not None:
            taus.append(r[0]); errs.append(r[1])
    tau = np.concatenate(taus); err = np.concatenate(errs) * 100  # cm
    print(f"\n=== REAL KF_pred (y,z) error vs realized crossing, by tau ===")
    print(f"serves={len(taus)}  frames={len(tau)}")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]
    print(f"  {'tau bucket (s)':16s} {'n':>6s} {'mean cm':>9s} {'median cm':>10s} {'p90 cm':>8s}")
    for a, b in zip(edges[:-1], edges[1:]):
        w = (tau >= a) & (tau < b)
        if w.sum() == 0:
            continue
        e = err[w]
        print(f"  [{a:.2f},{b:.2f})      {w.sum():6d} {e.mean():9.2f} {np.median(e):10.2f} {np.percentile(e,90):8.2f}")


if __name__ == "__main__":
    main()
