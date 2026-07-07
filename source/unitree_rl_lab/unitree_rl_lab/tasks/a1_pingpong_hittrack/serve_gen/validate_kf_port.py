"""V1 (KF PORT FIDELITY): replay real mocap through the vendored Python KF and compare to the
recorded ``KF_*`` (filtered state) and ``KF_pred_*`` (forward roll-out to the deploy hit plane)
columns that the C++ node logged. If the Python port matches the recordings, "run the same KF
offline" is justified; if not, the port must be tuned before any synthetic data is trusted.

We feed raw ``mocap`` positions (the KF observes position only) with their timestamps into
``ServeGatedKalman`` (rally already segmented by ``serve_id`` -> one HOLD->TRACKING cycle per serve),
then per TRACKING frame compare:
  * filtered state  : my [x,y,z,vx,vy,vz]  vs recorded KF_*
  * hit-plane pred  : my predict-to(x=-1.37) [y,z,vx,vy,vz]  vs recorded KF_pred_*

Usage:
    python validate_kf_port.py --data-dir /data/PPO-pingpong/data/0629_RL_traj
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import default_config, predict_to_plane
    from .recording_io import Serve, load_dir
    from .serve_gated_kalman import ServeGatedKalman
except ImportError:
    from ball_physics import default_config, predict_to_plane
    from recording_io import Serve, load_dir
    from serve_gated_kalman import ServeGatedKalman

DEPLOY_PLANE_X = -1.37     # m, where KF_pred is evaluated (bake docstring)
MM = 1000.0


def replay_serve(s: Serve, pred_every: int = 3) -> dict | None:
    sgk = ServeGatedKalman(default_config())
    sgk.reset_to_hold()

    my_kf = np.full((len(s.t), 6), np.nan)
    tracking = np.zeros(len(s.t), dtype=bool)
    for j in range(len(s.t)):
        state, gate = sgk.step(s.mocap[j, 0:3] * MM, float(s.t[j]))
        my_kf[j] = state[:6]
        tracking[j] = (gate == "TRACKING")

    # compare only where BOTH my gate is TRACKING and the recording flagged valid
    w = tracking & s.valid
    if w.sum() < 5:
        return None
    my = my_kf[w] / MM   # -> m, m/s
    rec = s.kf[w]        # m, m/s
    dpos = my[:, 0:3] - rec[:, 0:3]
    dvel = my[:, 3:6] - rec[:, 3:6]

    # KF_pred fidelity: predict-to-plane from my filtered state, subsampled frames
    pred_err_yz = []
    pred_err_v = []
    widx = np.where(w)[0]
    for j in widx[::pred_every]:
        pr = predict_to_plane(my_kf[j], DEPLOY_PLANE_X * MM)
        if pr is None:
            continue
        pr = pr / MM  # y,z,vx,vy,vz in m,m/s
        rec_pred = s.kf_pred[j]
        pred_err_yz.append(np.linalg.norm(pr[0:2] - rec_pred[0:2]))
        pred_err_v.append(np.linalg.norm(pr[2:5] - rec_pred[2:5]))

    return {
        "serve_id": s.serve_id,
        "n": int(w.sum()),
        "filt_pos_rmse": float(np.sqrt(np.mean(np.sum(dpos ** 2, axis=1)))),
        "filt_vel_rmse": float(np.sqrt(np.mean(np.sum(dvel ** 2, axis=1)))),
        "pred_yz_med": float(np.median(pred_err_yz)) if pred_err_yz else np.nan,
        "pred_v_med": float(np.median(pred_err_v)) if pred_err_v else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--pred-every", type=int, default=3)
    args = ap.parse_args()

    serves = load_dir(args.data_dir)
    rows = [r for s in serves for r in [replay_serve(s, args.pred_every)] if r is not None]
    print(f"\n=== KF port fidelity (replay real mocap -> vendored Python KF) ===")
    print(f"serves total={len(serves)}  evaluated={len(rows)}")
    if not rows:
        print("  (no serves reached TRACKING; check gating params)")
        return
    fp = np.array([r["filt_pos_rmse"] for r in rows]) * 100  # cm
    fv = np.array([r["filt_vel_rmse"] for r in rows])         # m/s
    py = np.array([r["pred_yz_med"] for r in rows]) * 100     # cm
    pv = np.array([r["pred_v_med"] for r in rows])            # m/s

    def stat(name, a, unit):
        a = a[np.isfinite(a)]
        print(f"  {name:26s} mean={np.mean(a):6.2f}  median={np.median(a):6.2f}  "
              f"p90={np.percentile(a,90):6.2f}  max={np.max(a):6.2f}  {unit}")

    print("-- FILTERED state: my KF vs recorded KF_* (should be ~0 if port faithful) --")
    stat("filtered pos RMSE", fp, "cm")
    stat("filtered vel RMSE", fv, "m/s")
    print("-- HIT-PLANE PRED: my predict->(-1.37) vs recorded KF_pred_* --")
    stat("pred (y,z) err (median/serve)", py, "cm")
    stat("pred vel err (median/serve)", pv, "m/s")


if __name__ == "__main__":
    main()
