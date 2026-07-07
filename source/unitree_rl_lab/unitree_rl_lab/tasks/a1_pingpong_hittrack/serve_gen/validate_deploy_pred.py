"""Which prediction method produced the recordings' KF_pred? Compare the recorded KF_pred_* column
against the reproduced deployment prediction with velocity_fit OFF vs ON, per real serve. The method
whose reproduction sits ~0 from the recording is the one deployment used; that is the method the
synthetic KF_pred must also use (removing the prediction-method confound).

Usage: python validate_deploy_pred.py --data-dir /data/PPO-pingpong/data/0629_RL_traj
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import default_config
    from .deploy_predict import deploy_predict_to_plane
    from .kalman import Kalman
    from .recording_io import load_dir
except ImportError:
    from ball_physics import default_config
    from deploy_predict import deploy_predict_to_plane
    from kalman import Kalman
    from recording_io import load_dir

DEPLOY_PLANE_X = -1.37
MM = 1000.0


def eval_serve(s, cfg, pred_every=4):
    # replay KF over real mocap, collect filtered state6 + positions
    kf = Kalman(cfg)
    kf.initialize(s.mocap[0, 0:3] * MM, float(s.t[0]))
    state6 = np.full((len(s.t), 6), np.nan)
    state6[0] = kf.get_state()[:6]
    for j in range(1, len(s.t)):
        kf.predict(float(s.t[j]))
        kf.update(s.mocap[j, 0:3] * MM)
        state6[j] = kf.get_state()[:6]
    filt_pos = state6[:, 0:3]

    off, on = [], []
    widx = np.where(s.valid)[0]
    for j in widx[::pred_every]:
        rec = s.kf_pred[j, 0:2]  # recorded KF_pred (y,z), m
        p_off = deploy_predict_to_plane(state6[j], s.t, filt_pos, j, DEPLOY_PLANE_X * MM, cfg,
                                        velocity_fit=False)
        p_on = deploy_predict_to_plane(state6[j], s.t, filt_pos, j, DEPLOY_PLANE_X * MM, cfg,
                                       velocity_fit=True, window=8, degree=2, min_samples=5, blend=1.0)
        if p_off is not None:
            off.append(np.linalg.norm(p_off[0:2] / MM - rec))
        if p_on is not None:
            on.append(np.linalg.norm(p_on[0:2] / MM - rec))
    if not off:
        return None
    return float(np.median(off)) * 100, float(np.median(on) if on else np.nan) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    args = ap.parse_args()
    serves = load_dir(args.data_dir)
    cfg = default_config()
    rows = [r for s in serves for r in [eval_serve(s, cfg)] if r is not None]
    off = np.array([r[0] for r in rows])
    on = np.array([r[1] for r in rows])
    on = on[np.isfinite(on)]
    print("\n=== reproduced deploy-predict vs recorded KF_pred (per-serve median, cm) ===")
    print(f"serves={len(rows)}")
    print(f"  velocity_fit OFF (plain KF rollout): mean={off.mean():.2f}  median={np.median(off):.2f}  p90={np.percentile(off,90):.2f}")
    print(f"  velocity_fit ON  (deg2, win8)       : mean={on.mean():.2f}  median={np.median(on):.2f}  p90={np.percentile(on,90):.2f}")
    print("  -> the smaller one is the method the recordings used")


if __name__ == "__main__":
    main()
