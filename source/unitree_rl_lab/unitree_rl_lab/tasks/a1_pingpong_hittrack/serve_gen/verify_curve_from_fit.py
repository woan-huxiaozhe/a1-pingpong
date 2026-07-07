"""VERIFY the unifying hypothesis: faithful (fitted spin+v0) synthetic serves, run through the KF
(magnus OFF, like deployment), reproduce the REAL KF_pred-error-vs-tau curve WITHOUT any manual
noise injection -- because the fitted-spin ball deviates from the KF's no-spin model exactly the way
real balls do.

Loads fitted_serves.npz (from serve_fit.py --save), regenerates each serve's faithful trajectory on
a 300 Hz grid, adds calibrated mocap noise, runs the KF, measures synthetic KF_pred error, and prints
it next to the real curve.

Usage: python verify_curve_from_fit.py --fit fitted_serves.npz --data-dir /data/PPO-pingpong/data/0629_RL_traj
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import ball_config, first_downward_crossing, default_config, predict_to_plane, rollout_state_stream
    from .compare_synthetic_curve import EDGES, _bucket, real_curve
    from .kalman import Kalman
    from .mocap_noise import calibrate
    from .recording_io import load_dir
except ImportError:
    from ball_physics import ball_config, first_downward_crossing, default_config, predict_to_plane, rollout_state_stream
    from compare_synthetic_curve import EDGES, _bucket, real_curve
    from kalman import Kalman
    from mocap_noise import calibrate
    from recording_io import load_dir

PLANE_X = -1.37
MM = 1000.0
FS = 300.0  # synthetic sampling rate (Hz), matches recordings


def synth_curve_from_fit(fit, nm, seed=0, pred_every=5):
    kf_cfg = default_config()
    rng = np.random.default_rng(seed)
    pos0 = fit["pos0_mm"]; v0 = fit["v0_mm_s"]; tau0 = fit["tau_lockon_s"]
    drag = fit["drag"]; az = fit["alpha_z"]; axy = fit["alpha_xy"]
    taus, errs = [], []
    for k in range(len(pos0)):
        t = np.arange(0.0, float(tau0[k]) + 0.15, 1.0 / FS)
        if len(t) < 8:
            continue
        gcfg = ball_config(drag[k], az[k], axy[k])
        clean = rollout_state_stream(pos0[k], v0[k], 0.0, t, gcfg, spin_rad_s=None)
        cr = first_downward_crossing(clean[:, 0] / MM, clean[:, 1] / MM, clean[:, 2] / MM, t, PLANE_X)
        if cr is None:
            continue
        yz = np.array(cr[:2]); tc = cr[2]
        noisy = nm.apply(clean[:, 0:3], rng)
        kf = Kalman(kf_cfg)
        kf.initialize(noisy[0], float(t[0]))
        for j in range(1, len(t)):
            kf.predict(float(t[j])); kf.update(noisy[j])
            if j % pred_every:
                continue
            tau = tc - t[j]
            if tau <= 0:
                continue
            pr = predict_to_plane(kf.get_state()[:6], PLANE_X * MM, kf_cfg)
            if pr is None:
                continue
            taus.append(tau); errs.append(np.linalg.norm(pr[0:2] / MM - yz) * 100)
    return np.array(taus), np.array(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default="fitted_serves.npz")
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    args = ap.parse_args()
    fit = np.load(args.fit, allow_pickle=True)
    serves = load_dir(args.data_dir)
    nm = calibrate(serves)
    rt, re = real_curve(serves)
    st, se = synth_curve_from_fit(fit, nm)
    rb = {(a, b): med for a, b, n, med, mean in _bucket(rt, re)}
    sb = {(a, b): med for a, b, n, med, mean in _bucket(st, se)}
    print(f"\n=== faithful-ball synthetic KF_pred curve vs REAL (n_fit={len(fit['pos0_mm'])}) ===")
    print(f"{'tau bucket (s)':16s} {'REAL cm':>9s} {'SYN-fit cm':>11s} {'diff cm':>9s}")
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        if (a, b) not in rb:
            continue
        rm = rb[(a, b)]; sm = sb.get((a, b), float('nan'))
        print(f"[{a:.2f},{b:.2f})       {rm:9.2f} {sm:11.2f} {rm - sm:9.2f}")
    print("\nsmall diff across buckets => faithful ball model reproduces observation noise naturally")


if __name__ == "__main__":
    main()
