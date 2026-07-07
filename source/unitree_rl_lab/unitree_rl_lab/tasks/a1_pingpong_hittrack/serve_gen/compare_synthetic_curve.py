"""CRUX EXPERIMENT: does pure-physics + calibrated-noise generation reproduce the REAL
KF_pred-error-vs-tau curve, or fall short (=> process mismatch / spin must be injected)?

For each real serve we take its lock-on KF state, roll the vendored physics forward on the same
timestamps to make a CLEAN synthetic trajectory, add calibrated mocap noise, run it back through the
KF, and measure the synthetic KF_pred error vs the synthetic realized crossing -- bucketed by tau.
Printed side-by-side with the REAL curve. If synthetic << real, the deficit is the unmodeled
dynamics (spin) the generator must add.

Usage: python compare_synthetic_curve.py --data-dir /data/PPO-pingpong/data/0629_RL_traj
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import default_config, first_downward_crossing, gen_config, predict_to_plane, rollout_state_stream
    from .kalman import Kalman
    from .mocap_noise import calibrate
    from .recording_io import Serve, load_dir
except ImportError:
    from ball_physics import default_config, first_downward_crossing, gen_config, predict_to_plane, rollout_state_stream
    from kalman import Kalman
    from mocap_noise import calibrate
    from recording_io import Serve, load_dir

PLANE_X = -1.37
MM = 1000.0
EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]


def _bucket(tau, err_cm):
    tau = np.asarray(tau); err_cm = np.asarray(err_cm)
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        w = (tau >= a) & (tau < b)
        if w.sum():
            rows.append((a, b, int(w.sum()), float(np.median(err_cm[w])), float(err_cm[w].mean())))
    return rows


def real_curve(serves):
    taus, errs = [], []
    for s in serves:
        cr = first_downward_crossing(s.kf[:, 0], s.kf[:, 1], s.kf[:, 2], s.t, PLANE_X)
        if cr is None:
            continue
        yz = np.array(cr[:2]); tc = cr[2]
        tau = tc - s.t
        m = (tau > 0) & s.valid
        if m.sum() < 3:
            continue
        taus.append(tau[m]); errs.append(np.linalg.norm(s.kf_pred[m, 0:2] - yz[None, :], axis=1) * 100)
    return np.concatenate(taus), np.concatenate(errs)


def synth_curve(serves, nm, seed=0, pred_every=5, spin_std=0.0, magnus_coeff=0.003604):
    gen_cfg = gen_config(magnus_coeff) if spin_std > 0 else default_config()
    kf_cfg = default_config()  # KF always predicts magnus-off (it cannot observe spin)
    rng = np.random.default_rng(seed)
    taus, errs = [], []
    for s in serves:
        vi = np.where(s.valid)[0]
        if vi.size < 8:
            continue
        i0 = int(vi[0])
        t = s.t[i0:] - s.t[i0]
        pos0 = s.kf[i0, 0:3] * MM
        vel0 = s.kf[i0, 3:6] * MM
        spin = rng.normal(0.0, spin_std, size=3) if spin_std > 0 else None
        clean = rollout_state_stream(pos0, vel0, 0.0, t, gen_cfg, spin_rad_s=spin)  # [N,6] mm
        # synthetic realized crossing (ground truth of the synthetic ball)
        cr = first_downward_crossing(clean[:, 0] / MM, clean[:, 1] / MM, clean[:, 2] / MM, t, PLANE_X)
        if cr is None:
            continue
        yz = np.array(cr[:2]); tc = cr[2]
        noisy = nm.apply(clean[:, 0:3], rng)  # mm
        # run KF on the synthetic noisy mocap (magnus-off, like deployment)
        kf = Kalman(kf_cfg)
        kf.initialize(noisy[0], float(t[0]))
        for j in range(1, len(t)):
            kf.predict(float(t[j]))
            kf.update(noisy[j])
            if j % pred_every:
                continue
            tau = tc - t[j]
            if tau <= 0:
                continue
            pr = predict_to_plane(kf.get_state()[:6], PLANE_X * MM, kf_cfg)
            if pr is None:
                continue
            err_cm = np.linalg.norm(pr[0:2] / MM - yz) * 100
            taus.append(tau); errs.append(err_cm)
    return np.array(taus), np.array(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--spin-sweep", type=float, nargs="+", default=[0.0],
                    help="isotropic spin std values (rad/s) to sweep")
    ap.add_argument("--magnus-coeff", type=float, default=0.003604)
    args = ap.parse_args()
    serves = load_dir(args.data_dir)
    nm = calibrate(serves)
    print(f"calibrated noise std x/y/z = {nm.std_xyz_mm.round(2)} mm")
    rt, re = real_curve(serves)
    rb = {(a, b): (n, med) for a, b, n, med, mean in _bucket(rt, re)}

    header = f"\n{'tau bucket (s)':16s} {'REAL':>8s}"
    for sp in args.spin_sweep:
        header += f" {('SYN@'+str(int(sp))):>9s}"
    print(header + "   (median cm)")
    syn_bs = []
    for sp in args.spin_sweep:
        st, se = synth_curve(serves, nm, spin_std=sp, magnus_coeff=args.magnus_coeff)
        syn_bs.append({(a, b): med for a, b, n, med, mean in _bucket(st, se)})
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        if (a, b) not in rb:
            continue
        line = f"[{a:.2f},{b:.2f})       {rb[(a,b)][1]:8.2f}"
        for sb in syn_bs:
            line += f" {sb.get((a,b), float('nan')):9.2f}"
        print(line)
    print("\ntarget: pick spin std whose SYN column best matches REAL across tau buckets")


if __name__ == "__main__":
    main()
