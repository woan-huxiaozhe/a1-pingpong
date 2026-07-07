"""Main generator: sample faithful serves, run them through the vendored KF, emit recording-schema
CSVs that bake_hittrack_references.py consumes UNCHANGED.

Pipeline per serve:
  1. sample (pos0, v0, spin) from the fitted serve-latent distribution (envelope_scale knob);
  2. roll the spin-aware physics (Magnus on) to a clean ball trajectory on a 300 Hz grid, until it
     crosses the bake plane; reject if it never reaches the plane / leaves the workspace;
  3. add calibrated mocap position noise;
  4. run ServeGatedKalman (magnus OFF, tuned params) on the noisy mocap -> filtered state + valid flag
     (reproduces the real serve-onset velocity ramp), and predict-to-(deploy plane) each tracking
     frame -> KF_pred;
  5. write rows in the exact recording column layout (meters).

Then: python bake_hittrack_references.py --data-dir <out> --out hittrack_references_synth.npz \
          --hit-plane-x -1.44   (same plane env_cfg.HIT_PLANE_X uses)

Usage:
  python generate_serves.py --fit fitted_serves.npz --data-dir /data/PPO-pingpong/data/0629_RL_traj \
      --n 2000 --out-dir /data/PPO-pingpong/data/synth_serves --envelope-scale 1.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from .ball_physics import ball_config, first_downward_crossing, default_config, predict_to_plane, rollout_state_stream
    from .mocap_noise import calibrate
    from .recording_io import load_dir
    from .serve_gated_kalman import ServeGatedKalman
    from .serve_state_dist import fit_dist
except ImportError:
    from ball_physics import ball_config, first_downward_crossing, default_config, predict_to_plane, rollout_state_stream
    from mocap_noise import calibrate
    from recording_io import load_dir
    from serve_gated_kalman import ServeGatedKalman
    from serve_state_dist import fit_dist

MM = 1000.0
DEPLOY_PLANE_X = -1.37   # KF_pred plane (deploy)
BAKE_PLANE_X = -1.44     # env_cfg.HIT_PLANE_X; trajectory must reach past this
HEADER = ("t,serve_id,mocap_x,mocap_y,mocap_z,mocap_vx,mocap_vy,mocap_vz,"
          "KF_x,KF_y,KF_z,KF_vx,KF_vy,KF_vz,KF_pred_y,KF_pred_z,KF_pred_vx,KF_pred_vy,KF_pred_vz,tau,valid")


def _gen_one(pos0_mm, v0_mm, gcfg, serve_id, nm, rng, kf_cfg, fps, horizon_s, post_margin_s):
    t = np.arange(0.0, horizon_s, 1.0 / fps)
    clean = rollout_state_stream(pos0_mm, v0_mm, 0.0, t, gcfg, spin_rad_s=None)  # [T,6] mm
    cr = first_downward_crossing(clean[:, 0] / MM, clean[:, 1] / MM, clean[:, 2] / MM, t, BAKE_PLANE_X)
    if cr is None:
        return None
    t_cross = cr[2]
    n = int(np.searchsorted(t, t_cross + post_margin_s)) + 1
    n = min(n, len(t))
    if n < 10:
        return None
    t = t[:n]; clean = clean[:n]
    noisy_mm = nm.apply(clean[:, 0:3], rng)  # [n,3] mm

    sgk = ServeGatedKalman(kf_cfg)
    sgk.reset_to_hold()
    rows = []
    last_pred = None
    for j in range(n):
        state, gate = sgk.step(noisy_mm[j], float(t[j]))
        kf6 = state[:6]  # mm, mm/s
        valid = gate == "TRACKING"
        pred = None
        if valid:
            pr = predict_to_plane(kf6, DEPLOY_PLANE_X * MM, kf_cfg)
            if pr is not None:
                last_pred = pr
            pred = pr if pr is not None else last_pred
        if pred is None:
            pred = np.array([kf6[1], kf6[2], kf6[3], kf6[4], kf6[5]])  # placeholder pre-lockon
        # emit meters
        m = noisy_mm[j] / MM
        k = kf6 / MM
        p = pred / MM
        rows.append(
            [t[j], serve_id, m[0], m[1], m[2], 0.0, 0.0, 0.0,
             k[0], k[1], k[2], k[3], k[4], k[5],
             p[0], p[1], p[2], p[3], p[4], 0.0, "True" if valid else "False"]
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default="fitted_serves.npz")
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj", help="real data for noise calibration")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out-dir", default="/data/PPO-pingpong/data/synth_serves")
    ap.add_argument("--envelope-scale", type=float, default=1.0)
    ap.add_argument("--magnus-coeff", type=float, default=0.003604)
    ap.add_argument("--fps", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serves-per-file", type=int, default=200)
    args = ap.parse_args()

    dist = fit_dist(args.fit)
    nm = calibrate(load_dir(args.data_dir))
    rng = np.random.default_rng(args.seed)
    kf_cfg = default_config()
    pos0, v0, drag, alpha_z, alpha_xy = dist.sample(args.n, args.envelope_scale, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    kept = 0
    file_rows: list = []
    file_idx = 0

    def flush(idx, rows):
        path = os.path.join(args.out_dir, f"synth_{idx:03d}.csv")
        with open(path, "w") as f:
            f.write(HEADER + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    for k in range(args.n):
        gcfg = ball_config(drag[k], alpha_z[k], alpha_xy[k])  # per-serve drag/bounce, magnus off
        rows = _gen_one(pos0[k], v0[k], gcfg, kept, nm, rng, kf_cfg,
                        args.fps, horizon_s=1.6, post_margin_s=0.12)
        if rows is None:
            continue
        file_rows.extend(rows)
        kept += 1
        if kept % args.serves_per_file == 0:
            flush(file_idx, file_rows); file_rows = []; file_idx += 1
    if file_rows:
        flush(file_idx, file_rows)
    print(f"generated {kept}/{args.n} serves (envelope_scale={args.envelope_scale}) -> {args.out_dir}")
    print(f"next: python bake_hittrack_references.py --data-dir {args.out_dir} "
          f"--out hittrack_references_synth.npz --hit-plane-x {BAKE_PLANE_X}")


if __name__ == "__main__":
    main()
