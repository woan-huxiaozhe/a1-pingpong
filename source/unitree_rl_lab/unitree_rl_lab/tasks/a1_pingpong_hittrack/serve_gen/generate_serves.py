"""Main generator: sample faithful serves, run them through the vendored KF, emit recording-schema
CSVs that bake_hittrack_references.py consumes UNCHANGED.

Pipeline per serve:
  1. sample (pos0, v0, drag/bounce) from the KDE mixture over the real fitted serves (a draw = one
     real serve + small jitter, so it stays a near-neighbour of a serve that reproduces its real
     trajectory to ~2 cm -- no off-manifold drift); ``--bandwidth`` scales the jitter, ``--interp-frac``
     optionally fills gaps between neighbouring real serves;
  2. roll the drag/bounce physics (Magnus OFF, per-serve air_drag/bounce) to a clean ball trajectory on
     a 300 Hz grid, until it crosses the bake plane; reject if it never reaches / leaves the workspace;
  3. add calibrated mocap position noise;
  4. run ServeGatedKalman (magnus OFF, tuned params) on the noisy mocap -> filtered state + valid flag
     (reproduces the real serve-onset velocity ramp), and predict-to-(deploy plane) each tracking
     frame -> KF_pred;
  5. write rows in the exact recording column layout (meters).

Then: python bake_hittrack_references.py --data-dir <out> --out hittrack_references_synth.npz \
          --hit-plane-x -1.44   (same plane env_cfg.HIT_PLANE_X uses)

Usage:
  python generate_serves.py --fit fitted_serves.npz --data-dir /data/PPO-pingpong/data/0629_RL_traj \
      --n 2000 --out-dir /data/PPO-pingpong/data/synth_serves --bandwidth 1.0
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

    # physical reachability gate on the crossing (recording frame; bake adds +0.73 z-offset to reach
    # the sim reach box z in [0.7,1.5], y in [-0.6,0.6]). Rejects Gaussian-tail samples that would put
    # the ball metres in the air -- the env's baked reset samples ALL serves regardless of the bake
    # `reachable` flag, so unphysical serves must be dropped here, not just flagged.
    y_c, z_c = cr[0], cr[1]  # recording-frame meters
    if not (-0.6 <= y_c <= 0.6 and 0.7 <= z_c + 0.73 <= 1.5):
        return None

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


def _load_replay_latents(fit_path, ids):
    """Per-serve fitted latents for exactly the serve_ids in `ids` (order = ids),
    skipping any id absent from the fit npz. Used by --replay to reproduce held-out
    serves from their OWN fit (no KDE, no jitter)."""
    d = np.load(fit_path, allow_pickle=True)
    sid = d["serve_id"].astype(int)
    idx = {int(s): i for i, s in enumerate(sid)}
    out = []
    for want in ids:
        i = idx.get(int(want))
        if i is None:
            continue
        out.append({
            "serve_id": int(want),
            "pos0_mm": np.asarray(d["pos0_mm"][i], dtype=float),
            "v0_mm_s": np.asarray(d["v0_mm_s"][i], dtype=float),
            "drag": float(d["drag"][i]),
            "alpha_z": float(d["alpha_z"][i]),
            "alpha_xy": float(d["alpha_xy"][i]),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default="fitted_serves.npz")
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj", help="real data for noise calibration")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out-dir", default="/data/PPO-pingpong/data/synth_serves")
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="KDE jitter scale (0=replay real serves exactly, 1=densify, >1 widens & risks drift)")
    ap.add_argument("--interp-frac", type=float, default=0.0,
                    help="fraction of draws moved toward a nearest real neighbour (fills gaps; 0=pure jitter)")
    ap.add_argument("--fps", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serves-per-file", type=int, default=200)
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="fraction of real serves held OUT of the KDE anchors (validate on unseen serves)")
    ap.add_argument("--holdout-seed", type=int, default=0, help="seed for the deterministic holdout split")
    ap.add_argument("--holdout-out", default=None, help="write held-out serve_ids here (json) for eval baking")
    ap.add_argument("--replay", action="store_true",
                    help="reproduce EXACTLY the held-out serves from their own fit (eval set); no KDE/jitter")
    ap.add_argument("--replay-ids", default=None,
                    help="holdout json to replay (default: --holdout-out path)")
    args = ap.parse_args()

    if args.replay:
        import json
        ids_path = args.replay_ids or args.holdout_out
        if not ids_path:
            raise SystemExit("--replay needs --replay-ids (or --holdout-out) pointing at a holdout json")
        with open(ids_path) as f:
            ids = json.load(f)["holdout_ids"]
        latents = _load_replay_latents(args.fit, ids)
        nm = calibrate(load_dir(args.data_dir))
        rng = np.random.default_rng(args.seed)
        kf_cfg = default_config()
        os.makedirs(args.out_dir, exist_ok=True)
        rows_all = []
        kept = 0
        for lat in latents:
            gcfg = ball_config(lat["drag"], lat["alpha_z"], lat["alpha_xy"])
            rows = _gen_one(lat["pos0_mm"], lat["v0_mm_s"], gcfg, lat["serve_id"], nm, rng, kf_cfg,
                            args.fps, horizon_s=1.6, post_margin_s=0.12)
            if rows is None:
                continue
            rows_all.extend(rows)
            kept += 1
        path = os.path.join(args.out_dir, "synth_000.csv")
        with open(path, "w") as f:
            f.write(HEADER + "\n")
            for r in rows_all:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"replay: emitted {kept}/{len(latents)} held-out serves -> {args.out_dir}")
        print(f"next: python bake_hittrack_references.py --data-dir {args.out_dir} "
              f"--out hittrack_references_eval_real.npz --hit-plane-x {BAKE_PLANE_X}")
        return

    holdout = None
    if args.holdout_frac > 0.0:
        import json
        ids = np.load(args.fit, allow_pickle=True)["serve_id"].astype(int)
        k = int(round(len(ids) * args.holdout_frac))
        holdout = sorted(int(x) for x in np.random.default_rng(args.holdout_seed).choice(ids, size=k, replace=False))
        print(f"holdout {len(holdout)}/{len(ids)} serves (seed={args.holdout_seed}): {holdout}")
        if args.holdout_out:
            with open(args.holdout_out, "w") as f:
                json.dump({"holdout_ids": holdout, "seed": args.holdout_seed, "frac": args.holdout_frac}, f)
    dist = fit_dist(args.fit, holdout_ids=holdout)
    nm = calibrate(load_dir(args.data_dir))
    rng = np.random.default_rng(args.seed)
    kf_cfg = default_config()
    pos0, v0, drag, alpha_z, alpha_xy = dist.sample(args.n, args.bandwidth, rng, interp_frac=args.interp_frac)

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
    print(f"generated {kept}/{args.n} serves (bandwidth={args.bandwidth}, interp_frac={args.interp_frac}) -> {args.out_dir}")
    print(f"next: python bake_hittrack_references.py --data-dir {args.out_dir} "
          f"--out hittrack_references_synth.npz --hit-plane-x {BAKE_PLANE_X}")


if __name__ == "__main__":
    main()
