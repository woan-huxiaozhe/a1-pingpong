"""Build a filtered SAC ball-reset state table from real mocap trajectories.

The output is a ``serve_states.npz``-style table consumed by
``table_tennis.mdp.events.launch_ball`` via ``serve_states_path``.  Each row is
``[x, y, z, vx, vy, vz]`` in simulation/world coordinates.

Default intent for the current A1 SAC task:

* sample around the measured state at x=+1.0 m;
* keep both full-serve pre-bounce states and post-bounce incoming states by default;
* reject samples that do not clear the net, do not bounce once on the robot side,
  or arrive outside the reachable hit window at ``robot_x=-1.47``.

Examples:

  python3 source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/create_serve_states.py \
      --data-dir /home/woan/kalman_filter_pingpong/data/0617_traj_data \
      --num-states 5000

  python3 source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/create_serve_states.py \
      --data-dir /path/to/day1 /path/to/day2 \
      --max-opponent-bounces 1 \
      --sampler kde --kde-bw 0.8 --num-states 10000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = "/home/woan/kalman_filter_pingpong/data/0617_traj_data"
DEFAULT_OUT = os.path.join(THIS_DIR, "serve_states_x1.npz")


@dataclass(frozen=True)
class RolloutCfg:
    table_z: float = 0.76
    table_x_half: float = 1.37
    table_y_half: float = 0.7625
    net_x: float = 0.0
    net_min_z: float = 0.9525
    robot_x: float = -1.47
    robot_side: int = -1
    gravity: float = 9.81
    drag_k: float = 0.0
    bounce_ch: float = 0.85
    bounce_cv: float = 0.90
    dt: float = 0.0015
    t_max: float = 2.0


def load_vrpn(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                row = [float(parts[i]) for i in range(min(len(parts), 7))]
            except ValueError:
                continue
            if np.isfinite(row).all():
                rows.append(row)
    if not rows:
        raise ValueError(f"no valid rows in {path}")

    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0] - arr[0, 0]
    pos = arr[:, 1:4]
    vel = arr[:, 4:7] if arr.shape[1] >= 7 else None
    return t, pos, vel


def state_at_x(
    t: np.ndarray,
    pos: np.ndarray,
    vel_file: np.ndarray | None,
    target_x: float,
    velocity_source: str,
    vel_window_s: float,
) -> tuple[np.ndarray, float] | None:
    x = pos[:, 0]
    for i in range(len(x) - 1):
        if (x[i] - target_x) >= 0.0 >= (x[i + 1] - target_x):
            dx = x[i] - x[i + 1]
            alpha = (x[i] - target_x) / dx if abs(dx) > 1.0e-9 else 0.0
            t_cross = t[i] + alpha * (t[i + 1] - t[i])
            p_cross = pos[i] + alpha * (pos[i + 1] - pos[i])

            if velocity_source == "file" and vel_file is not None:
                vel = vel_file[i] + alpha * (vel_file[i + 1] - vel_file[i])
            else:
                mask = np.abs(t - t_cross) <= vel_window_s
                if mask.sum() < 5:
                    lo = max(0, i - 4)
                    hi = min(len(t), i + 5)
                    mask = np.zeros(len(t), dtype=bool)
                    mask[lo:hi] = True
                tau = t[mask] - t_cross
                deg = 2 if len(tau) >= 5 else 1
                vel = np.empty(3)
                for axis in range(3):
                    coeff = np.polyfit(tau, pos[mask, axis], deg)
                    vel[axis] = coeff[-2] if deg == 2 else coeff[0]

            if vel[0] >= -0.05:
                continue
            return np.concatenate([p_cross, vel]), float(t_cross)
    return None


def extract_plane_states(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    files = []
    for data_dir in args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(data_dir, "*.txt"))))
    if not files:
        raise SystemExit(f"no .txt trajectory files found in {args.data_dir}")

    states = []
    used = []
    skipped = Counter()
    for path in files:
        try:
            t, pos, vel_file = load_vrpn(path)
        except Exception:
            skipped["load"] += 1
            continue
        birth = state_at_x(t, pos, vel_file, args.birth_x, args.velocity_source, args.vel_window_s)
        if birth is None:
            skipped["no_birth_x"] += 1
            continue
        state, _ = birth
        state[2] += args.z_offset

        if args.source_hit_filter:
            hit = state_at_x(t, pos, vel_file, args.source_filter_robot_x, args.velocity_source, args.vel_window_s)
            if hit is None:
                skipped["no_source_filter_x"] += 1
                continue
            hit_state, _ = hit
            hit_state = hit_state.copy()
            hit_state[2] += args.z_offset
            hit_height = hit_state[2] - args.source_filter_table_z
            if not (args.source_filter_y_range[0] < hit_state[1] < args.source_filter_y_range[1]):
                skipped["source_filter_y"] += 1
                continue
            if not (args.source_filter_height_range[0] < hit_height < args.source_filter_height_range[1]):
                skipped["source_filter_z"] += 1
                continue

        if args.post_bounce_only:
            if state[2] < args.post_bounce_z_min or state[5] < args.post_bounce_vz_min:
                skipped["pre_bounce_branch"] += 1
                continue

        states.append(state)
        used.append(os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)))

    states_arr = np.asarray(states, dtype=float)
    print(
        f"[extract] dirs={len(args.data_dir)} files={len(files)} states={len(states_arr)} skipped={dict(skipped)} "
        f"birth_x={args.birth_x:.3f}"
    )
    if len(states_arr) < 5:
        raise SystemExit("too few usable states; relax gates or check data_dir")
    return states_arr, used


def make_sampler(
    states6: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
):
    features = states6[:, 1:6]
    qlo, qhi = args.clip_percentiles
    clip_lo = np.percentile(features, qlo, axis=0)
    clip_hi = np.percentile(features, qhi, axis=0)
    std = features.std(axis=0, ddof=1)
    jitter = std * args.jitter_frac

    if args.sampler == "kde":
        try:
            from scipy.stats import gaussian_kde
        except Exception as exc:
            raise SystemExit(f"--sampler kde requires scipy: {exc}") from exc

        kde = gaussian_kde(features.T, bw_method=args.kde_bw)

        def sample(n: int) -> np.ndarray:
            drawn = kde.resample(n, seed=rng).T
            return np.clip(drawn, clip_lo, clip_hi)

    elif args.sampler == "gaussian":
        mu = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)

        def sample(n: int) -> np.ndarray:
            drawn = rng.multivariate_normal(mu, cov, size=n)
            return np.clip(drawn, clip_lo, clip_hi)

    else:

        def sample(n: int) -> np.ndarray:
            idx = rng.integers(0, features.shape[0], size=n)
            drawn = features[idx].copy()
            if args.jitter_frac > 0.0:
                drawn += rng.normal(scale=jitter, size=drawn.shape)
            return np.clip(drawn, clip_lo, clip_hi)

    return sample, clip_lo, clip_hi


def own_side(x: float, robot_side: int) -> bool:
    return x * float(robot_side) > 0.0


def rollout(state6: np.ndarray, cfg: RolloutCfg) -> dict:
    p = state6[:3].astype(float).copy()
    v = state6[3:].astype(float).copy()
    t = 0.0
    bounces = []
    net = None

    while t < cfg.t_max:
        speed = np.linalg.norm(v)
        acc = np.array([0.0, 0.0, -cfg.gravity]) - cfg.drag_k * speed * v
        v_next = v + acc * cfg.dt
        p_next = p + v_next * cfg.dt

        if p[0] > cfg.net_x >= p_next[0]:
            alpha = (p[0] - cfg.net_x) / (p[0] - p_next[0]) if abs(p[0] - p_next[0]) > 1.0e-9 else 0.0
            p_net = p + alpha * (p_next - p)
            net = (float(p_net[1]), float(p_net[2]), float(t + alpha * cfg.dt))
            if p_net[2] < cfg.net_min_z:
                return {"reason": "net", "net": net, "bounces": bounces}

        if p_next[2] <= cfg.table_z and v_next[2] < 0.0:
            on_table = (
                -cfg.table_x_half <= p_next[0] <= cfg.table_x_half
                and abs(p_next[1]) <= cfg.table_y_half
            )
            if not on_table:
                return {
                    "reason": "out_landing",
                    "net": net,
                    "bounces": bounces,
                    "landing": (float(t + cfg.dt), float(p_next[0]), float(p_next[1]), float(p_next[2])),
                }
            bounces.append(
                {
                    "t": float(t + cfg.dt),
                    "x": float(p_next[0]),
                    "y": float(p_next[1]),
                    "z": float(p_next[2]),
                    "own": own_side(float(p_next[0]), cfg.robot_side),
                }
            )
            v_next[2] = -cfg.bounce_cv * v_next[2]
            v_next[0] *= cfg.bounce_ch
            v_next[1] *= cfg.bounce_ch
            p_next[2] = cfg.table_z

        if p_next[0] <= cfg.robot_x:
            alpha = (p[0] - cfg.robot_x) / (p[0] - p_next[0]) if abs(p[0] - p_next[0]) > 1.0e-9 else 0.0
            p_hit = p + alpha * (p_next - p)
            v_hit = v + alpha * (v_next - v)
            return {
                "reason": "ok",
                "net": net,
                "bounces": bounces,
                "hit": (
                    float(p_hit[1]),
                    float(p_hit[2]),
                    float(v_hit[0]),
                    float(v_hit[1]),
                    float(v_hit[2]),
                    float(t + alpha * cfg.dt),
                ),
            }

        p = p_next
        v = v_next
        t += cfg.dt

    return {"reason": "timeout", "net": net, "bounces": bounces}


def is_valid_rollout(result: dict, args: argparse.Namespace) -> bool:
    if result["reason"] != "ok":
        return False
    bounces = result["bounces"]
    own_bounces = [b for b in bounces if b["own"]]
    opponent_bounces = [b for b in bounces if not b["own"]]
    if not (args.min_own_bounces <= len(own_bounces) <= args.max_own_bounces):
        return False
    if len(opponent_bounces) > args.max_opponent_bounces:
        return False

    hit_y, hit_z, _, _, _, hit_t = result["hit"]
    return (
        args.hit_y_range[0] <= hit_y <= args.hit_y_range[1]
        and args.hit_z_range[0] <= hit_z <= args.hit_z_range[1]
        and args.hit_time_range[0] <= hit_t <= args.hit_time_range[1]
    )


def summarize(label: str, arr: np.ndarray, cols: list[str]) -> None:
    if len(arr) == 0:
        print(f"[{label}] empty")
        return
    print(f"[{label}] n={len(arr)}")
    for idx, col in enumerate(cols):
        q = np.percentile(arr[:, idx], [5, 25, 50, 75, 95])
        print(
            f"  {col:>2s}: p5={q[0]:+.3f} p25={q[1]:+.3f} p50={q[2]:+.3f} "
            f"p75={q[3]:+.3f} p95={q[4]:+.3f}"
        )


def summarize_source_validity(source_states: np.ndarray, args: argparse.Namespace, cfg: RolloutCfg) -> None:
    valid = 0
    rejected = Counter()
    bounce_counts = Counter()
    for state in source_states:
        result = rollout(state, cfg)
        own_count = sum(1 for b in result.get("bounces", []) if b["own"])
        opp_count = sum(1 for b in result.get("bounces", []) if not b["own"])
        bounce_counts[(opp_count, own_count, result["reason"])] += 1
        if is_valid_rollout(result, args):
            valid += 1
        else:
            rejected[result["reason"]] += 1
    print(
        f"[source_valid] valid={valid}/{len(source_states)} rejected={dict(rejected)} "
        f"bounce_counts(opp,own,reason)={dict(bounce_counts)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", default=[DEFAULT_DATA_DIR])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--num-states", type=int, default=5000)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--birth-x", type=float, default=1.0)
    parser.add_argument("--z-offset", type=float, default=0.714)
    parser.add_argument("--velocity-source", choices=("fit", "file"), default="file")
    parser.add_argument("--vel-window-s", type=float, default=0.035)
    parser.add_argument("--source-hit-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-filter-robot-x", type=float, default=-1.47)
    parser.add_argument("--source-filter-table-z", type=float, default=0.76)
    parser.add_argument("--source-filter-y-range", type=float, nargs=2, default=(-0.10, 0.30))
    parser.add_argument("--source-filter-height-range", type=float, nargs=2, default=(0.0, 0.80))

    parser.add_argument("--sampler", choices=("empirical", "kde", "gaussian"), default="empirical")
    parser.add_argument("--jitter-frac", type=float, default=0.05)
    parser.add_argument("--kde-bw", type=float, default=None)
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(2.0, 98.0))

    parser.add_argument("--post-bounce-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--post-bounce-z-min", type=float, default=1.10)
    parser.add_argument("--post-bounce-vz-min", type=float, default=0.70)

    parser.add_argument("--robot-x", type=float, default=-1.47)
    parser.add_argument("--robot-side", type=int, default=-1)
    parser.add_argument("--drag-k", type=float, default=0.08)
    parser.add_argument("--bounce-ch", type=float, default=0.85)
    parser.add_argument("--bounce-cv", type=float, default=0.90)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--hit-y-range", type=float, nargs=2, default=(-0.25, 0.35))
    parser.add_argument("--hit-z-range", type=float, nargs=2, default=(0.90, 1.25))
    parser.add_argument("--hit-time-range", type=float, nargs=2, default=(0.62, 0.90))
    parser.add_argument("--min-own-bounces", type=int, default=1)
    parser.add_argument("--max-own-bounces", type=int, default=1)
    parser.add_argument("--max-opponent-bounces", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    source_states, source_files = extract_plane_states(args)
    summarize("source_state_at_birth_x", source_states, ["x", "y", "z", "vx", "vy", "vz"])

    sample_features, clip_lo, clip_hi = make_sampler(source_states, args, rng)
    cfg = RolloutCfg(
        robot_x=args.robot_x,
        robot_side=args.robot_side,
        drag_k=args.drag_k,
        bounce_ch=args.bounce_ch,
        bounce_cv=args.bounce_cv,
        dt=args.dt,
    )
    summarize_source_validity(source_states, args, cfg)

    kept = []
    hit_rows = []
    bounce_rows = []
    attempts = 0
    rejected = Counter()

    while len(kept) < args.num_states and attempts < args.max_attempts:
        n = min(args.batch_size, args.max_attempts - attempts)
        feats = sample_features(n)
        for feat in feats:
            attempts += 1
            state6 = np.concatenate([[args.birth_x], feat])
            result = rollout(state6, cfg)
            if is_valid_rollout(result, args):
                kept.append(state6)
                hit_rows.append(result["hit"])
                own = [b for b in result["bounces"] if b["own"]]
                b = own[0] if own else result["bounces"][0]
                bounce_rows.append([b["t"], b["x"], b["y"], b["z"]])
                if len(kept) >= args.num_states:
                    break
            else:
                rejected[result["reason"]] += 1
            if attempts >= args.max_attempts:
                break

    states = np.asarray(kept, dtype=np.float32)
    hit_arr = np.asarray(hit_rows, dtype=np.float32)
    bounce_arr = np.asarray(bounce_rows, dtype=np.float32)
    keep_rate = len(states) / max(attempts, 1)
    print(
        f"[rollout] attempts={attempts} kept={len(states)} keep_rate={keep_rate:.3f} "
        f"rejected={dict(rejected)}"
    )
    summarize("accepted_states", states, ["x", "y", "z", "vx", "vy", "vz"])
    summarize("accepted_hit_yzvxt", hit_arr[:, [0, 1, 2, 5]] if len(hit_arr) else hit_arr, ["y", "z", "vx", "t"])
    summarize("accepted_own_bounce", bounce_arr, ["t", "x", "y", "z"])

    if len(states) == 0:
        raise SystemExit("no accepted states; relax filters or inspect source data")

    metadata = {
        "args": vars(args),
        "rollout_cfg": asdict(cfg),
        "source_files": source_files,
        "rejected": dict(rejected),
        "keep_rate": keep_rate,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        states=states,
        ball_arrive_time_est=np.float32(np.median(hit_arr[:, 5])),
        source_states=source_states.astype(np.float32),
        accepted_hit=hit_arr,
        accepted_own_bounce=bounce_arr,
        clip_lo=clip_lo.astype(np.float32),
        clip_hi=clip_hi.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"[save] {args.out} states={states.shape} ball_arrive_time_est={np.median(hit_arr[:, 5]):.3f}s")


if __name__ == "__main__":
    main()
