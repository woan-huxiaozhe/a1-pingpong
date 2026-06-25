"""Validate real serve states against IsaacSim ball/table physics.

For each mocap trajectory:

1. extract the real ball state at ``x=+1.0``;
2. optionally filter by the real state at ``robot_x=-1.47``;
3. reset an IsaacSim ball to the ``x=+1.0`` state;
4. step physics until the simulated ball crosses ``robot_x``;
5. compare simulated vs real ``y/z/v`` at the crossing plane.

This isolates the ball-table-net physics from robot/paddle contact.  It is meant
to answer whether the sampled initial state distribution is dynamically
consistent with the recorded trajectories before regenerating ``serve_states``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_DIR = os.path.join(REPO_ROOT, "source", "unitree_rl_lab")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=[
            "/home/woan/kalman_filter_pingpong/data/0611_data_vel",
            "/home/woan/kalman_filter_pingpong/data/0617_traj_data",
        ],
    )
    parser.add_argument("--birth-x", type=float, default=1.0)
    parser.add_argument("--robot-x", type=float, default=-1.47)
    parser.add_argument("--z-offset", type=float, default=0.714)
    parser.add_argument("--table-z", type=float, default=0.76)
    parser.add_argument("--velocity-source", choices=("file", "fit"), default="file")
    parser.add_argument("--vel-window-s", type=float, default=0.035)
    parser.add_argument("--filter-y-range", type=float, nargs=2, default=(-0.10, 0.30))
    parser.add_argument("--filter-height-range", type=float, nargs=2, default=(0.0, 0.80))
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 means all filtered trajectories")
    parser.add_argument("--num-envs", type=int, default=0, help="0 means one env per trajectory")
    parser.add_argument("--env-spacing", type=float, default=4.0)
    parser.add_argument("--sim-dt", type=float, default=0.0025)
    parser.add_argument("--sim-time", type=float, default=2.0)
    parser.add_argument("--ball-mass", type=float, default=0.0027)
    parser.add_argument("--linear-damping", type=float, default=0.0)
    parser.add_argument(
        "--drag-k",
        type=float,
        default=0.0,
        help="Quadratic drag coefficient in a_drag = -drag_k * |v| * v.",
    )
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument(
        "--skip-app-close",
        action="store_true",
        help="Print results and let the process exit without simulation_app.close(); useful when Kit shutdown hangs.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from unitree_rl_lab.tasks.table_tennis_sac.create_serve_states import load_vrpn, state_at_x  # noqa: E402
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (  # noqa: E402
    BALL_USD_PATH,
    TABLE_USD_PATH,
)


@configclass
class BallTableSceneCfg(InteractiveSceneCfg):
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BALL_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                linear_damping=args_cli.linear_damping,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.2)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def collect_records(args: argparse.Namespace) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    skipped: Counter = Counter()
    files: list[str] = []
    for data_dir in args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(data_dir, "*.txt"))))

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
        hit = state_at_x(t, pos, vel_file, args.robot_x, args.velocity_source, args.vel_window_s)
        if hit is None:
            skipped["no_robot_x"] += 1
            continue

        birth_state, birth_t = birth
        hit_state, hit_t = hit
        birth_state = birth_state.copy()
        hit_state = hit_state.copy()
        birth_state[2] += args.z_offset
        hit_state[2] += args.z_offset
        hit_height = hit_state[2] - args.table_z

        if not (args.filter_y_range[0] < hit_state[1] < args.filter_y_range[1]):
            skipped["hit_y_filter"] += 1
            continue
        if not (args.filter_height_range[0] < hit_height < args.filter_height_range[1]):
            skipped["hit_z_filter"] += 1
            continue

        records.append(
            {
                "file": os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)),
                "birth": birth_state.astype(np.float64),
                "hit": hit_state.astype(np.float64),
                "real_tau": float(hit_t - birth_t),
            }
        )

    if args.max_trajectories > 0:
        records = records[: args.max_trajectories]
    return records, skipped


def simulate_batch(args: argparse.Namespace, records: list[dict]) -> list[dict]:
    num_envs = args.num_envs if args.num_envs > 0 else len(records)
    if num_envs < len(records):
        raise ValueError("--num-envs must be 0 or at least the number of selected records")

    print(
        f"[sim] create context device={args.device} dt={args.sim_dt} num_envs={num_envs} "
        f"linear_damping={args.linear_damping} drag_k={args.drag_k}",
        flush=True,
    )
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args.sim_dt, device=args.device))
    print("[sim] create scene", flush=True)
    scene = InteractiveScene(BallTableSceneCfg(num_envs=num_envs, env_spacing=args.env_spacing))
    print("[sim] reset", flush=True)
    sim.reset()
    scene.reset()

    device = scene.device
    ball = scene["ball"]
    env_ids = torch.arange(len(records), device=device)

    root_state = ball.data.default_root_state[: len(records)].clone()
    root_state[:, 0:3] = torch.tensor([r["birth"][0:3] for r in records], device=device, dtype=root_state.dtype)
    root_state[:, 0:3] += scene.env_origins[: len(records)]
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=root_state.dtype).reshape(1, 4)
    root_state[:, 7:10] = torch.tensor([r["birth"][3:6] for r in records], device=device, dtype=root_state.dtype)
    root_state[:, 10:13] = 0.0
    ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    scene.write_data_to_sim()
    scene.update(args.sim_dt)
    print("[sim] start stepping", flush=True)

    done = torch.zeros(len(records), dtype=torch.bool, device=device)
    sim_cross = torch.full((len(records), 7), float("nan"), device=device)
    prev_pos = ball.data.root_pos_w[: len(records), :3].clone() - scene.env_origins[: len(records)]
    prev_vel = ball.data.root_lin_vel_w[: len(records), :3].clone()
    max_steps = int(args.sim_time / args.sim_dt)
    zero_torque = torch.zeros((len(records), 1, 3), device=device)

    for step in range(max_steps):
        if args.drag_k > 0.0:
            vel_for_drag = ball.data.root_lin_vel_w[: len(records), :3]
            speed = torch.linalg.norm(vel_for_drag, dim=-1, keepdim=True)
            force = -args.ball_mass * args.drag_k * speed * vel_for_drag
            ball.permanent_wrench_composer.set_forces_and_torques(
                forces=force.reshape(len(records), 1, 3),
                torques=zero_torque,
                env_ids=env_ids,
                is_global=True,
            )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(args.sim_dt)
        if not args.quiet_progress and step > 0 and step % 200 == 0:
            print(f"[sim] step={step}/{max_steps} crossed={int(done.sum().item())}/{len(records)}", flush=True)

        pos = ball.data.root_pos_w[: len(records), :3].clone() - scene.env_origins[: len(records)]
        vel = ball.data.root_lin_vel_w[: len(records), :3].clone()
        crossing = (~done) & (prev_pos[:, 0] > args.robot_x) & (pos[:, 0] <= args.robot_x)
        if torch.any(crossing):
            ids = torch.nonzero(crossing, as_tuple=False).squeeze(-1)
            denom = (prev_pos[ids, 0] - pos[ids, 0]).clamp(min=1.0e-9)
            alpha = ((prev_pos[ids, 0] - args.robot_x) / denom).reshape(-1, 1)
            cross_pos = prev_pos[ids] + alpha * (pos[ids] - prev_pos[ids])
            cross_vel = prev_vel[ids] + alpha * (vel[ids] - prev_vel[ids])
            sim_cross[ids, 0:3] = cross_pos
            sim_cross[ids, 3:6] = cross_vel
            sim_cross[ids, 6] = (step + alpha.squeeze(-1)) * args.sim_dt
            done[ids] = True
            if bool(torch.all(done)):
                break
        prev_pos = pos
        prev_vel = vel
        if args.drag_k > 0.0:
            ball.permanent_wrench_composer.reset(env_ids)

    out: list[dict] = []
    sim_np = sim_cross.cpu().numpy()
    for i, rec in enumerate(records):
        real = rec["hit"]
        if np.isfinite(sim_np[i]).all():
            sim_state = sim_np[i]
            err = {
                "dy": float(sim_state[1] - real[1]),
                "dz": float(sim_state[2] - real[2]),
                "dvx": float(sim_state[3] - real[3]),
                "dvy": float(sim_state[4] - real[4]),
                "dvz": float(sim_state[5] - real[5]),
                "dtau": float(sim_state[6] - rec["real_tau"]),
            }
        else:
            sim_state = None
            err = None
        out.append({"file": rec["file"], "real": real, "real_tau": rec["real_tau"], "sim": sim_state, "err": err})

    return out


def summarize_results(results: list[dict], args: argparse.Namespace) -> dict:
    valid = [r for r in results if r["err"] is not None]
    missed = [r for r in results if r["err"] is None]
    summary: dict = {
        "linear_damping": args.linear_damping,
        "drag_k": args.drag_k,
        "velocity_source": args.velocity_source,
        "total": len(results),
        "crossed": len(valid),
        "missed": len(missed),
    }
    keys = ["dy", "dz", "dvx", "dvy", "dvz", "dtau"]
    if valid:
        arr = np.array([[r["err"][k] for k in keys] for r in valid], dtype=np.float64)
        for i, key in enumerate(keys):
            vals = arr[:, i]
            q = np.percentile(vals, [5, 50, 95])
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_mae"] = float(np.mean(np.abs(vals)))
            summary[f"{key}_rmse"] = float(np.sqrt(np.mean(vals * vals)))
            summary[f"{key}_p5"] = float(q[0])
            summary[f"{key}_p50"] = float(q[1])
            summary[f"{key}_p95"] = float(q[2])
        yz = arr[:, 0:2]
        yz_norm = np.linalg.norm(yz, axis=1)
        summary["yz_norm_mean"] = float(yz_norm.mean())
        summary["yz_norm_p50"] = float(np.percentile(yz_norm, 50))
        summary["yz_norm_p95"] = float(np.percentile(yz_norm, 95))
    return summary


def print_stats(results: list[dict], args: argparse.Namespace) -> None:
    valid = [r for r in results if r["err"] is not None]
    missed = [r for r in results if r["err"] is None]
    print(f"[sim] crossed={len(valid)}/{len(results)} missed={len(missed)}")
    if missed:
        print("[sim] missed files:", [r["file"] for r in missed[:20]])
    if not valid:
        print(f"[summary] {json.dumps(summarize_results(results, args), sort_keys=True)}")
        return

    keys = ["dy", "dz", "dvx", "dvy", "dvz", "dtau"]
    arr = np.array([[r["err"][k] for k in keys] for r in valid], dtype=np.float64)
    print("[error] sim - real at x=-1.47")
    for i, key in enumerate(keys):
        vals = arr[:, i]
        q = np.percentile(vals, [5, 25, 50, 75, 95])
        rmse = np.sqrt(np.mean(vals * vals))
        mae = np.mean(np.abs(vals))
        print(
            f"  {key:>4s}: mean={vals.mean():+.4f} mae={mae:.4f} rmse={rmse:.4f} "
            f"p5={q[0]:+.4f} p50={q[2]:+.4f} p95={q[4]:+.4f}"
        )

    yz = arr[:, 0:2]
    yz_norm = np.linalg.norm(yz, axis=1)
    print(
        f"[error] yz_norm: mean={yz_norm.mean():.4f} m "
        f"p50={np.percentile(yz_norm, 50):.4f} m p95={np.percentile(yz_norm, 95):.4f} m"
    )

    worst = sorted(valid, key=lambda r: (r["err"]["dy"] ** 2 + r["err"]["dz"] ** 2), reverse=True)[:8]
    print("[worst_yz]")
    for r in worst:
        e = r["err"]
        print(
            f"  {r['file']}: dy={e['dy']:+.4f} dz={e['dz']:+.4f} "
            f"dv=({e['dvx']:+.3f},{e['dvy']:+.3f},{e['dvz']:+.3f}) dtau={e['dtau']:+.4f}"
        )
    print(f"[summary] {json.dumps(summarize_results(results, args), sort_keys=True)}")


def main() -> None:
    records, skipped = collect_records(args_cli)
    print(f"[data] filtered_records={len(records)} skipped={dict(skipped)}")
    if not records:
        raise SystemExit("no records after filtering")
    results = simulate_batch(args_cli, records)
    print_stats(results, args_cli)
    if not args_cli.skip_app_close:
        simulation_app.close()


if __name__ == "__main__":
    main()
