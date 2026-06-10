# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe the TRAINED policy's contact geometry (per-serve min_gap + offset).

Marries play.py's checkpoint loader with play_pure_ref.py's per-trial geometry
tracker: runs the trained policy (residual + phase_speed ACTIVE, multi-ref
bucket selection live) over N serves and prints the same median min_gap +
offset(ball-racket) dx/dy/dz diagnosis. Use this to disambiguate, after a
multi-ref run plateaus at zero returns, whether the residual contact gap is:
  - height-dominant (|dz| big, min_gap ~ pure-ref level) -> needs height-matched refs
  - residual-wander (min_gap WORSE than pure-ref) -> tighten residual instead.

Usage (num_envs forced to 1 by the play env_cfg; no video, fast):
  python scripts/rsl_rl/probe_policy_geom.py --task A1-TableTennis-Backhand \
      --checkpoint logs/rsl_rl/a1_tabletennis_backhand/<run>/model_4500.pt \
      --max_trials 100
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Probe trained-policy contact geometry.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--max_trials", type=int, default=100, help="Stop after this many serves.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from importlib.metadata import version

import gymnasium as gym
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # rsl-rl 5.x schema: infer actor/critic from legacy `policy`, else OnPolicyRunner KeyErrors
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    racket_body_idx = robot.find_bodies("Link_yb_paddle")[0][0]

    # robot side (own-table / direction sign), mirrors play_pure_ref
    robot_side = getattr(env_cfg, "_robot_side", None)
    if robot_side is None:
        robot_side = 1 if env_cfg.scene.robot.init_state.pos[0] > 0 else -1

    TELEPORT_THRESH = 0.30  # ball jump > 0.3m => relaunched (new trial)

    # reset + initial obs (rsl-rl 2.3 returns a tuple)
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    num_trials = 0
    min_gap = 999.0
    min_gap_phase = 0.0
    min_gap_racket = None
    min_gap_ball = None
    min_gap_mid = -1
    prev_pos = None
    trial_stats = []  # (gap, phase, bx,by,bz, rx,ry,rz, motion_id)

    motion_term = env.unwrapped.command_manager.get_term("motion")

    print(f"[INFO] Probing trained policy over {args_cli.max_trials} serves (residual+phase_speed ACTIVE)...")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        try:
            phase_val = float(motion_term.phase[0])
            mid_val = int(motion_term.motion_ids[0])
        except Exception:
            phase_val = -1.0
            mid_val = -1

        ball_pos_local = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
        racket_pos_local = (robot.data.body_pos_w[0, racket_body_idx] - scene.env_origins[0]).cpu().numpy()
        gap = float(((ball_pos_local - racket_pos_local) ** 2).sum() ** 0.5)

        if gap < min_gap:
            min_gap = gap
            min_gap_phase = phase_val
            min_gap_racket = racket_pos_local.copy()
            min_gap_ball = ball_pos_local.copy()
            min_gap_mid = mid_val

        # trial boundary: ball teleported (relaunch)
        if prev_pos is not None and float(((ball_pos_local - prev_pos) ** 2).sum() ** 0.5) > TELEPORT_THRESH:
            num_trials += 1
            if min_gap_racket is not None:
                _off = min_gap_ball - min_gap_racket
                print(f"[trial {num_trials:4d}] mid={min_gap_mid} min_gap={min_gap:.3f}m  phase@min_gap={min_gap_phase:.3f}  "
                      f"offset(ball-racket) dx={_off[0]:+.3f} dy={_off[1]:+.3f} dz={_off[2]:+.3f}")
                trial_stats.append((float(min_gap), float(min_gap_phase),
                                    float(min_gap_ball[0]), float(min_gap_ball[1]), float(min_gap_ball[2]),
                                    float(min_gap_racket[0]), float(min_gap_racket[1]), float(min_gap_racket[2]),
                                    float(min_gap_mid)))
            min_gap = 999.0
            min_gap_racket = None
            if args_cli.max_trials and num_trials >= args_cli.max_trials:
                break

        prev_pos = ball_pos_local.copy()

    if trial_stats:
        arr = np.array(trial_stats)  # gap,phase,bx,by,bz,rx,ry,rz,mid
        med = np.median(arr, axis=0)
        dabs = np.median(np.abs(arr[:, 2:5] - arr[:, 5:8]), axis=0)
        gaps = arr[:, 0]
        mids = arr[:, 8].astype(int)
        print("\n========== TRAINED-POLICY CONTACT GEOMETRY (model under residual+phase_speed) ==========")
        print(f"trials = {len(trial_stats)}")
        print(f"median min_gap        = {med[0]:.3f} m   (pure-ref baseline 0.109; contact needs <~0.06)")
        print(f"min_gap pctiles       = p10 {np.percentile(gaps,10):.3f} | p25 {np.percentile(gaps,25):.3f} | "
              f"p50 {np.percentile(gaps,50):.3f} | p75 {np.percentile(gaps,75):.3f}")
        print(f"frac min_gap<0.08     = {(gaps<0.08).mean()*100:.1f}%   frac<0.10 = {(gaps<0.10).mean()*100:.1f}%")
        print(f"median phase@min_gap  = {med[1]:.3f}      (hit_phase target = 0.4643)")
        print(f"median ball  @min_gap = x={med[2]:+.3f} y={med[3]:+.3f} z={med[4]:+.3f}")
        print(f"median racket@min_gap = x={med[5]:+.3f} y={med[6]:+.3f} z={med[7]:+.3f}")
        print(f"median SIGNED offset(ball-racket) dx={med[2]-med[5]:+.3f} dy={med[3]-med[6]:+.3f} dz={med[4]-med[7]:+.3f}")
        print(f"median |offset| per axis           |dx|={dabs[0]:.3f} |dy|={dabs[1]:.3f} |dz|={dabs[2]:.3f}")
        print("\n----- per-bucket breakdown (motion_id 0=middle 1=left 2=right) -----")
        for b, name in [(0, "middle"), (1, "left  "), (2, "right ")]:
            m = mids == b
            if not m.any():
                print(f"  id{b} {name}: 0 trials  (NEVER SELECTED)")
                continue
            sub = arr[m]
            sdy = np.median(sub[:, 3] - sub[:, 6])
            sdz = np.median(sub[:, 4] - sub[:, 7])
            sby = np.median(sub[:, 3])  # ball y this bucket actually got
            sgap = np.median(sub[:, 0])
            print(f"  id{b} {name}: {m.sum():3d} trials ({m.mean()*100:4.0f}%)  "
                  f"median ball_y={sby:+.3f}  signed dy={sdy:+.3f}  dz={sdz:+.3f}  min_gap={sgap:.3f}")
        print("\nread: selection collapsed to 1 bucket -> SELECTION bug; a bucket's signed dy biased -> that")
        print("      offset miscalibrated; all buckets ~+0.11 dy & min_gap>pure-ref -> POLICY wanders off-ref")
        print("      (residual/phase_speed) -> tighten residual / penalize phase_speed; |dz| was the gate's")
        print("      worry but here |dy| dominates -> height-matched refs would fix the WRONG axis.")
        print("========================================================================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
