"""Verify the two new reward terms FIRE with finite, in-range values on live env state.
Builds the env, steps with small random actions, calls the terms directly, reports stats."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="A1-TableTennis-SAC-Catch")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=160)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (  # noqa: E402
    OPP_TABLE_CENTER_X, RACKET_BODY_NAME, ROBOT_SIDE, TABLE_Z,
)
from unitree_rl_lab.tasks.table_tennis_sac.env_cfg import SAC_BALL_DRAG_K, SAC_BALL_LINEAR_DAMPING  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.mdp.rewards import (  # noqa: E402
    racket_ideal_normal_match, racket_predicted_landing,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=cfg); raw = env.unwrapped
env.reset()
common = dict(ball_name="ball", racket_body_name=RACKET_BODY_NAME, robot_side=ROBOT_SIDE,
             target_x=OPP_TABLE_CENTER_X, target_y=0.0, target_z=TABLE_Z,
             drag_k=SAC_BALL_DRAG_K, lin_damp=SAC_BALL_LINEAR_DAMPING, proximity_gate=0.45)
adim = int(env.action_space.shape[-1])
stats = {"normal": [], "landing": []}
fired = {"normal": 0, "landing": 0}; total = 0; nan = 0
for _ in range(args_cli.steps):
    act = (torch.rand(args_cli.num_envs, adim, device=raw.device) * 2 - 1) * 0.5
    env.step(act)
    nm = racket_ideal_normal_match(raw, angle_tol_deg=45.0, **common)
    ld = racket_predicted_landing(raw, sigma_x=0.35, sigma_y=0.3, **common)
    for tag, t in (("normal", nm), ("landing", ld)):
        nan += int(torch.isnan(t).sum() + torch.isinf(t).sum())
        pos = t[t > 0]
        fired[tag] += int((t > 0).sum())
        if pos.numel(): stats[tag].append(pos)
    total += args_cli.num_envs
print("\n==== NEW REWARD TERM VERIFICATION ====")
print(f"steps={args_cli.steps} envs={args_cli.num_envs} total_samples={total}  NaN/Inf={nan}")
for tag in ("normal", "landing"):
    if stats[tag]:
        v = torch.cat(stats[tag])
        print(f"{tag:8s} fired {fired[tag]:6d}/{total} ({100*fired[tag]/total:.1f}%)  "
              f"when>0: min={v.min():.3f} mean={v.mean():.3f} max={v.max():.3f}")
    else:
        print(f"{tag:8s} fired 0/{total}  (NEVER FIRED -- check gating!)")
env.close()
