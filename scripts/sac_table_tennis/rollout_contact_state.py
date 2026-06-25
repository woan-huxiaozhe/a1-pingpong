"""Capture the SAC policy's paddle state AT CONTACT and compare to the analytic ideal.

This is the confirmatory experiment for the "why RL lobs" analysis (H1: the blade
*normal* is the unrewarded degree of freedom that sets the rebound direction).

At every first-contact step it records, from the PRE-contact step state (one control
step before the collision rebounds the ball):

  - actual paddle face normal ``n_actual`` (link-local +Y, world frame)
  - actual paddle velocity: body-origin translational ``v_paddle_body`` (what the
    reward matches) and blade-center ``v_paddle_center`` (adds omega x r)
  - incoming ball position/velocity ``v_in``

and computes ``hitting.ideal_racket_velocity`` -> ``(v_paddle_ideal, v_out_ideal,
n_ideal)`` for the SAME contact (target = opponent-table center). The actual ball
outgoing (vx, vz) cached by the episode tracker is recorded too.

Everything is dumped raw to JSON; analysis (angles, gaps) is a separate offline pass
so it needs no Isaac. Runs headless, many envs, deterministic policy by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Capture SAC contact-state vs analytic ideal.")
parser.add_argument("--task", type=str, default="A1-TableTennis-SAC-Catch")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--out", type=str, default="logs/sac_table_tennis/contact_state.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (  # noqa: E402
    OPP_TABLE_CENTER_X,
    RACKET_BODY_NAME,
    ROBOT_SIDE,
    TABLE_Z,
)
from unitree_rl_lab.tasks.table_tennis_sac.env_cfg import SAC_BALL_DRAG_K, SAC_BALL_LINEAR_DAMPING  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT, decode_events  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.mdp.hitting import NEUTRAL_THETA, ideal_racket_velocity  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import (  # noqa: E402
    _racket_body_lin_vel,
    _racket_body_state,
    racket_normal,
)
from unitree_rl_lab.tasks.table_tennis_sac.runtime import extract_final_episode_infos, split_actor_critic_obs  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.sac import SACAgent  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

HIT_BIT = EVENT_TO_BIT["hit"]


def read_state(raw):
    ball = raw.scene["ball"]
    ball_pos = ball.data.root_pos_w[:, :3].clone()
    ball_vel = ball.data.root_lin_vel_w[:, :3].clone()
    n_actual = racket_normal(raw, RACKET_BODY_NAME).clone()
    v_body = _racket_body_lin_vel(raw, RACKET_BODY_NAME).clone()
    center, center_vel, _ = _racket_body_state(raw, RACKET_BODY_NAME)
    return ball_pos, ball_vel, n_actual, v_body, center_vel.clone(), center.clone()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    obs_actor, _ = split_actor_critic_obs(obs)
    agent = SACAgent.load(args_cli.checkpoint, device=obs_actor.device)

    num_envs = obs_actor.shape[0]
    device = obs_actor.device
    origins = raw.scene.env_origins  # [N,3]
    target = torch.zeros(num_envs, 3, device=device)
    target[:, 0] = origins[:, 0] + OPP_TABLE_CENTER_X
    target[:, 1] = origins[:, 1] + 0.0
    target[:, 2] = origins[:, 2] + TABLE_Z

    pending: dict[int, dict] = {}
    records: list[dict] = []
    counts: dict[str, int] = {}

    completed = 0
    steps = 0
    episode_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_total = args_cli.max_steps * max(1, args_cli.episodes)

    while completed < args_cli.episodes and steps < max_total:
        pre = read_state(raw)  # pre-step == pre-contact for a hit detected this step
        with torch.no_grad():
            action = agent.act(obs_actor, deterministic=not args_cli.stochastic)
        obs, _, terminated, truncated, _ = env.step(action.to(device))
        done = terminated | truncated
        obs_actor, _ = split_actor_critic_obs(obs)
        steps += 1

        mask = raw._sac_step_event_mask
        hit_now = ((mask & HIT_BIT) != 0).nonzero(as_tuple=True)[0]
        if hit_now.numel() > 0:
            idx = hit_now
            ball_pos, ball_vel, n_actual, v_body, v_center, _ = (t[idx] for t in pre)
            v_p_ideal, v_out_ideal, n_ideal = ideal_racket_velocity(
                ball_pos, ball_vel, target[idx],
                theta=NEUTRAL_THETA, drag_k=SAC_BALL_DRAG_K,
                lin_damp=SAC_BALL_LINEAR_DAMPING, control_dt=float(raw.step_dt),
            )
            out_vx = raw._sac_hit_outgoing_speed[idx]   # forward ball speed at contact (=-side*vx)
            out_vz = raw._sac_hit_up_speed[idx]
            offset = raw._sac_hit_center_offset[idx]
            ball_local = ball_pos - origins[idx]
            for k, e in enumerate(idx.tolist()):
                pending[e] = {
                    "env": e,
                    "ball_pos_local": ball_local[k].cpu().tolist(),
                    "v_in": ball_vel[k].cpu().tolist(),
                    "n_actual": n_actual[k].cpu().tolist(),
                    "v_paddle_body": v_body[k].cpu().tolist(),
                    "v_paddle_center": v_center[k].cpu().tolist(),
                    "v_paddle_ideal": v_p_ideal[k].cpu().tolist(),
                    "v_out_ideal": v_out_ideal[k].cpu().tolist(),
                    "n_ideal": n_ideal[k].cpu().tolist(),
                    "actual_out_vx_fwd": float(out_vx[k].cpu()),
                    "actual_out_vz_up": float(out_vz[k].cpu()),
                    "hit_center_offset": float(offset[k].cpu()),
                }

        final_infos = extract_final_episode_infos(env, done)
        for env_id, info in final_infos.items():
            events = decode_events(info.event_mask)
            for ev in events:
                counts[ev] = counts.get(ev, 0) + 1
            rec = pending.pop(env_id, None)
            if rec is not None:
                rec["outcome"] = events
                rec["landing_x"] = float(info.landing_x)
                rec["landing_y"] = float(info.landing_y)
                records.append(rec)
            completed += 1
            if completed >= args_cli.episodes:
                break

        done_ids = done.reshape(-1).nonzero(as_tuple=True)[0]
        if done_ids.numel() > 0:
            episode_ids[done_ids] += 1

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": args_cli.checkpoint,
        "robot_side": ROBOT_SIDE,
        "opp_center_x": OPP_TABLE_CENTER_X,
        "table_z": TABLE_Z,
        "neutral_theta_deg": __import__("math").degrees(NEUTRAL_THETA),
        "completed": completed,
        "counts": counts,
        "n_contacts": len(records),
        "records": records,
    }, indent=2))
    print(f"[ROLLOUT] completed={completed} counts={counts} contacts={len(records)}")
    print(f"[ROLLOUT] wrote {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
