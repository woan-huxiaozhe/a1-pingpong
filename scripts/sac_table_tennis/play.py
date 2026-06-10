"""Play/evaluate an A1 table-tennis SAC checkpoint."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Run a trained A1-TableTennis-SAC-Catch policy.")
parser.add_argument("--task", type=str, default="A1-TableTennis-SAC-Catch")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--zero_action", action="store_true", help="Run a zero-action agent for visual inspection.")
parser.add_argument("--real_time", action="store_true", help="Sleep after each step to approximate env step time.")
parser.add_argument("--sleep_per_step", type=float, default=0.0, help="Seconds to sleep after each env step.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.table_tennis_sac.event_tags import decode_events  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.runtime import (  # noqa: E402
    extract_final_episode_infos,
    split_actor_critic_obs,
)
from unitree_rl_lab.tasks.table_tennis_sac.sac import SACAgent  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    obs_actor, _ = split_actor_critic_obs(obs)
    agent = None
    if args_cli.zero_action:
        print("[PLAY] Running zero-action agent.")
    else:
        if args_cli.checkpoint is None:
            raise ValueError("--checkpoint is required unless --zero_action is set.")
        agent = SACAgent.load(args_cli.checkpoint, device=obs_actor.device)
    sleep_per_step = float(args_cli.sleep_per_step)
    if args_cli.real_time:
        sleep_per_step = max(sleep_per_step, float(getattr(env.unwrapped, "step_dt", 0.02)))

    completed = 0
    steps = 0
    counts: dict[str, int] = {}
    while completed < args_cli.episodes and steps < args_cli.max_steps * max(1, args_cli.episodes):
        with torch.no_grad():
            if args_cli.zero_action:
                action = torch.zeros(obs_actor.shape[0], int(env.action_space.shape[-1]), device=obs_actor.device)
            else:
                action = agent.act(obs_actor, deterministic=not args_cli.stochastic)
        obs, _, terminated, truncated, _ = env.step(action.to(obs_actor.device))
        done = terminated | truncated
        obs_actor, _ = split_actor_critic_obs(obs)
        steps += 1
        if sleep_per_step > 0.0:
            time.sleep(sleep_per_step)

        final_infos = extract_final_episode_infos(env, done)
        for info in final_infos.values():
            events = decode_events(info.event_mask)
            for event in events:
                counts[event] = counts.get(event, 0) + 1
            completed += 1
            print(f"[PLAY] episode={completed} events={events} landing_y={info.landing_y:.3f}")
            if completed >= args_cli.episodes:
                break

    print(f"[PLAY] completed={completed} counts={counts}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
