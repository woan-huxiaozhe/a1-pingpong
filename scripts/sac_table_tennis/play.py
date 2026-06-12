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


def _termination_reasons(env, done: torch.Tensor) -> dict[int, list[str]]:
    raw_env = env.unwrapped
    manager = getattr(raw_env, "termination_manager", None)
    if manager is None:
        return {}

    done_cpu = done.detach().cpu().bool().reshape(-1)
    reasons: dict[int, list[str]] = {}
    for env_id in done_cpu.nonzero(as_tuple=True)[0].tolist():
        active: list[str] = []
        for term_name in manager.active_terms:
            term_done = manager.get_term(term_name).detach().cpu().bool().reshape(-1)
            if bool(term_done[env_id]):
                active.append(term_name)
        reasons[env_id] = active if active else ["unknown"]
    return reasons


def _joint_limit_details(env, env_id: int) -> list[str]:
    raw_env = env.unwrapped
    mask = getattr(raw_env, "_sac_final_joint_limit_mask", None)
    if mask is None or env_id >= mask.shape[0]:
        return []

    mask_env = mask[env_id].detach().cpu().bool()
    if not torch.any(mask_env):
        return []

    q = raw_env._sac_final_joint_limit_q[env_id].detach().cpu()
    low = raw_env._sac_final_joint_limit_low[env_id].detach().cpu()
    high = raw_env._sac_final_joint_limit_high[env_id].detach().cpu()
    names = getattr(raw_env, "_sac_joint_limit_joint_names", None)
    if names is None:
        names = [f"joint_{idx}" for idx in range(mask_env.numel())]

    details: list[str] = []
    for joint_idx in mask_env.nonzero(as_tuple=True)[0].tolist():
        q_value = float(q[joint_idx])
        low_value = float(low[joint_idx])
        high_value = float(high[joint_idx])
        lower_clearance = q_value - low_value
        upper_clearance = high_value - q_value
        side = "low" if lower_clearance < upper_clearance else "high"
        clearance = min(lower_clearance, upper_clearance)
        details.append(
            f"{names[joint_idx]}:{side} q={q_value:.4f} "
            f"limit=[{low_value:.4f},{high_value:.4f}] clearance={clearance:.4f}"
        )
    return details


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
        termination_reasons = _termination_reasons(env, done)
        for env_id, info in final_infos.items():
            events = decode_events(info.event_mask)
            for event in events:
                counts[event] = counts.get(event, 0) + 1
            completed += 1
            reasons = termination_reasons.get(env_id, ["unknown"])
            joint_limits = _joint_limit_details(env, env_id) if "joint_limit" in reasons else []
            joint_limit_text = f" joint_limits={joint_limits}" if joint_limits else ""
            print(
                f"[PLAY] episode={completed} env={env_id} terminations={reasons} "
                f"events={events} landing_y={info.landing_y:.3f}{joint_limit_text}"
            )
            if completed >= args_cli.episodes:
                break

    print(f"[PLAY] completed={completed} counts={counts}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
