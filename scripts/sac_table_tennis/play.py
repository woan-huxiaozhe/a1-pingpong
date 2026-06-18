"""Play/evaluate an A1 table-tennis SAC checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
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
parser.add_argument(
    "--sim_log_dir",
    type=str,
    default="logs/sac_table_tennis/sim_logs",
    help="Directory for per-step simulated joint logs.",
)
parser.add_argument("--no_sim_log", action="store_true", help="Disable per-step simulated joint logging.")
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


def _safe_log_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("._") or "unnamed"


def _policy_log_stem(task: str, checkpoint: str | None, zero_action: bool) -> str:
    task_name = _safe_log_name(task)
    if zero_action:
        return f"{task_name}__zero_action"

    checkpoint_path = Path(checkpoint or "policy")
    checkpoint_name = _safe_log_name(checkpoint_path.stem)
    run_name = ""
    if checkpoint_path.parent.name == "checkpoints":
        run_name = checkpoint_path.parent.parent.name
    elif checkpoint_path.parent.name:
        run_name = checkpoint_path.parent.name
    run_name = _safe_log_name(run_name) if run_name else ""

    parts = [task_name]
    if run_name:
        parts.append(run_name)
    parts.append(checkpoint_name)
    return "__".join(parts)


def _unique_csv_path(log_dir: Path, stem: str) -> Path:
    path = log_dir / f"{stem}.csv"
    suffix = 2
    while path.exists():
        path = log_dir / f"{stem}_{suffix:02d}.csv"
        suffix += 1
    return path


class SimJointLogger:
    def __init__(self, env, log_dir: str, task: str, checkpoint: str | None, zero_action: bool):
        raw_env = env.unwrapped
        self._robot = raw_env.scene["robot"]
        self._action_term = raw_env.action_manager.get_term("right_arm")
        self._joint_ids = list(self._action_term._joint_ids)
        self._joint_names = list(self._action_term._joint_names)
        self._step_dt = float(getattr(raw_env, "step_dt", 0.0))

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.path = _unique_csv_path(log_path, _policy_log_stem(task, checkpoint, zero_action))
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)

        header = ["global_step", "time_s", "env_id", "episode", "episode_step", "done", "terminated", "truncated"]
        for prefix in ("action", "q_target", "q_sim", "qd_sim", "tau_sim"):
            header.extend(f"{prefix}_{name}" for name in self._joint_names)
        self._writer.writerow(header)

    def write_step(
        self,
        *,
        global_step: int,
        action: torch.Tensor,
        episode_ids: torch.Tensor,
        episode_steps: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        action_cpu = action.detach().cpu()
        target_cpu = self._action_term.processed_actions.detach().cpu()
        pos_cpu = self._robot.data.joint_pos[:, self._joint_ids].detach().cpu()
        vel_cpu = self._robot.data.joint_vel[:, self._joint_ids].detach().cpu()
        torque_cpu = self._robot.data.applied_torque[:, self._joint_ids].detach().cpu()
        episode_ids_cpu = episode_ids.detach().cpu()
        episode_steps_cpu = episode_steps.detach().cpu()
        terminated_cpu = terminated.detach().cpu().bool().reshape(-1)
        truncated_cpu = truncated.detach().cpu().bool().reshape(-1)
        done_cpu = terminated_cpu | truncated_cpu

        time_s = global_step * self._step_dt if self._step_dt > 0.0 else global_step
        for env_id in range(action_cpu.shape[0]):
            row = [
                global_step,
                f"{time_s:.6f}",
                env_id,
                int(episode_ids_cpu[env_id].item()) + 1,
                int(episode_steps_cpu[env_id].item()),
                int(done_cpu[env_id].item()),
                int(terminated_cpu[env_id].item()),
                int(truncated_cpu[env_id].item()),
            ]
            for values in (action_cpu[env_id], target_cpu[env_id], pos_cpu[env_id], vel_cpu[env_id], torque_cpu[env_id]):
                row.extend(f"{float(value):.9g}" for value in values)
            self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


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

    sim_logger = None
    if not args_cli.no_sim_log:
        sim_logger = SimJointLogger(env, args_cli.sim_log_dir, args_cli.task, args_cli.checkpoint, args_cli.zero_action)
        print(f"[PLAY] sim_joint_log={sim_logger.path}")

    completed = 0
    steps = 0
    num_envs = obs_actor.shape[0]
    episode_ids = torch.zeros(num_envs, dtype=torch.long, device=obs_actor.device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=obs_actor.device)
    counts: dict[str, int] = {}
    metric_names = (
        "min_dist",
        "hit_center_offset",
        "landing_x",
        "landing_y",
        "hit_outgoing_speed",
        "hit_up_speed",
        "post_hit_max_height",
    )
    metric_sums = {name: 0.0 for name in metric_names}
    metric_counts = {name: 0 for name in metric_names}
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
        episode_steps += 1
        if sim_logger is not None:
            sim_logger.write_step(
                global_step=steps,
                action=action,
                episode_ids=episode_ids,
                episode_steps=episode_steps,
                terminated=terminated,
                truncated=truncated,
            )
        if sleep_per_step > 0.0:
            time.sleep(sleep_per_step)

        final_infos = extract_final_episode_infos(env, done)
        termination_reasons = _termination_reasons(env, done)
        for env_id, info in final_infos.items():
            events = decode_events(info.event_mask)
            for event in events:
                counts[event] = counts.get(event, 0) + 1
            for metric in metric_names:
                value = float(getattr(info, metric))
                if math.isfinite(value):
                    metric_sums[metric] += value
                    metric_counts[metric] += 1
            completed += 1
            reasons = termination_reasons.get(env_id, ["unknown"])
            joint_limits = _joint_limit_details(env, env_id) if "joint_limit" in reasons else []
            joint_limit_text = f" joint_limits={joint_limits}" if joint_limits else ""
            print(
                f"[PLAY] episode={completed} env={env_id} terminations={reasons} "
                f"events={events} landing=({info.landing_x:.3f},{info.landing_y:.3f}) "
                f"min_dist={info.min_dist:.3f} hit_center_offset={info.hit_center_offset:.3f} "
                f"hit_vx={info.hit_outgoing_speed:.3f} hit_vz={info.hit_up_speed:.3f} "
                f"post_hit_max_h={info.post_hit_max_height:.3f}{joint_limit_text}"
            )
            if completed >= args_cli.episodes:
                break
        done_env_ids = done.detach().reshape(-1).nonzero(as_tuple=True)[0]
        if done_env_ids.numel() > 0:
            episode_ids[done_env_ids] += 1
            episode_steps[done_env_ids] = 0

    metric_means = {
        name: metric_sums[name] / metric_counts[name]
        for name in metric_names
        if metric_counts[name] > 0
    }
    print(f"[PLAY] completed={completed} counts={counts} metric_means={metric_means}")
    if sim_logger is not None:
        sim_logger.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
