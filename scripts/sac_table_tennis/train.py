"""Train the independent A1 table-tennis SAC catch policy."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from datetime import datetime

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train A1-TableTennis-SAC-Catch with custom PyTorch SAC.")
parser.add_argument("--task", type=str, default="A1-TableTennis-SAC-Catch")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_updates", type=int, default=10_000)
parser.add_argument("--start_steps", type=int, default=20_000, help="Random-action transitions before SAC updates.")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--replay_size", type=int, default=1_000_000)
parser.add_argument("--event_table_size", type=int, default=250_000)
parser.add_argument("--sampler", choices=("uniform", "stratified"), default="stratified")
parser.add_argument("--updates_per_step", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=1000)
parser.add_argument("--log_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TAGS, EVENT_TO_BIT  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.replay import (  # noqa: E402
    EpisodeTraceBuffer,
    StratifiedReplaySampler,
    UniformReplayBuffer,
    make_event_tables,
    table_sizes,
)
from unitree_rl_lab.tasks.table_tennis_sac.runtime import (  # noqa: E402
    extract_final_episode_infos,
    split_actor_critic_obs,
)
from unitree_rl_lab.tasks.table_tennis_sac.sac import SACAgent, SACConfig  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


def _make_log_dir() -> str:
    if args_cli.log_dir is not None:
        return os.path.abspath(args_cli.log_dir)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(os.path.join("logs", "sac_table_tennis", args_cli.task, stamp))


EPISODE_METRICS = (
    "min_dist",
    "hit_outgoing_speed",
    "hit_up_speed",
    "post_hit_max_outgoing_speed",
    "post_hit_max_height",
)


def _new_episode_stats() -> dict:
    return {
        "episodes": 0,
        "event_counts": {name: 0 for name in EVENT_TAGS},
        "metric_sums": defaultdict(float),
        "metric_counts": defaultdict(int),
    }


def _add_metric(stats: dict, name: str, value: float):
    if math.isfinite(value):
        stats["metric_sums"][name] += float(value)
        stats["metric_counts"][name] += 1


def _accumulate_episode_stats(stats: dict, infos):
    for info in infos.values():
        stats["episodes"] += 1
        for event in EVENT_TAGS:
            if info.event_mask & EVENT_TO_BIT[event]:
                stats["event_counts"][event] += 1
        for metric in EPISODE_METRICS:
            _add_metric(stats, metric, getattr(info, metric))


def _log_episode_stats(writer, stats: dict, step: int) -> str:
    episodes = int(stats["episodes"])
    if writer is not None:
        writer.add_scalar("episode/count", episodes, step)

    if episodes <= 0:
        return "episodes=0"

    rates: dict[str, float] = {}
    for event in EVENT_TAGS:
        rate = stats["event_counts"][event] / episodes
        rates[event] = rate
        if writer is not None:
            writer.add_scalar(f"episode/{event}_rate", rate, step)

    for metric in EPISODE_METRICS:
        count = stats["metric_counts"][metric]
        if count <= 0:
            continue
        mean = stats["metric_sums"][metric] / count
        if writer is not None:
            writer.add_scalar(f"episode/{metric}_mean", mean, step)

    return (
        f"episodes={episodes} hit_rate={rates['hit']:.3f} return_rate={rates['return']:.3f} "
        f"valid_return_rate={rates['valid_return']:.3f} bad_hit_rate={rates['bad_hit']:.3f}"
    )


def main():
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed

    log_dir = _make_log_dir()
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    print(f"[INFO] Logging SAC run to: {log_dir}")
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else None
    if writer is None:
        print("[WARN] torch.utils.tensorboard is unavailable; TensorBoard scalars will not be written.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    obs_actor, obs_critic = split_actor_critic_obs(obs)
    num_envs = obs_actor.shape[0]
    action_dim = int(env.action_space.shape[-1])
    device = obs_actor.device

    agent = SACAgent(
        obs_actor.shape[-1],
        obs_critic.shape[-1],
        action_dim,
        config=SACConfig(),
        device=device,
    )
    replay = UniformReplayBuffer(args_cli.replay_size, device="cpu")
    event_tables = make_event_tables(args_cli.event_table_size)
    sampler = StratifiedReplaySampler(replay, event_tables)
    traces = EpisodeTraceBuffer(num_envs)
    env_ids = torch.arange(num_envs, dtype=torch.long)
    episode_steps = torch.zeros(num_envs, dtype=torch.long)

    global_transitions = 0
    update_count = 0
    last_losses: dict[str, float] = {}
    episode_stats = _new_episode_stats()

    while update_count < args_cli.max_updates:
        if global_transitions < args_cli.start_steps:
            action = torch.empty(num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        else:
            action = agent.act(obs_actor, deterministic=False).to(device)
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite action produced by policy.")

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        next_actor, next_critic = split_actor_critic_obs(next_obs)
        if not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite reward returned by environment.")

        indices, versions = replay.add_batch(
            obs_actor=obs_actor,
            obs_critic=obs_critic,
            action=action,
            reward=reward,
            next_obs_actor=next_actor,
            next_obs_critic=next_critic,
            done=done,
            episode_id=traces.episode_ids,
            step_index=episode_steps,
        )
        traces.append(env_ids, indices, versions)

        final_infos = extract_final_episode_infos(env, done)
        _accumulate_episode_stats(episode_stats, final_infos)
        for env_id, info in final_infos.items():
            windows = traces.finalize(env_id, info)
            for event, (event_indices, event_versions) in windows.items():
                event_tables[event].add(event_indices, event_versions)
                replay.or_event_masks(event_indices, EVENT_TO_BIT[event])

        episode_steps += 1
        if final_infos:
            done_ids = torch.tensor(list(final_infos.keys()), dtype=torch.long)
            episode_steps[done_ids] = 0

        obs_actor, obs_critic = next_actor, next_critic
        global_transitions += num_envs

        if len(replay) >= args_cli.batch_size and global_transitions >= args_cli.start_steps:
            for _ in range(args_cli.updates_per_step):
                if args_cli.sampler == "stratified":
                    batch, composition = sampler.sample(args_cli.batch_size, device=device)
                else:
                    batch = replay.sample(args_cli.batch_size, device=device)
                    composition = {"uniform": args_cli.batch_size}
                last_losses = agent.update(batch)
                if not all(torch.isfinite(torch.tensor(v)) for v in last_losses.values()):
                    raise RuntimeError(f"Non-finite SAC loss: {last_losses}")
                update_count += 1
                if writer is not None:
                    for key, value in last_losses.items():
                        writer.add_scalar(f"loss/{key}", value, update_count)
                    writer.add_scalar("replay/size", len(replay), update_count)
                    writer.add_scalar("train/transitions", global_transitions, update_count)

                if update_count % args_cli.log_interval == 0:
                    sizes = table_sizes(event_tables, replay)
                    episode_summary = _log_episode_stats(writer, episode_stats, update_count)
                    episode_stats = _new_episode_stats()
                    if writer is not None:
                        for key, value in sizes.items():
                            writer.add_scalar(f"event_table/{key}", value, update_count)
                        for key, value in composition.items():
                            writer.add_scalar(f"batch/{key}", value, update_count)
                    print(
                        f"[SAC] update={update_count} transitions={global_transitions} "
                        f"replay={len(replay)} alpha={last_losses['alpha']:.4f} "
                        f"critic={last_losses['critic_loss']:.4f} actor={last_losses['actor_loss']:.4f} "
                        f"tables={sizes} batch={composition} {episode_summary}"
                    )

                if update_count % args_cli.checkpoint_interval == 0:
                    ckpt = os.path.join(log_dir, "checkpoints", f"agent_{update_count:07d}.pt")
                    agent.save(ckpt, step=update_count)

                if update_count >= args_cli.max_updates:
                    break

    final_ckpt = os.path.join(log_dir, "checkpoints", "agent_final.pt")
    agent.save(final_ckpt, step=update_count)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"[INFO] Finished {update_count} SAC updates. Final checkpoint: {final_ckpt}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
