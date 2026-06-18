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
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_updates", type=int, default=30_000)
parser.add_argument("--start_steps", type=int, default=20_000, help="Random-action transitions before SAC updates.")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--replay_size", type=int, default=1_000_000)
parser.add_argument("--event_table_size", type=int, default=250_000)
parser.add_argument("--sampler", choices=("uniform", "stratified"), default="stratified")
parser.add_argument("--updates_per_step", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=1000)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default=None,
    help="Resume SAC network, optimizer, alpha, and update counter from this checkpoint.",
)
parser.add_argument(
    "--resume_random_start",
    action="store_true",
    help="Use random actions during start_steps when resuming. By default resumed warmup uses the loaded policy.",
)
parser.add_argument(
    "--resume_actor_only",
    action="store_true",
    help="Warm-start only the actor and alpha from checkpoint; initialize critics for the current observation shape.",
)
parser.add_argument(
    "--reward_weight_override",
    type=str,
    default=None,
    help="Comma-separated term=weight overrides applied to env_cfg.rewards before env creation, "
    "e.g. 'racket_approach=0,racket_ball_proximity=0'. For reward-ablation experiments; "
    "does not alter the default config.",
)
parser.add_argument(
    "--min_alpha_override",
    type=float,
    default=None,
    help="Override SAC entropy-temperature floor (config.min_alpha) after agent creation/resume. "
    "For entropy-floor ablation; on resume this supersedes the value stored in the checkpoint.",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=10_000,
    help="Run a deterministic (greedy) eval every N updates and save the best checkpoint "
    "(agent_best.pt) by eval valid_return_rate. 0 disables eval/best-checkpoint selection. "
    "Decouples the deployed policy from the persistent valid_return oscillation: the periodic "
    "interval/final checkpoints can land in a bad-hit storm trough, agent_best.pt cannot.",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=150,
    help="Control steps per deterministic eval rollout (>= one episode_length so every env "
    "completes at least one greedy episode). Episode length is ~125 steps at the default dt.",
)
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
    "landing_x",
    "landing_y",
    "hit_center_offset",
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


def _new_reward_term_stats(reward_manager) -> dict:
    return {
        "steps": 0,
        "total_sum": 0.0,
        "term_sums": {name: 0.0 for name in reward_manager.active_terms},
    }


def _accumulate_reward_term_stats(stats: dict, env, reward: torch.Tensor):
    reward_manager = env.unwrapped.reward_manager
    stats["steps"] += 1
    stats["total_sum"] += float(reward.detach().mean().cpu())
    step_reward = reward_manager._step_reward.detach()
    for term_idx, term_name in enumerate(reward_manager.active_terms):
        stats["term_sums"][term_name] += float(step_reward[:, term_idx].mean().cpu())


def _log_reward_term_stats(writer, stats: dict, step: int):
    if writer is None or stats["steps"] <= 0:
        return
    inv_steps = 1.0 / float(stats["steps"])
    writer.add_scalar("reward/total_mean", stats["total_sum"] * inv_steps, step)
    for term_name, value_sum in stats["term_sums"].items():
        writer.add_scalar(f"reward_terms/{term_name}", value_sum * inv_steps, step)


def run_eval(env, agent, eval_steps: int, device) -> tuple[dict, int]:
    """Deterministic greedy rollout used for best-checkpoint selection.

    Resets all envs, rolls the policy out greedily (``deterministic=True``, matching
    deployment) for ``eval_steps`` control steps, and returns per-event episode rates over
    the episodes that completed during the rollout. Greedy actions strip the ``min_alpha``
    exploration noise, so even a single pass over the env's many parallel envs is a
    low-variance estimate of true policy quality -- unlike the stochastic training
    valid_return_rate, which oscillates on the flat reward ridge.

    Does NOT touch the replay buffer or episode-trace bookkeeping; the caller must rebuild
    the training env state afterwards so this perturbation does not leak into training.
    """
    obs, _ = env.reset()
    obs_actor, _ = split_actor_critic_obs(obs)
    stats = _new_episode_stats()
    with torch.no_grad():
        for _ in range(eval_steps):
            action = agent.act(obs_actor, deterministic=True).to(device)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            _accumulate_episode_stats(stats, extract_final_episode_infos(env, done))
            obs_actor, _ = split_actor_critic_obs(next_obs)
    episodes = int(stats["episodes"])
    rates = {
        event: (stats["event_counts"][event] / episodes if episodes > 0 else 0.0)
        for event in EVENT_TAGS
    }
    return rates, episodes


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

    if args_cli.reward_weight_override:
        for pair in args_cli.reward_weight_override.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, _, raw = pair.partition("=")
            name = name.strip()
            term = getattr(env_cfg.rewards, name, None)
            if term is None:
                raise ValueError(f"--reward_weight_override: unknown reward term '{name}'")
            new_w = float(raw)
            print(f"[INFO] reward override: {name} weight {term.weight} -> {new_w}")
            term.weight = new_w

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

    is_resume = args_cli.resume_checkpoint is not None
    resume_step = 0
    if is_resume:
        resume_checkpoint = os.path.abspath(os.path.expanduser(args_cli.resume_checkpoint))
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        if args_cli.resume_actor_only:
            agent = SACAgent.load_actor_only(
                resume_checkpoint,
                actor_obs_dim=obs_actor.shape[-1],
                critic_obs_dim=obs_critic.shape[-1],
                action_dim=action_dim,
                device=device,
            )
        else:
            agent = SACAgent.load(resume_checkpoint, device=device)
        resume_step = int(agent.checkpoint_step)
        if agent.actor_obs_dim != obs_actor.shape[-1]:
            raise ValueError(f"Actor obs dim mismatch: checkpoint={agent.actor_obs_dim}, env={obs_actor.shape[-1]}")
        if agent.critic_obs_dim != obs_critic.shape[-1]:
            raise ValueError(
                f"Critic obs dim mismatch: checkpoint={agent.critic_obs_dim}, env={obs_critic.shape[-1]}. "
                "Use --resume_actor_only to warm-start the actor after changing critic-only observations."
            )
        if agent.action_dim != action_dim:
            raise ValueError(f"Action dim mismatch: checkpoint={agent.action_dim}, env={action_dim}")
        if resume_step >= args_cli.max_updates:
            raise ValueError(f"--max_updates ({args_cli.max_updates}) must be greater than resume step ({resume_step}).")
        resume_mode = "actor-only warm start" if args_cli.resume_actor_only else "full resume"
        print(f"[INFO] Resumed SAC checkpoint: {resume_checkpoint} (step={resume_step}, mode={resume_mode})")
        if args_cli.resume_actor_only:
            print("[INFO] Critics and optimizers were initialized for the current environment observation shape.")
        print("[INFO] Replay buffer is not stored in SAC checkpoints; this run starts with an empty replay buffer.")
    else:
        agent = SACAgent(
            obs_actor.shape[-1],
            obs_critic.shape[-1],
            action_dim,
            config=SACConfig(),
            device=device,
        )

    if args_cli.min_alpha_override is not None:
        new_floor = float(args_cli.min_alpha_override)
        old_floor = agent.config.min_alpha
        agent.config.min_alpha = new_floor
        agent._min_log_alpha = math.log(new_floor) if new_floor > 0.0 else None
        agent._clamp_log_alpha()
        print(
            f"[INFO] min_alpha override: {old_floor} -> {new_floor} "
            f"(current alpha={float(agent.alpha.detach().cpu()):.4f})"
        )

    replay = UniformReplayBuffer(args_cli.replay_size, device="cpu")
    event_tables = make_event_tables(args_cli.event_table_size)
    sampler = StratifiedReplaySampler(replay, event_tables)
    traces = EpisodeTraceBuffer(num_envs)
    env_ids = torch.arange(num_envs, dtype=torch.long)
    episode_steps = torch.zeros(num_envs, dtype=torch.long)

    global_transitions = 0
    update_count = resume_step
    best_valid_return = float("-inf")
    last_losses: dict[str, float] = {}
    episode_stats = _new_episode_stats()
    reward_term_stats = _new_reward_term_stats(env.unwrapped.reward_manager)

    while update_count < args_cli.max_updates:
        if global_transitions < args_cli.start_steps and (not is_resume or args_cli.resume_random_start):
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
        _accumulate_reward_term_stats(reward_term_stats, env, reward)

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
                    _log_reward_term_stats(writer, reward_term_stats, update_count)
                    reward_term_stats = _new_reward_term_stats(env.unwrapped.reward_manager)
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

                if args_cli.eval_interval > 0 and update_count % args_cli.eval_interval == 0:
                    eval_rates, eval_episodes = run_eval(env, agent, args_cli.eval_steps, device)
                    eval_vr = eval_rates.get("valid_return", 0.0)
                    improved = eval_episodes > 0 and eval_vr > best_valid_return
                    if improved:
                        best_valid_return = eval_vr
                        best_ckpt = os.path.join(log_dir, "checkpoints", "agent_best.pt")
                        agent.save(best_ckpt, step=update_count)
                    if writer is not None:
                        for event in EVENT_TAGS:
                            writer.add_scalar(f"eval/{event}_rate", eval_rates[event], update_count)
                        writer.add_scalar("eval/episodes", eval_episodes, update_count)
                        if best_valid_return > float("-inf"):
                            writer.add_scalar("eval/best_valid_return_rate", best_valid_return, update_count)
                    print(
                        f"[EVAL] update={update_count} episodes={eval_episodes} "
                        f"valid_return={eval_vr:.3f} hit={eval_rates['hit']:.3f} "
                        f"bad_hit={eval_rates['bad_hit']:.3f} best={best_valid_return:.3f}"
                        + ("  <- saved agent_best.pt" if improved else "")
                    )
                    # The eval rollout reset and stepped the env under greedy actions; rebuild
                    # a clean training state so the deterministic perturbation does not leak
                    # into the replay buffer / episode-trace bookkeeping.
                    obs, _ = env.reset()
                    obs_actor, obs_critic = split_actor_critic_obs(obs)
                    traces = EpisodeTraceBuffer(num_envs)
                    episode_steps = torch.zeros(num_envs, dtype=torch.long)
                    episode_stats = _new_episode_stats()
                    reward_term_stats = _new_reward_term_stats(env.unwrapped.reward_manager)

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
