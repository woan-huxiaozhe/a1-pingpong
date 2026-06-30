# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate an ``A1-Pingpong-HitTrack`` RSL-RL PPO checkpoint and print per-episode error.

HitTrack is the non-end-to-end hit-reference *tracking* task: the ball is OUT of the MDP, the
arm only learns to drive the FK blade-center to the model-derived reference ``(p_ref, v_ref)``
at the predicted hit instant. Per the design (``docs/sac_vs_ppo_对比.md`` §7) HitTrack is trained
with RSL-RL **PPO** (dense reward + massively parallel sim), via
``scripts/rsl_rl/train.py --task A1-Pingpong-HitTrack`` which saves rsl-rl ``model_*.pt`` runner
checkpoints under ``logs/rsl_rl/a1_tabletennis_hittrack/``.

This loads such a checkpoint with the rsl-rl ``OnPolicyRunner`` (NOT the SAC agent). Crucially it
calls ``handle_deprecated_rsl_rl_cfg`` -- exactly like ``train.py`` -- to migrate the legacy
``HitTrackPPORunnerCfg.policy`` form to the rsl-rl 5.x actor/critic schema; the generic
``scripts/rsl_rl/play.py`` omits this and so dies with ``KeyError: 'class_name'`` on this cfg.

This script:
  * parks the leftover scene ``ball`` far below the world (HitTrack inherits the Catch scene,
    which always spawns a ball; HitTrack never launches it, so it would otherwise free-fall in
    view and could physically bump the arm and corrupt the tracking measurement);
  * runs a fixed number of episodes and, at the END OF EACH EPISODE, prints that hit's
    position / velocity tracking error (vs the CLEAN true-crossing reference) and the success
    verdict (``pos_err < 0.05 m`` and ``vel_err < 0.2 m/s``);
  * prints an aggregate summary with the per-axis (x/y/z) breakdown drained from the env's
    tracking accumulators (the same quantity train.py logs under ``hittrack/*``).

每个 episode 结束打印击球时刻的跟踪误差（位置/速度）与成功判定，并在最后给出逐轴聚合统计。

Usage:
    # zero-action baseline (no checkpoint; verifies ball is hidden + error printing)
    python scripts/rsl_rl/play_hittrack.py --task A1-Pingpong-HitTrack --zero_action \
        --episodes 5 --headless

    # play a trained PPO checkpoint, real-time, one env (headless recommended on laptop GPUs)
    python scripts/rsl_rl/play_hittrack.py --task A1-Pingpong-HitTrack --real-time --episodes 50 \
        --checkpoint logs/rsl_rl/a1_tabletennis_hittrack/<run>/model_xxxx.pt

    # headless multi-env evaluation (lower-variance aggregate stats)
    python scripts/rsl_rl/play_hittrack.py --task A1-Pingpong-HitTrack --num_envs 256 \
        --episodes 256 --headless --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import statistics
import time
from importlib.metadata import version as _pkg_version

from isaaclab.app import AppLauncher  # noqa: E402

# local import (scripts/rsl_rl is on sys.path[0] when run as a script)
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play / evaluate an A1-Pingpong-HitTrack PPO checkpoint.")
parser.add_argument("--task", type=str, default="A1-Pingpong-HitTrack", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (default: play cfg's 1).")
parser.add_argument("--episodes", type=int, default=50, help="Stop after this many completed episodes.")
parser.add_argument("--zero_action", action="store_true", help="Feed zero actions (no checkpoint needed).")
parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions instead of the greedy mean.")
parser.add_argument("--real-time", "--real_time", dest="real_time", action="store_true",
                    help="Sleep after each step to approximate real-time playback.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.reference_commands import pop_hittrack_tracking_stats  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

PARK_OFFSET = (0.0, 0.0, -5.0)  # where to stash the unused ball (env-local), far from arm + camera


def _termination_reasons(raw_env, done: torch.Tensor) -> dict[int, list[str]]:
    """Map each done env -> the list of termination terms that fired this step (best-effort)."""
    manager = getattr(raw_env, "termination_manager", None)
    if manager is None:
        return {}
    done_cpu = done.detach().cpu().bool().reshape(-1)
    reasons: dict[int, list[str]] = {}
    for env_id in done_cpu.nonzero(as_tuple=True)[0].tolist():
        active: list[str] = []
        for term_name in manager.active_terms:
            try:
                term_done = manager.get_term(term_name).detach().cpu().bool().reshape(-1)
            except Exception:
                continue
            if env_id < term_done.numel() and bool(term_done[env_id]):
                active.append(term_name)
        reasons[env_id] = active if active else ["unknown"]
    return reasons


def _park_ball(raw_env, device) -> None:
    """Stash the (unused) scene ball far below the world so it neither shows nor collides.

    Called every step because IsaacLab auto-resets done envs inside ``env.step`` (which respawns
    the ball at its default pose) and gravity nudges it between steps."""
    ball = raw_env.scene["ball"]
    n = raw_env.num_envs
    offset = torch.tensor(PARK_OFFSET, device=device).reshape(1, 3)
    pose = torch.zeros(n, 7, device=device)
    pose[:, :3] = raw_env.scene.env_origins + offset
    pose[:, 3] = 1.0  # quat w (identity)
    ball.write_root_pose_to_sim(pose)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))


def main():
    # --- env (HitTrackPlayEnvCfg via play_env_cfg_entry_point) ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    raw = env.unwrapped
    device = raw.device

    if not hasattr(raw, "_ht_hit_done"):
        raise RuntimeError(
            f"Task '{args_cli.task}' has no HitTrack reference buffers (_ht_*); this play script is "
            "only for A1-Pingpong-HitTrack."
        )

    # --- policy (rsl-rl PPO; skipped for --zero_action) ---
    policy = None
    if args_cli.zero_action:
        print("[PLAY] zero-action baseline (no checkpoint).")
    else:
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        # migrate legacy `policy=...` cfg to the rsl-rl 5.x actor/critic schema (same as train.py);
        # without this OnPolicyRunner construction dies with KeyError: 'class_name'.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[PLAY] Loading PPO checkpoint: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)

    # success thresholds: read straight from the env cfg's update_ref event params (single source)
    _update_params = raw.cfg.events.update_ref.params
    success_pos = float(_update_params.get("success_pos_thresh", 0.05))
    success_vel = float(_update_params.get("success_vel_thresh", 0.2))
    step_dt = float(getattr(raw, "step_dt", 0.01))
    action_dim = env.num_actions

    # initial obs + park the ball before the first frame
    obs = env.get_observations()
    if isinstance(obs, tuple):  # rsl-rl 2.3 returns (obs, extras)
        obs = obs[0]
    _park_ball(raw, device)

    n = raw.num_envs
    # per-episode hit snapshot, latched at the hit step BEFORE IsaacLab's in-step reset wipes the
    # per-env _ht_* buffers (see reference_commands.py:55-66).
    recorded = torch.zeros(n, dtype=torch.bool, device=device)
    pend_pos = torch.full((n,), float("nan"), device=device)
    pend_vel = torch.full((n,), float("nan"), device=device)
    pend_succ = torch.zeros(n, dtype=torch.bool, device=device)
    pend_tau = torch.full((n,), float("nan"), device=device)

    # host-side aggregates over completed episodes
    ep_pos: list[float] = []
    ep_vel: list[float] = []
    ep_succ: list[bool] = []
    no_hit = 0
    completed = 0

    print(f"[PLAY] task={args_cli.task} num_envs={n} episodes={args_cli.episodes} "
          f"success_thresh=(pos<{success_pos} m, vel<{success_vel} m/s)")

    while completed < args_cli.episodes and simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if args_cli.zero_action:
                actions = torch.zeros(n, action_dim, device=device)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        _park_ball(raw, device)

        # 1) snapshot newly-latched hits (hit_done flips False->True at the hit step)
        hit_done = raw._ht_hit_done
        newly = hit_done & (~recorded)
        if bool(newly.any()):
            sel = newly.nonzero(as_tuple=True)[0]
            pend_pos[sel] = raw._ht_pos_err_at_hit[sel]
            pend_vel[sel] = raw._ht_vel_err_at_hit[sel]
            pend_succ[sel] = raw._ht_success[sel]
            pend_tau[sel] = raw._ht_tau_true[sel]
            recorded[sel] = True

        # 2) at episode end, print the latched error for each finished env
        done = dones.detach().bool().reshape(-1)
        if bool(done.any()):
            reasons = _termination_reasons(raw, done)
            for env_id in done.nonzero(as_tuple=True)[0].tolist():
                completed += 1
                term = reasons.get(env_id, ["unknown"])
                if bool(recorded[env_id]):
                    pe = float(pend_pos[env_id])
                    ve = float(pend_vel[env_id])
                    ok = bool(pend_succ[env_id])
                    tau = float(pend_tau[env_id])
                    ep_pos.append(pe)
                    ep_vel.append(ve)
                    ep_succ.append(ok)
                    print(
                        f"[HITTRACK] ep={completed} env={env_id} hit=Y success={'Y' if ok else 'N'} "
                        f"pos_err={pe:.4f} m vel_err={ve:.3f} m/s tau@hit={tau:+.3f} s term={term}"
                    )
                else:
                    no_hit += 1
                    print(f"[HITTRACK] ep={completed} env={env_id} hit=N (no hit latched) term={term}")
                if completed >= args_cli.episodes:
                    break
            done_idx = done.nonzero(as_tuple=True)[0]
            recorded[done_idx] = False
            pend_pos[done_idx] = float("nan")
            pend_vel[done_idx] = float("nan")
            pend_succ[done_idx] = False
            pend_tau[done_idx] = float("nan")

        if args_cli.real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # --- aggregate summary ---
    print("\n========== HITTRACK PLAY SUMMARY ==========")
    print(f"episodes completed = {completed}  (hit={len(ep_pos)}, no-hit={no_hit})")
    if ep_pos:
        succ_rate = sum(1 for s in ep_succ if s) / len(ep_succ)
        print(f"success_rate        = {succ_rate * 100:.1f}%  (pos<{success_pos} m & vel<{success_vel} m/s)")
        print(f"pos_err [m]   mean={statistics.fmean(ep_pos):.4f}  "
              f"median={statistics.median(ep_pos):.4f}  max={max(ep_pos):.4f}")
        print(f"vel_err [m/s] mean={statistics.fmean(ep_vel):.3f}  "
              f"median={statistics.median(ep_vel):.3f}  max={max(ep_vel):.3f}")
    stats = pop_hittrack_tracking_stats(raw)
    if stats is not None:
        print(f"per-axis |Δ| over {stats['hit_count']} hits:")
        print(f"  pos_err  x={stats['pos_err_x']:.4f}  y={stats['pos_err_y']:.4f}  z={stats['pos_err_z']:.4f}  (m)")
        print(f"  vel_err  x={stats['vel_err_x']:.3f}  y={stats['vel_err_y']:.3f}  z={stats['vel_err_z']:.3f}  (m/s)")
        print("  read: z big -> reach/height; x big -> arrive_time/timing; y big -> lateral centering")
    print("===========================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
