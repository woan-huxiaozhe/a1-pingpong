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
parser.add_argument("--record", type=str, nargs="?", const="__auto__", default="__auto__",
                    help="Record a per-step / per-env CSV of policy target + sim joint state + "
                         "observed end-effector state + reference command (for offline plotting). "
                         "Pass a path, or bare --record to auto-name next to the checkpoint. "
                         "One row per (env, control step) -- large num_envs => large file.")
parser.add_argument("--no_record", action="store_true", help="Disable the data-recording CSV.")
parser.add_argument("--smoothing", type=float, default=None,
                    help="Override the action-term target smoothing (EMA factor) at inference. For "
                         "zero-cost A/B swing-speed checks WITHOUT retraining (the policy net output "
                         "is unchanged; only the action->joint-target mapping changes).")
parser.add_argument("--action_scale", type=float, default=None,
                    help="Override the action-term action_scale at inference (same A/B purpose).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.a1_pingpong_hittrack.env_cfg import RACKET_BODY_NAME  # noqa: E402
from unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.reference_commands import pop_hittrack_tracking_stats  # noqa: E402
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state, racket_normal  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


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


_VEC3 = ("x", "y", "z")


class _HitTrackRecorder:
    """Accumulate a per-step / per-env snapshot and dump one tidy CSV for offline visualization
    of the end-effector tracking error and the joint state.

    One row per (env, control step). All quantities are read AFTER ``env.step`` (so the joint /
    racket columns are the realized sim state that resulted from the target applied that step;
    the reference columns are the cursor the env advanced to). Columns -- ``J`` arm joints, in
    action-term order ``RIGHT_ARM_JOINT_NAMES``:

        g_step,env,ep,t            loop iter, env id, per-env episode index, step-within-episode
        tau_true,tau_noisy         countdown to the hit instant (clean / observed) [s]
        q_tgt_<j>                  policy TARGET joint pos sent to set_joint_position_target
                                   (after smoothing / velocity / limit clamps) [rad]
        q_<j>                      sim ACTUAL joint pos [rad]
        qd_<j>                     sim ACTUAL joint vel [rad/s]
        torque_<j>                 sim ACTUAL applied (PhysX-clamped) joint torque [N·m]
        act_<j>                    raw policy action (normalized joint delta, pre-processing)
        ee_{x,y,z}                 observed FK blade-center pos (env-local) [m]
        ee_v{x,y,z}                observed blade-center vel [m/s]
        rn_{x,y,z}                 observed racket face-normal (actual blade orientation) [unit]
        pref_{x,y,z}               estimate hit command pos -- noisy/deployable (actor sees) [m]
        vref_{x,y,z}               estimate hit command vel -- noisy [m/s]
        nref_{x,y,z}               estimate hit command face-normal -- noisy [unit]
        pref_c{x,y,z}              clean/true reference pos (privileged ground truth) [m]
        vref_c{x,y,z}              clean/true reference vel [m/s]
        perr_{x,y,z},perr          pos tracking error (ee - clean pref) + L2 norm [m]
        verr_{x,y,z},verr          vel tracking error (ee_v - clean vref) + L2 norm [m/s]
        hit_done,success           1.0 once the hit step latched / once it passed both thresholds
    """

    def __init__(self, path: str, joint_names: list[str]):
        self.path = path
        self.joint_names = list(joint_names)
        self._blocks: list[np.ndarray] = []  # each [n, D]
        cols = ["g_step", "env", "ep", "t", "tau_true", "tau_noisy"]
        for tag in ("q_tgt", "q", "qd", "torque", "act"):
            cols += [f"{tag}_{name}" for name in self.joint_names]
        cols += [f"ee_{a}" for a in _VEC3] + [f"ee_v{a}" for a in _VEC3] + [f"rn_{a}" for a in _VEC3]
        cols += [f"pref_{a}" for a in _VEC3] + [f"vref_{a}" for a in _VEC3] + [f"nref_{a}" for a in _VEC3]
        cols += [f"pref_c{a}" for a in _VEC3] + [f"vref_c{a}" for a in _VEC3]
        cols += [f"perr_{a}" for a in _VEC3] + ["perr"] + [f"verr_{a}" for a in _VEC3] + ["verr"]
        cols += ["hit_done", "success"]
        self.columns = cols

    @torch.no_grad()
    def capture(self, env, action_term, robot, joint_ids, racket_body_name, g_step, ep_ids):
        n = env.num_envs
        q_tgt = action_term.processed_actions                       # [n,J] applied target
        raw_act = action_term.raw_actions                           # [n,J] policy raw delta
        q = robot.data.joint_pos[:, joint_ids]
        qd = robot.data.joint_vel[:, joint_ids]
        torque = robot.data.applied_torque[:, joint_ids]
        center, center_vel, _ = _racket_body_state(env, racket_body_name)
        ee = center - env.scene.env_origins                         # env-local blade center
        rn = racket_normal(env, racket_body_name)                   # actual blade face-normal (unit)
        p_c, v_c = env._ht_p_ref_clean, env._ht_v_ref_clean
        perr, verr = ee - p_c, center_vel - v_c
        block = torch.cat(
            [
                env._ht_tau_true.unsqueeze(-1), env._ht_tau_noisy.unsqueeze(-1),
                q_tgt, q, qd, torque, raw_act,
                ee, center_vel, rn,
                env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy,
                p_c, v_c,
                perr, perr.norm(dim=-1, keepdim=True),
                verr, verr.norm(dim=-1, keepdim=True),
                env._ht_hit_done.float().unsqueeze(-1), env._ht_success.float().unsqueeze(-1),
            ],
            dim=-1,
        ).detach().to("cpu", torch.float32).numpy()
        meta = np.empty((n, 4), dtype=np.float64)
        meta[:, 0] = g_step
        meta[:, 1] = np.arange(n)
        meta[:, 2] = ep_ids
        meta[:, 3] = env.episode_length_buf.detach().cpu().numpy()
        self._blocks.append(np.concatenate([meta, block.astype(np.float64)], axis=1))

    def save(self) -> None:
        if not self._blocks:
            print("[RECORD] no rows captured; nothing written.")
            return
        data = np.concatenate(self._blocks, axis=0)
        out_dir = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(out_dir, exist_ok=True)
        fmt = ["%d", "%d", "%d", "%d"] + ["%.6g"] * (data.shape[1] - 4)
        np.savetxt(self.path, data, delimiter=",", header=",".join(self.columns), comments="", fmt=fmt)
        print(f"[RECORD] wrote {data.shape[0]} rows x {data.shape[1]} cols -> {self.path}")


def main():
    # --- env (HitTrackPlayEnvCfg via play_env_cfg_entry_point) ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # inference-side action-term overrides (A/B swing-speed checks; do not affect the policy net)
    if args_cli.smoothing is not None:
        env_cfg.actions.right_arm.smoothing = args_cli.smoothing
        print(f"[PLAY] OVERRIDE action smoothing -> {args_cli.smoothing}")
    if args_cli.action_scale is not None:
        env_cfg.actions.right_arm.action_scale = args_cli.action_scale
        print(f"[PLAY] OVERRIDE action_scale -> {args_cli.action_scale}")
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
    resume_path = None
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

    # --- data recorder (per-step / per-env joint + EE + reference snapshot CSV) ---
    recorder = None
    action_term = robot = joint_ids = None
    ep_ids = np.zeros(n, dtype=np.int64)  # per-env episode index (advanced on the done step)
    g_step = 0
    if not args_cli.no_record:
        if args_cli.record and args_cli.record != "__auto__":
            record_path = args_cli.record
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            if resume_path:
                record_path = f"{os.path.splitext(resume_path)[0]}_play_record_{ts}.csv"
            else:
                record_path = os.path.join("logs", "rsl_rl", "hittrack_play_records", f"zero_action_{ts}.csv")
        action_term = raw.action_manager.get_term("right_arm")
        robot = raw.scene["robot"]
        joint_ids = action_term._joint_ids
        recorder = _HitTrackRecorder(record_path, action_term._joint_names)
        print(f"[RECORD] enabled -> {record_path} ({len(recorder.columns)} cols/row)")

    while completed < args_cli.episodes and simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if args_cli.zero_action:
                actions = torch.zeros(n, action_dim, device=device)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # 0) record this step's snapshot (read post-step; done envs were auto-reset inside step, so
        #    their row is the next episode's frame -- bump ep_id for them BEFORE capturing).
        if recorder is not None:
            done_cpu = dones.detach().reshape(-1).bool().cpu().numpy()
            ep_ids[done_cpu] += 1
            recorder.capture(raw, action_term, robot, joint_ids, RACKET_BODY_NAME, g_step, ep_ids)
        g_step += 1

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

    if recorder is not None:
        recorder.save()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
