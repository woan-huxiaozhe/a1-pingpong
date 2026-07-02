"""HitTrack reference-command manager (env-facing).

Owns the per-episode end-effector reference buffers (``_ht_*``) the HitTrack task tracks.
At reset it samples a serve (synthetic source here; baked-real source added in a later task),
runs the runtime planner once over the whole noisy ball stream + the clean ball state
(the planner is memoryless, so reset-batch == per-step), and seeds the cursor-0 row. The
``mode="interval"`` ``update_hit_track_state`` advances the cursor each control step, copies the
current noisy/clean references + ``tau`` into the live buffers, and latches success at the hit
step. Only the buffer math lives here; the tracking-reward kernels are in ``mdp.tracking``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .reference_planner import plan_hit_reference
from .reference_source import (
    is_reachable,
    phase_scaled_ball_noise,
    sample_hit_ball_states,
    sample_lateral_shift,
    tau_streams,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ensure_ht_runtime_buffers(env):
    """Create the per-step ("current row") reference + success/flag buffers that the HitTrack
    observation, reward and termination terms read every step. These are ``n_steps``-independent
    (``[n,3]`` / ``[n]``), so this is safe to call BEFORE the first reset -- which matters because
    the ``ObservationManager`` probes each obs term's output shape by calling its ``func`` once at
    env construction, before any reset event has populated the ``_ht_*`` buffers. Idempotent: a
    no-op once the buffers exist (the reset overwrites their contents in place)."""
    if hasattr(env, "_ht_p_ref_clean"):
        return
    n, device = env.num_envs, env.device

    def z3():
        return torch.zeros(n, 3, device=device)

    env._ht_p_ref_clean, env._ht_v_ref_clean, env._ht_n_ref_clean = z3(), z3(), z3()
    env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy = z3(), z3(), z3()
    env._ht_tau_true = torch.zeros(n, device=device)
    env._ht_tau_noisy = torch.zeros(n, device=device)
    env._ht_hit_done = torch.zeros(n, dtype=torch.bool, device=device)
    env._ht_success = torch.zeros(n, dtype=torch.bool, device=device)
    env._ht_pos_err_at_hit = torch.full((n,), float("nan"), device=device)
    env._ht_vel_err_at_hit = torch.full((n,), float("nan"), device=device)

    # --- end-effector tracking-error accumulators (global, NOT per-env) ---
    # Filled at the hit instant in ``update_hit_track_state`` and drained by the train loop via
    # ``pop_hittrack_tracking_stats``. They are scalars/[3] not indexed by env, so the per-env
    # reset (which IsaacLab runs inside ``step`` before it returns) cannot wipe them -- this is why
    # the hit error is summed here at tau~=0 rather than read back at episode ``done``.
    env._ht_acc_n = torch.zeros((), dtype=torch.long, device=device)  # hits in the window
    env._ht_acc_success = torch.zeros((), dtype=torch.long, device=device)
    env._ht_acc_pe_sum = torch.zeros((), device=device)  # sum ||p_racket - p_ref_clean||
    env._ht_acc_ve_sum = torch.zeros((), device=device)  # sum ||v_racket - v_ref_clean||
    env._ht_acc_pe_abs = torch.zeros(3, device=device)  # sum |Δp| per axis (x,y,z)
    env._ht_acc_ve_abs = torch.zeros(3, device=device)  # sum |Δv| per axis (x,y,z)
    env._ht_acc_ne_deg_sum = torch.zeros((), device=device)  # sum angle(n_racket, n_ref_clean) [deg]
    env._ht_acc_ne_dot_sum = torch.zeros((), device=device)  # sum dot(n_racket, n_ref_clean)


def pop_hittrack_tracking_stats(env):
    """Drain the windowed end-effector tracking-error accumulators (means over every hit since the
    last call), then zero them. Returns ``None`` when the HitTrack buffers were never created (e.g.
    the Catch task) or no hit landed in the window. Errors are measured at the hit instant
    (``tau~=0``) against the CLEAN true-crossing reference -- the same quantity the success gate
    uses, and ~= the commanded (noisy) reference there since the KF noise vanishes at the hit."""
    if not hasattr(env, "_ht_acc_n"):
        return None
    n = int(env._ht_acc_n.item())
    if n == 0:
        return None
    inv = 1.0 / n
    pe_abs = (env._ht_acc_pe_abs * inv).tolist()
    ve_abs = (env._ht_acc_ve_abs * inv).tolist()
    stats = {
        "hit_count": n,
        "success_rate": float(env._ht_acc_success.item()) * inv,
        "pos_err_total": float(env._ht_acc_pe_sum.item()) * inv,
        "vel_err_total": float(env._ht_acc_ve_sum.item()) * inv,
        "normal_err_deg": float(env._ht_acc_ne_deg_sum.item()) * inv,
        "normal_dot": float(env._ht_acc_ne_dot_sum.item()) * inv,
        "normal_align_score": 0.5 * (1.0 + float(env._ht_acc_ne_dot_sum.item()) * inv),
        "pos_err_x": pe_abs[0], "pos_err_y": pe_abs[1], "pos_err_z": pe_abs[2],
        "vel_err_x": ve_abs[0], "vel_err_y": ve_abs[1], "vel_err_z": ve_abs[2],
    }
    for buf in (env._ht_acc_n, env._ht_acc_success, env._ht_acc_pe_sum,
                env._ht_acc_ve_sum, env._ht_acc_pe_abs, env._ht_acc_ve_abs,
                env._ht_acc_ne_deg_sum, env._ht_acc_ne_dot_sum):
        buf.zero_()
    return stats


def _ensure_ht_buffers(env, n_steps: int):
    n, device = env.num_envs, env.device
    if hasattr(env, "_ht_p_ref_noisy_stream") and getattr(env, "_ht_n_steps", None) == n_steps:
        return
    env._ht_n_steps = n_steps
    ensure_ht_runtime_buffers(env)  # current-row + flag buffers (may already exist from the obs probe)

    def zT3():
        return torch.zeros(n, n_steps, 3, device=device)

    env._ht_p_ref_noisy_stream, env._ht_v_ref_noisy_stream, env._ht_n_ref_noisy_stream = zT3(), zT3(), zT3()
    env._ht_tau_true_stream = torch.zeros(n, n_steps, device=device)
    env._ht_tau_noisy_stream = torch.zeros(n, n_steps, device=device)
    env._ht_valid_len = torch.full((n,), n_steps, dtype=torch.long, device=device)
    env._ht_hit_step = torch.zeros(n, dtype=torch.long, device=device)


def load_baked_references(path: str, device) -> dict:
    """Load a baked ``hittrack_references.npz`` (see ``bake_hittrack_references.py``) into tensors
    on ``device``: ``noisy_ball_stream[S,T,5]``, ``clean_ball_state[S,6]``, ``tau_true[S,T]``,
    ``valid_len[S]``, ``reachable[S]``."""
    data = np.load(path)
    return {
        "noisy_ball_stream": torch.as_tensor(np.asarray(data["noisy_ball_stream"]), dtype=torch.float32, device=device),
        "clean_ball_state": torch.as_tensor(np.asarray(data["clean_ball_state"]), dtype=torch.float32, device=device),
        "tau_true": torch.as_tensor(np.asarray(data["tau_true"]), dtype=torch.float32, device=device),
        "valid_len": torch.as_tensor(np.asarray(data["valid_len"]), dtype=torch.long, device=device),
        "reachable": torch.as_tensor(np.asarray(data["reachable"]), dtype=torch.bool, device=device),
    }


def reset_reference_command(
    env,
    env_ids,
    *,
    hit_plane_x,
    target_xyz,
    box,
    max_prep_s,
    post_margin_s,
    step_dt,
    reach_y_range,
    reach_z_range,
    restitution=None,
    noise_params=None,
    baked=None,
    baked_path=None,
):
    n_steps = int(round(max_prep_s / step_dt)) + int(round(post_margin_s / step_dt)) + 1
    _ensure_ht_buffers(env, n_steps)
    env._ht_post_margin_steps = int(round(post_margin_s / step_dt))
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    k = ids.shape[0]
    device = env.device

    # lazily load + cache the baked reference table (curriculum (3)) when only a path is given
    if baked is None and baked_path is not None:
        if not hasattr(env, "_ht_baked"):
            env._ht_baked = load_baked_references(baked_path, device)
        baked = env._ht_baked

    if baked is None:
        # --- synthetic source (curriculum (1)/(2)): sample reachable hit-plane ball states ---
        clean = sample_hit_ball_states(k, box, hit_plane_x, device=device)
        bad = ~is_reachable(clean[:, :3], reach_y_range, reach_z_range)
        for _ in range(8):
            if not bool(bad.any()):
                break
            clean[bad] = sample_hit_ball_states(int(bad.sum()), box, hit_plane_x, device=device)
            bad = ~is_reachable(clean[:, :3], reach_y_range, reach_z_range)

        tau_initial = torch.full((k,), float(max_prep_s), device=device)  # E1: capped synthetic horizon
        tau_true_stream = tau_streams(tau_initial, n_steps, step_dt)  # [k,T]
        noisy_stream = clean.unsqueeze(1).repeat(1, n_steps, 1)  # [k,T,6]
        if noise_params is not None:
            bias_unit = torch.nn.functional.normalize(torch.randn(k, 5, device=device), dim=-1)
            noise5 = phase_scaled_ball_noise(tau_true_stream.clamp(min=0.0), bias_unit, **noise_params)  # [k,T,5]
            noisy_stream[..., 1:6] += noise5
        valid_len = torch.full((k,), n_steps, dtype=torch.long, device=device)
    else:
        # --- baked real source (curriculum (3)): sample a recorded serve per env ---
        s_total = baked["clean_ball_state"].shape[0]
        idx = torch.randint(0, s_total, (k,), device=device)
        clean = baked["clean_ball_state"][idx]  # [k,6]
        noisy5 = baked["noisy_ball_stream"][idx]  # [k,Tb,5] = (y,z,vx,vy,vz)
        tau_baked = baked["tau_true"][idx]  # [k,Tb]
        valid_baked = baked["valid_len"][idx]  # [k]
        length = min(noisy5.shape[1], n_steps)
        # noisy ball stream: hold the clean state, overwrite the valid window with the baked KF-pred
        noisy_stream = clean.unsqueeze(1).repeat(1, n_steps, 1)  # [k,T,6] (x col stays hit-plane)
        noisy_stream[:, :length, 1:6] = noisy5[:, :length, :]
        # baked tau is a uniform countdown (t_cross - grid); reproduce + linearly extend to n_steps
        tau_true_stream = tau_streams(tau_baked[:, 0], n_steps, step_dt)
        valid_len = valid_baked.clamp(max=n_steps).to(torch.long)

        # lateral (y) data-augmentation: real serves cluster in the central ~half of the reach band,
        # so shift each one to a uniform y target across the full band (pure translation preserves
        # vy / timing / z). Applied to clean + every noisy-stream row by the same Δy => consistent.
        dy = sample_lateral_shift(clean[:, 1], reach_y_range)  # [k]
        clean[:, 1] = clean[:, 1] + dy
        noisy_stream[:, :, 1] = noisy_stream[:, :, 1] + dy.unsqueeze(1)

    hit_step = torch.round(tau_true_stream[:, 0] / step_dt).long()
    tau_noisy_stream = tau_true_stream.clone()

    target = torch.tensor(target_xyz, device=device).reshape(1, 3).repeat(k, 1)
    # plan clean (constant) and noisy (per-step) references at reset (planner is memoryless)
    plan_kw = {} if restitution is None else {"restitution": restitution}
    pc, vc, nc = plan_hit_reference(clean[:, :3], clean[:, 3:6], target, **plan_kw)
    flat = noisy_stream.reshape(k * n_steps, 6)
    target_flat = target.repeat_interleave(n_steps, dim=0)
    pn, vn, nn = plan_hit_reference(flat[:, :3], flat[:, 3:6], target_flat, **plan_kw)

    env._ht_p_ref_clean[ids], env._ht_v_ref_clean[ids], env._ht_n_ref_clean[ids] = pc, vc, nc
    env._ht_p_ref_noisy_stream[ids] = pn.reshape(k, n_steps, 3)
    env._ht_v_ref_noisy_stream[ids] = vn.reshape(k, n_steps, 3)
    env._ht_n_ref_noisy_stream[ids] = nn.reshape(k, n_steps, 3)
    env._ht_tau_true_stream[ids] = tau_true_stream
    env._ht_tau_noisy_stream[ids] = tau_noisy_stream
    env._ht_valid_len[ids] = valid_len
    env._ht_hit_step[ids] = hit_step
    env._ht_hit_done[ids] = False
    env._ht_success[ids] = False
    env._ht_pos_err_at_hit[ids] = float("nan")
    env._ht_vel_err_at_hit[ids] = float("nan")
    # initialize current row at cursor 0
    env._ht_p_ref_noisy[ids] = env._ht_p_ref_noisy_stream[ids, 0]
    env._ht_v_ref_noisy[ids] = env._ht_v_ref_noisy_stream[ids, 0]
    env._ht_n_ref_noisy[ids] = env._ht_n_ref_noisy_stream[ids, 0]
    env._ht_tau_noisy[ids] = env._ht_tau_noisy_stream[ids, 0]
    env._ht_tau_true[ids] = env._ht_tau_true_stream[ids, 0]


def update_hit_track_state(
    env,
    env_ids,
    *,
    success_pos_thresh,
    success_vel_thresh,
    racket_pos=None,
    racket_vel=None,
):
    step = env.episode_length_buf.to(torch.long).clamp(min=0)
    # p/v/n references: HELD at the converged hit value past the valid window (the ball is out of
    # the MDP, so no post-crossing prediction exists). This freezes the *target point* at p_hit.
    cursor = torch.minimum(step, env._ht_valid_len - 1)
    # tau: NOT clamped to valid_len -- let it count through 0 into negative during the post-hit
    # margin so the Gaussian time-gate CLOSES symmetrically (follow-through) instead of being pinned
    # open ~12 steps at gate~=0.98. ``tau_true_stream`` already extends linearly negative (see
    # ``tau_streams``). Without this the identical static ``p_ref`` is rewarded ~13x over the held
    # window, which makes "park at p_ref" beat "swing through it" -- the root cause of the near-zero
    # contact velocity. Only the reward gate + observed tau change here; the success latch and
    # ``hit_window_elapsed`` termination key off ``step``/``_ht_hit_step`` directly, so the
    # measurement instant and episode length are unchanged.
    tau_cursor = step.clamp(max=env._ht_n_steps - 1)
    arange = torch.arange(env.num_envs, device=env.device)
    env._ht_p_ref_noisy = env._ht_p_ref_noisy_stream[arange, cursor]
    env._ht_v_ref_noisy = env._ht_v_ref_noisy_stream[arange, cursor]
    env._ht_n_ref_noisy = env._ht_n_ref_noisy_stream[arange, cursor]
    env._ht_tau_noisy = env._ht_tau_noisy_stream[arange, tau_cursor]
    env._ht_tau_true = env._ht_tau_true_stream[arange, tau_cursor]

    at_hit = (step == env._ht_hit_step) & (~env._ht_hit_done)
    if bool(at_hit.any()):
        # Fetch the racket state lazily -- only needed at the hit step -- so the per-step cursor
        # advance stays Isaac-free (the deferred observations import pulls in isaaclab/USD).
        from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import RACKET_BODY_NAME
        from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import racket_normal

        if racket_pos is None or racket_vel is None:
            from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

            center, center_vel, _ = _racket_body_state(env, RACKET_BODY_NAME)
            racket_pos = center - env.scene.env_origins
            racket_vel = center_vel

        racket_n = racket_normal(env, RACKET_BODY_NAME)

        pe = torch.norm(racket_pos - env._ht_p_ref_clean, dim=-1)
        ve = torch.norm(racket_vel - env._ht_v_ref_clean, dim=-1)
        ok = (pe < success_pos_thresh) & (ve < success_vel_thresh)
        env._ht_pos_err_at_hit[at_hit] = pe[at_hit]
        env._ht_vel_err_at_hit[at_hit] = ve[at_hit]
        env._ht_success[at_hit] = ok[at_hit]
        env._ht_hit_done[at_hit] = True

        # accumulate this step's hits into the global tracking-error window (drained by the train
        # loop). Per-axis errors are |Δ| against the clean reference; kept on-device (no host sync).
        dp = (racket_pos - env._ht_p_ref_clean)[at_hit]
        dv = (racket_vel - env._ht_v_ref_clean)[at_hit]
        dot = (racket_n[at_hit] * env._ht_n_ref_clean[at_hit]).sum(dim=-1).clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(dot))
        env._ht_acc_n += at_hit.sum()
        env._ht_acc_success += ok[at_hit].sum()
        env._ht_acc_pe_sum += pe[at_hit].sum()
        env._ht_acc_ve_sum += ve[at_hit].sum()
        env._ht_acc_pe_abs += dp.abs().sum(dim=0)
        env._ht_acc_ve_abs += dv.abs().sum(dim=0)
        env._ht_acc_ne_deg_sum += angle_deg.sum()
        env._ht_acc_ne_dot_sum += dot.sum()
