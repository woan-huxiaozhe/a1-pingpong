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

from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_planner import plan_hit_reference
from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_source import (
    is_reachable,
    phase_scaled_ball_noise,
    sample_hit_ball_states,
    tau_streams,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _ensure_ht_buffers(env, n_steps: int):
    n, device = env.num_envs, env.device
    if hasattr(env, "_ht_p_ref_clean") and getattr(env, "_ht_n_steps", None) == n_steps:
        return
    env._ht_n_steps = n_steps

    def z3():
        return torch.zeros(n, 3, device=device)

    def zT3():
        return torch.zeros(n, n_steps, 3, device=device)

    env._ht_p_ref_clean, env._ht_v_ref_clean, env._ht_n_ref_clean = z3(), z3(), z3()
    env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy = z3(), z3(), z3()
    env._ht_p_ref_noisy_stream, env._ht_v_ref_noisy_stream, env._ht_n_ref_noisy_stream = zT3(), zT3(), zT3()
    env._ht_tau_true_stream = torch.zeros(n, n_steps, device=device)
    env._ht_tau_noisy_stream = torch.zeros(n, n_steps, device=device)
    env._ht_tau_true = torch.zeros(n, device=device)
    env._ht_tau_noisy = torch.zeros(n, device=device)
    env._ht_valid_len = torch.full((n,), n_steps, dtype=torch.long, device=device)
    env._ht_hit_step = torch.zeros(n, dtype=torch.long, device=device)
    env._ht_hit_done = torch.zeros(n, dtype=torch.bool, device=device)
    env._ht_success = torch.zeros(n, dtype=torch.bool, device=device)
    env._ht_pos_err_at_hit = torch.full((n,), float("nan"), device=device)
    env._ht_vel_err_at_hit = torch.full((n,), float("nan"), device=device)


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

    hit_step = torch.round(tau_true_stream[:, 0] / step_dt).long()
    tau_noisy_stream = tau_true_stream.clone()

    target = torch.tensor(target_xyz, device=device).reshape(1, 3).repeat(k, 1)
    # plan clean (constant) and noisy (per-step) references at reset (planner is memoryless)
    pc, vc, nc = plan_hit_reference(clean[:, :3], clean[:, 3:6], target)
    flat = noisy_stream.reshape(k * n_steps, 6)
    target_flat = target.repeat_interleave(n_steps, dim=0)
    pn, vn, nn = plan_hit_reference(flat[:, :3], flat[:, 3:6], target_flat)

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
    step = env.episode_length_buf.to(torch.long)
    cursor = step.clamp(min=0)
    cursor = torch.minimum(cursor, env._ht_valid_len - 1)
    arange = torch.arange(env.num_envs, device=env.device)
    env._ht_p_ref_noisy = env._ht_p_ref_noisy_stream[arange, cursor]
    env._ht_v_ref_noisy = env._ht_v_ref_noisy_stream[arange, cursor]
    env._ht_n_ref_noisy = env._ht_n_ref_noisy_stream[arange, cursor]
    env._ht_tau_noisy = env._ht_tau_noisy_stream[arange, cursor]
    env._ht_tau_true = env._ht_tau_true_stream[arange, cursor]

    at_hit = (step == env._ht_hit_step) & (~env._ht_hit_done)
    if bool(at_hit.any()):
        # Fetch the racket state lazily -- only needed at the hit step -- so the per-step cursor
        # advance stays Isaac-free (the deferred observations import pulls in isaaclab/USD).
        if racket_pos is None or racket_vel is None:
            from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import RACKET_BODY_NAME
            from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

            center, center_vel, _ = _racket_body_state(env, RACKET_BODY_NAME)
            racket_pos = center - env.scene.env_origins
            racket_vel = center_vel

        pe = torch.norm(racket_pos - env._ht_p_ref_clean, dim=-1)
        ve = torch.norm(racket_vel - env._ht_v_ref_clean, dim=-1)
        ok = (pe < success_pos_thresh) & (ve < success_vel_thresh)
        env._ht_pos_err_at_hit[at_hit] = pe[at_hit]
        env._ht_vel_err_at_hit[at_hit] = ve[at_hit]
        env._ht_success[at_hit] = ok[at_hit]
        env._ht_hit_done[at_hit] = True
