"""HitTrack model-derived end-effector reference observations (additive).

These read the per-step ``_ht_*`` reference buffers maintained by
``mdp.reference_commands.update_hit_track_state``. They call ``ensure_ht_runtime_buffers`` first
because the ``ObservationManager`` probes each obs term's output shape once at env construction --
before any reset event has populated the buffers. Used only by the A1-Pingpong-HitTrack task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

from .reference_commands import ensure_ht_runtime_buffers as _ensure_ht_runtime_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv  # noqa: F401


def hit_reference_command(env) -> torch.Tensor:
    """Actor reference command ``[p_ref, v_ref, n_ref, tau]`` (noisy, deployable). 10-dim."""
    _ensure_ht_runtime_buffers(env)
    return torch.cat(
        [env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy, env._ht_tau_noisy.unsqueeze(-1)],
        dim=-1,
    )


def hit_reference_command_clean(env) -> torch.Tensor:
    """Critic reference command ``[p_ref, v_ref, n_ref, tau_true]`` (clean/privileged). 10-dim."""
    _ensure_ht_runtime_buffers(env)
    return torch.cat(
        [env._ht_p_ref_clean, env._ht_v_ref_clean, env._ht_n_ref_clean, env._ht_tau_true.unsqueeze(-1)],
        dim=-1,
    )


def hit_ref_pos_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center (env-local) minus noisy p_ref. Deployable (FK) -> actor."""
    _ensure_ht_runtime_buffers(env)
    center, _, _ = _racket_body_state(env, racket_body_name)
    return (center - env.scene.env_origins) - env._ht_p_ref_noisy


def hit_ref_vel_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center velocity minus noisy v_ref. Privileged (sim vel) -> critic only."""
    _ensure_ht_runtime_buffers(env)
    _, center_vel, _ = _racket_body_state(env, racket_body_name)
    return center_vel - env._ht_v_ref_noisy
