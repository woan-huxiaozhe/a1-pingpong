"""HitTrack model-derived reference-tracking reward kernels (additive).

Time-gated Gaussian position / velocity tracking of the noisy end-effector reference, with the
gate driven by the privileged ``_ht_tau_true``. The per-term magnitude is carried by the
``RewardTermCfg.weight`` (20/20), matching the repo convention, so ``w_*`` here stay at 1.0.
Used only by the A1-Pingpong-HitTrack task.
"""

from __future__ import annotations

from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

from .tracking import hit_track_terms


def hit_ref_pos(env, racket_body_name: str, *, sigma_t: float, sigma_p: float, w_pos: float = 1.0):
    """Time-gated Gaussian position-tracking reward of the racket center vs the noisy p_ref."""
    center, center_vel, _ = _racket_body_state(env, racket_body_name)
    p_racket = center - env.scene.env_origins
    pos_term, _ = hit_track_terms(
        p_racket, center_vel, env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_tau_true,
        sigma_t=sigma_t, sigma_p=sigma_p, sigma_v=1.0, w_pos=w_pos, w_vel=0.0)
    return pos_term


def hit_ref_vel(env, racket_body_name: str, *, sigma_t: float, sigma_v: float, w_vel: float = 1.0):
    """Time-gated Gaussian (full-vector) velocity-tracking reward of the racket vs the noisy v_ref."""
    center, center_vel, _ = _racket_body_state(env, racket_body_name)
    p_racket = center - env.scene.env_origins
    _, vel_term = hit_track_terms(
        p_racket, center_vel, env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_tau_true,
        sigma_t=sigma_t, sigma_p=1.0, sigma_v=sigma_v, w_pos=0.0, w_vel=w_vel)
    return vel_term
