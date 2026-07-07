"""HitTrack model-derived reference-tracking reward kernels (additive).

Time-gated Gaussian position / velocity tracking of the noisy end-effector reference, with the
gate driven by the privileged ``_ht_tau_true``. The per-term magnitude is carried by the
``RewardTermCfg.weight`` (20/20), matching the repo convention, so ``w_*`` here stay at 1.0.
Used only by the A1-Pingpong-HitTrack task.
"""

from __future__ import annotations

from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import (
    _racket_body_state,
    racket_ang_vel,
    racket_normal,
)

from .tracking import face_still_term, hit_track_terms, normal_align_term


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


def hit_ref_normal(
    env,
    racket_body_name: str,
    *,
    sigma_t: float,
    sigma_normal_deg: float,
    w_normal: float = 1.0,
):
    """Time-gated angular-Gaussian blade-normal alignment vs the noisy n_ref.

    The blade normal is otherwise an unconstrained DOF (there is no orientation term besides
    pos/vel), so it drifts to ~100 deg error at the hit instant. ``racket_normal`` is the FK
    world-frame unit normal (deployable); ``n_ref`` is a pure direction, so the env-origin offset
    is irrelevant (no subtraction needed, unlike position).
    """
    n_racket = racket_normal(env, racket_body_name)
    return normal_align_term(
        n_racket,
        env._ht_n_ref_noisy,
        env._ht_tau_true,
        sigma_t=sigma_t,
        sigma_normal_deg=sigma_normal_deg,
        w_normal=w_normal,
    )


def hit_ref_normal_rate(
    env,
    racket_body_name: str,
    *,
    sigma_t: float,
    sigma_rate: float,
    sigma_v: float,
    w: float = 1.0,
):
    """Time-gated reward for a *steady* blade face while swinging through (low ``|dn/dt|``, coupled to speed).

    Complements ``hit_ref_normal`` (which sets WHERE the face points): this penalises the face
    tumbling as it arrives. Measured RL play shows the face turning ~189 deg/s at contact (vs the
    traditional controller's ~58 deg/s) because the final swing speed is sourced from a distal wrist
    snap; that snap is the same wrist DOF that carries the normal, so normal_err spikes exactly at
    peak speed. Rewarding a still face pushes the policy to source speed proximally and hold the
    blade -- the traditional cruise pattern -- so orientation improves without paying swing speed.

    The face-still score is multiplied INSIDE the kernel by the velocity-matching Gaussian (same
    ``sigma_v`` and noisy ``v_ref`` as ``hit_ref_vel``), so "still because the arm stopped" earns
    nothing -- only a still face reached *while swinging at v_ref* is rewarded. See ``face_still_term``.
    """
    center, center_vel, _ = _racket_body_state(env, racket_body_name)
    n = racket_normal(env, racket_body_name)
    w_ang = racket_ang_vel(env, racket_body_name)
    return face_still_term(
        n, w_ang, center_vel, env._ht_v_ref_noisy, env._ht_tau_true,
        sigma_t=sigma_t, sigma_rate=sigma_rate, sigma_v=sigma_v, w=w,
    )
