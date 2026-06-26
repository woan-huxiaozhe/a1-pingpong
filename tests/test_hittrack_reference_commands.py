from __future__ import annotations

import types

import torch

from _hittrack_loader import load_pure

rc = load_pure("reference_commands")


def _fake_env(n=4, device="cpu"):
    env = types.SimpleNamespace()
    env.num_envs = n
    env.device = device
    env.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
    env.scene = types.SimpleNamespace(env_origins=torch.zeros(n, 3, device=device))
    env.step_dt = 0.01
    return env


PARAMS = dict(
    hit_plane_x=-1.37, target_xyz=(0.685, 0.0, 0.76),
    box={"y": (-0.2, 0.3), "z": (0.9, 1.25), "vx": (-4.5, -3.0), "vy": (-0.3, 0.3), "vz": (-1.0, 0.5)},
    max_prep_s=0.6, post_margin_s=0.12, step_dt=0.01,
    reach_y_range=(-0.6, 0.6), reach_z_range=(0.7, 1.5),
)


def test_reset_fills_constant_clean_and_tau_initial():
    env = _fake_env()
    rc.reset_reference_command(env, None, **PARAMS)
    # clean reference is constant over the episode; p_ref_clean x == hit plane
    assert env._ht_p_ref_clean.shape == (4, 3)
    assert torch.allclose(env._ht_p_ref_clean[:, 0], torch.full((4,), -1.37), atol=1e-5)
    # tau_true at cursor 0 equals tau_initial (<= max_prep)
    assert (env._ht_tau_true <= 0.6 + 1e-6).all()
    assert (env._ht_tau_true > 0.0).all()


def test_cursor_advances_and_tau_counts_down():
    env = _fake_env()
    rc.reset_reference_command(env, None, **PARAMS)
    tau0 = env._ht_tau_true.clone()
    env.episode_length_buf += 1
    rc.update_hit_track_state(env, None, success_pos_thresh=0.05, success_vel_thresh=0.2)
    assert torch.all(env._ht_tau_true < tau0 + 1e-9)


def test_success_recorded_at_hit_when_racket_matches_clean():
    env = _fake_env(n=1)
    rc.reset_reference_command(env, None, **PARAMS)
    # jump cursor to the hit step (tau_true≈0)
    hit_step = int(round((env._ht_tau_true[0].item()) / env.step_dt))
    env.episode_length_buf[0] = hit_step
    # place a fake racket exactly on the clean reference via injected getter
    env._ht_test_racket_pos = env._ht_p_ref_clean.clone()
    env._ht_test_racket_vel = env._ht_v_ref_clean.clone()
    rc.update_hit_track_state(env, None, success_pos_thresh=0.05, success_vel_thresh=0.2,
                              racket_pos=env._ht_test_racket_pos, racket_vel=env._ht_test_racket_vel)
    assert bool(env._ht_success[0])
