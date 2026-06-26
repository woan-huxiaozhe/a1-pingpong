from __future__ import annotations

import os
import tempfile
import types

import numpy as np
import torch

from _hittrack_loader import load_pure

rc = load_pure("reference_commands")


def _write_fixture(path, S=3, T=40):
    np.savez(
        path,
        noisy_ball_stream=np.zeros((S, T, 5), np.float32),
        clean_ball_state=np.tile(np.array([-1.37, 0, 1.0, -4, 0, -0.3], np.float32), (S, 1)),
        tau_true=np.tile(np.linspace(0.4, 0.4 - (T - 1) * 0.01, T, dtype=np.float32), (S, 1)),
        valid_len=np.full((S,), T, np.int64),
        reachable=np.ones((S,), bool),
    )


def test_baked_source_loads_and_resets():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ref.npz")
        _write_fixture(p)
        baked = rc.load_baked_references(p, device="cpu")
        env = types.SimpleNamespace(
            num_envs=2,
            device="cpu",
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            scene=types.SimpleNamespace(env_origins=torch.zeros(2, 3)),
            step_dt=0.01,
        )
        rc.reset_reference_command(
            env, None, hit_plane_x=-1.37, target_xyz=(0.685, 0, 0.76),
            box=None, max_prep_s=0.4, post_margin_s=0.0, step_dt=0.01,
            reach_y_range=(-0.6, 0.6), reach_z_range=(0.7, 1.5), baked=baked)
        assert env._ht_p_ref_clean.shape == (2, 3)
        assert (env._ht_tau_true > 0).all()
