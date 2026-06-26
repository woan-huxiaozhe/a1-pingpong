from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_BAKE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "source",
    "unitree_rl_lab",
    "unitree_rl_lab",
    "tasks",
    "table_tennis_sac",
    "bake_hittrack_references.py",
)


def _load_bake():
    spec = importlib.util.spec_from_file_location("bake_hittrack_references", _BAKE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bake = pytest.importorskip("numpy") and _load_bake()
true_crossing = _bake.true_crossing
resample_to_grid = _bake.resample_to_grid


def test_true_crossing_linear():
    # straight-line ball crossing x=-1.37 at t=0.5925
    t = np.linspace(0.0, 1.0, 101)
    x = 1.0 - 4.0 * t  # crosses -1.37 at t=(1+1.37)/4=0.5925
    y = np.zeros_like(t)
    z = 1.0 + 0.0 * t
    traj = np.stack([x, y, z, -4 * np.ones_like(t), 0 * t, 0 * t], axis=1)
    state, t_cross = true_crossing(traj, t, hit_plane_x=-1.37)
    assert abs(state[0] - (-1.37)) < 1e-6
    assert abs(t_cross - 0.5925) < 1e-3


def test_resample_linear_midpoint():
    t = np.array([0.0, 0.02])
    v = np.array([[0.0], [2.0]])
    grid = np.array([0.01])
    out = resample_to_grid(t, v, grid)
    assert abs(out[0, 0] - 1.0) < 1e-6
