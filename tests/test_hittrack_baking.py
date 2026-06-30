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
    "a1_pingpong_hittrack",
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
load_recording = _bake.load_recording
kf_downward_crossing = _bake.kf_downward_crossing
bake = _bake.bake


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


# --- recording schema / CSV loader -------------------------------------------------------------

_HEADER = (
    "t,serve_id,mocap_x,mocap_y,mocap_z,mocap_vx,mocap_vy,mocap_vz,"
    "KF_x,KF_y,KF_z,KF_vx,KF_vy,KF_vz,KF_pred_y,KF_pred_z,KF_pred_vx,KF_pred_vy,KF_pred_vz,tau,valid"
)


def test_load_recording_csv_header_zoffset_valid(tmp_path):
    p = tmp_path / "kalman_trajectory_1.csv"
    # mocap_z / KF_z at col 4 / 10; KF_pred_z at col 15. All heights must get the z_offset.
    p.write_text(
        _HEADER
        + "\n"
        + "0.000,1,0.90,0.0,0.40,-4,0,0,0.90,0.0,0.40,0,0,0,0.0,0.00,0,0,0,0.0,False\n"
        + "0.003,1,0.88,0.0,0.41,-4,0,0,0.88,0.0,0.41,-4,0,0,0.0,0.42,-4,0,0,0.0,True\n"
    )
    rec = load_recording(str(p), z_offset=0.7)
    assert rec["t"].shape == (2,) and rec["serve_id"].tolist() == [1, 1]
    assert rec["valid"].tolist() == [False, True]  # header skipped, bool parsed
    assert abs(rec["mocap"][1, 2] - (0.41 + 0.7)) < 1e-6  # mocap z shifted
    assert abs(rec["kf"][1, 2] - (0.41 + 0.7)) < 1e-6  # kf z shifted
    assert abs(rec["kf_pred"][1, 1] - (0.42 + 0.7)) < 1e-6  # kf_pred z (index 1) shifted too
    assert rec["kf_pred"].shape == (2, 5)


# --- KF-based downward crossing ----------------------------------------------------------------

def test_kf_downward_crossing_ignores_upward_and_respects_start():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, -2.0, -2.0, 0.0])  # crosses -1.37 DOWN at idx0, back UP between idx2..3
    traj = np.stack([x, np.zeros(4), np.ones(4), -2 * np.ones(4), np.zeros(4), np.zeros(4)], axis=1)
    out = kf_downward_crossing(traj, t, hit_plane_x=-1.37, start=0)
    assert out is not None
    state, t_cross, idx = out
    assert idx == 0 and abs(state[0] - (-1.37)) < 1e-6 and 0.0 < t_cross < 1.0
    # starting after the downward crossing leaves only the upward one -> None
    assert kf_downward_crossing(traj, t, hit_plane_x=-1.37, start=2) is None


# --- bake() gates ------------------------------------------------------------------------------

def _serve(sid, x0, vx, *, n=200, dt=0.005, y=0.1, z=1.0):
    """A straight incoming KF serve from ball-x ``x0`` at ``vx`` m/s; all samples valid."""
    t = dt * np.arange(n)
    x = x0 + vx * t
    kf = np.stack([x, np.full(n, y), np.full(n, z), np.full(n, vx), np.zeros(n), np.zeros(n)], axis=1)
    kf_pred = np.stack([np.full(n, y), np.full(n, z), np.full(n, vx), np.zeros(n), np.zeros(n)], axis=1)
    return {
        "t": t,
        "serve_id": np.full(n, sid, dtype=np.int64),
        "kf": kf,
        "kf_pred": kf_pred,
        "valid": np.ones(n, dtype=bool),
    }


def _bake_kw():
    return dict(hit_plane_x=-1.37, step_dt=0.01, reach_y_range=(-0.6, 0.6), reach_z_range=(0.7, 1.5))


def test_bake_clean_velocity_comes_from_kf():
    out = bake([_serve(1, 1.2, -4.0)], **_bake_kw())
    assert out["clean_ball_state"].shape[0] == 1
    clean = out["clean_ball_state"][0]
    assert abs(clean[0] - (-1.37)) < 1e-4 and abs(clean[1] - 0.1) < 1e-4
    np.testing.assert_allclose(clean[3:6], [-4.0, 0.0, 0.0], atol=1e-4)  # KF velocity, not mocap


def test_bake_velocity_gate_drops_spike():
    # serve #2 has a 400 m/s crossing (mocap-dropout-style spike) -> dropped; #1 kept.
    out = bake([_serve(1, 1.2, -4.0), _serve(2, 1.2, -400.0)], **_bake_kw())
    assert out["clean_ball_state"].shape[0] == 1
    np.testing.assert_allclose(out["clean_ball_state"][0, 3:6], [-4.0, 0.0, 0.0], atol=1e-4)


def test_bake_start_gate_drops_late_lockon():
    # serve #2 locks on at x0=0.5 (<= 0.6 gate) -> dropped; #1 (x0=1.2) kept.
    out = bake([_serve(1, 1.2, -4.0), _serve(2, 0.5, -4.0)], start_x_gate=0.6, **_bake_kw())
    assert out["clean_ball_state"].shape[0] == 1


def test_bake_reachable_flag_on_z():
    # z=2.0 is outside REACH_Z (0.7,1.5) -> reachable False but serve still baked.
    out = bake([_serve(1, 1.2, -4.0, z=2.0)], **_bake_kw())
    assert out["clean_ball_state"].shape[0] == 1 and not bool(out["reachable"][0])


# --- lateral (y) augmentation (pure-torch kernel; skipped if torch absent) ----------------------

def test_sample_lateral_shift_lands_in_band():
    torch = pytest.importorskip("torch")
    import importlib.util

    src = os.path.join(os.path.dirname(_BAKE_PATH), "mdp", "reference_source.py")
    spec = importlib.util.spec_from_file_location("reference_source", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    y_cross = torch.tensor([-0.29, 0.0, 0.29, 0.1])
    dy = mod.sample_lateral_shift(y_cross, (-0.6, 0.6), margin=0.05)
    shifted = y_cross + dy
    assert bool((shifted >= -0.55 - 1e-5).all()) and bool((shifted <= 0.55 + 1e-5).all())
