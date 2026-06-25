"""Standalone physics check for the analytic hitting model (no Isaac needed).

Loads ``hitting.py`` directly by file path so the package ``__init__`` (which imports
isaaclab) is bypassed -- this runs with only torch installed. Validates two things:

  1. The drag-aware launch solve: an INDEPENDENT 3D forward integration of the returned
     ``v_out`` lands the ball at the opponent table center.
  2. The contact inversion algebra: feeding (v_in, v_paddle, n) back through the rigid
     frictionless bounce reproduces ``v_out`` exactly.
"""

import importlib.util
import math
import os

import pytest

torch = pytest.importorskip("torch")

_HITTING_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "source",
    "unitree_rl_lab",
    "unitree_rl_lab",
    "tasks",
    "table_tennis_sac",
    "mdp",
    "hitting.py",
)


def _load_hitting():
    spec = importlib.util.spec_from_file_location("tt_hitting", os.path.abspath(_HITTING_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hitting = _load_hitting()

DRAG_K = 0.08
LIN_DAMP = 0.05
GRAVITY = 9.81


def _simulate_landing_3d(origin, v0, target_z, *, control_dt=0.005, substeps=1, max_t=2.5):
    """Independent fine-grained 3D integration (same physics as the solver, finer dt).

    Integrate until z descends back through ``target_z`` after the apex; return the
    interpolated landing (x, y). Replicates gravity + linear damping + quadratic air drag.
    """
    pos = origin.clone()
    vel = v0.clone()
    n = pos.shape[0]
    landed = torch.zeros(n, dtype=torch.bool)
    land_xy = torch.zeros(n, 2)
    sub_dt = control_dt / substeps
    damp = (1.0 / (1.0 + LIN_DAMP * sub_dt)) ** substeps
    passed_apex = torch.zeros(n, dtype=torch.bool)
    steps = int(max_t / control_dt)
    for _ in range(steps):
        prev_pos = pos.clone()
        pos = pos + vel * control_dt
        # detect descending crossing of target_z
        passed_apex = passed_apex | (vel[:, 2] <= 0.0)
        crossing = (~landed) & passed_apex & (prev_pos[:, 2] >= target_z) & (pos[:, 2] < target_z)
        if torch.any(crossing):
            denom = (prev_pos[:, 2] - pos[:, 2]).clamp(min=1.0e-9)
            frac = ((prev_pos[:, 2] - target_z) / denom).clamp(0.0, 1.0)
            xy = prev_pos[:, :2] + frac.unsqueeze(-1) * (pos[:, :2] - prev_pos[:, :2])
            land_xy = torch.where(crossing.unsqueeze(-1), xy, land_xy)
            landed = landed | crossing
        # advance velocity: gravity + linear damping + quadratic drag (per control step)
        vel = vel.clone()
        vel[:, 2] = vel[:, 2] - GRAVITY * control_dt
        vel = vel * damp
        speed = torch.norm(vel, dim=-1, keepdim=True).clamp(min=1.0e-9)
        vel = vel * (1.0 - DRAG_K * speed * control_dt).clamp(min=0.0)
        if bool(landed.all()):
            break
    return land_xy, landed


def test_v_out_lands_at_target():
    origin = torch.tensor(
        [
            [-1.40, 0.00, 1.00],
            [-1.40, 0.15, 0.95],
            [-1.45, -0.20, 1.10],
        ]
    )
    # incoming ball velocities (moving toward -x = robot side), varied
    v_in = torch.tensor(
        [
            [-3.0, 0.0, -0.4],
            [-2.6, -0.2, 0.6],
            [-3.4, 0.3, -0.8],
        ]
    )
    target = torch.tensor([[0.685, 0.0, 0.76]]).repeat(3, 1)

    v_paddle, v_out, n = hitting.ideal_racket_velocity(
        origin, v_in, target, drag_k=DRAG_K, lin_damp=LIN_DAMP, gravity=GRAVITY
    )

    land_xy, landed = _simulate_landing_3d(origin, v_out, target[:, 2])
    assert bool(landed.all()), f"ball never landed: {landed}"
    err_x = (land_xy[:, 0] - target[:, 0]).abs()
    err_y = (land_xy[:, 1] - target[:, 1]).abs()
    assert torch.all(err_x < 0.15), f"x landing error too large: {err_x.tolist()}"
    assert torch.all(err_y < 0.05), f"y landing error too large: {err_y.tolist()}"


def test_contact_inversion_reproduces_v_out():
    v_in = torch.tensor(
        [
            [-3.0, 0.0, -0.4],
            [-2.6, -0.2, 0.6],
            [-3.4, 0.3, -0.8],
        ]
    )
    v_out = torch.tensor(
        [
            [3.0, 0.0, 1.6],
            [2.4, 0.1, 1.4],
            [3.2, -0.2, 1.8],
        ]
    )
    e = hitting.PADDLE_RESTITUTION
    v_paddle, n = hitting.contact_inverse(v_in, v_out, restitution=e)

    # rigid frictionless bounce off a plane moving at v_paddle, normal n
    rel_n = torch.sum((v_in - v_paddle) * n, dim=-1, keepdim=True)
    v_out_recon = v_in - (1.0 + e) * rel_n * n
    assert torch.allclose(v_out_recon, v_out, atol=1.0e-4), (
        f"contact model mismatch:\n recon={v_out_recon}\n want={v_out}"
    )


def test_slow_ball_requires_real_swing():
    """A slow incoming ball (post-drag) cannot be returned to center by passive blocking:
    the required paddle speed must be clearly positive (a real forward swing)."""
    origin = torch.tensor([[-1.40, 0.0, 1.0]])
    v_in = torch.tensor([[-1.8, 0.0, 0.2]])  # slow incoming
    target = torch.tensor([[0.685, 0.0, 0.76]])
    v_paddle, v_out, n = hitting.ideal_racket_velocity(
        origin, v_in, target, drag_k=DRAG_K, lin_damp=LIN_DAMP, gravity=GRAVITY
    )
    assert torch.norm(v_paddle, dim=-1).item() > 0.5, "expected a real swing for a slow ball"
    assert v_out[0, 0].item() > 0.0, "v_out must go forward (+x) toward the opponent"


def test_predict_landing_xy_matches_solver_target():
    """The drag-aware landing predictor (control_dt/substeps matching the sim) must place the
    solver's own ``v_out`` at the launch target -- a round-trip consistency check that also
    guards the reward that consumes it (racket_predicted_landing)."""
    origin = torch.tensor(
        [
            [-1.40, 0.00, 1.00],
            [-1.45, -0.20, 1.10],
            [-1.40, 0.15, 0.95],
        ]
    )
    v_in = torch.tensor(
        [
            [-3.0, 0.0, -0.4],
            [-3.4, 0.3, -0.8],
            [-2.6, -0.2, 0.6],
        ]
    )
    target = torch.tensor([[0.685, 0.0, 0.76]]).repeat(3, 1)
    _, v_out, _ = hitting.ideal_racket_velocity(
        origin, v_in, target, drag_k=DRAG_K, lin_damp=LIN_DAMP, gravity=GRAVITY
    )
    lx, ly, valid = hitting.predict_landing_xy(
        origin, v_out, table_z=0.76, drag_k=DRAG_K, lin_damp=LIN_DAMP,
        control_dt=0.02, substeps=4, gravity=GRAVITY,
    )
    assert bool(valid.all()), f"all launches should land: {valid}"
    err_x = (lx - target[:, 0]).abs()
    err_y = (ly - target[:, 1]).abs()
    assert torch.all(err_x < 0.12), f"x landing error too large: {err_x.tolist()}"
    assert torch.all(err_y < 0.05), f"y landing error too large: {err_y.tolist()}"


def test_predict_landing_xy_invalid_when_never_reaching_table():
    """A ball that never rises to table height (apex below table_z) has no valid landing and
    must report valid=False (the reward gates it to zero rather than scoring (0, 0))."""
    origin = torch.tensor([[-1.0, 0.0, 0.50]])  # below table height
    v_out = torch.tensor([[1.0, 0.0, 0.3]])      # rises only to ~0.5x m, never reaches 0.76
    lx, ly, valid = hitting.predict_landing_xy(
        origin, v_out, table_z=0.76, drag_k=DRAG_K, lin_damp=LIN_DAMP,
        control_dt=0.02, substeps=4, gravity=GRAVITY,
    )
    assert not bool(valid.any()), "a ball below table height that never rises should be invalid"
