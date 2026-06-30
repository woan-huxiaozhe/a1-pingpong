# A1-TableTennis-SAC-HitTrack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **NOTE (2026-06-30 refactor):** HitTrack has since been extracted into its own task package `a1_pingpong_hittrack` (gym id `A1-Pingpong-HitTrack`, trained with RSL-RL PPO). The paths and task id below reflect the ORIGINAL `table_tennis_sac`-additive layout this plan was executed under. `hitting.py` and `_racket_body_state` still live in `table_tennis_sac` and are imported by the new package. See `docs/hittrack_方案简介.md` for the current layout.

**Goal:** Add a new, non-end-to-end RL task `A1-TableTennis-SAC-HitTrack` where the arm tracks a model-derived end-effector hit reference (`p_ref, v_ref, n_ref`) at the predicted ball hit time, with no ball in the MDP.

**Architecture:** Purely additive to the shared `table_tennis_sac` package. A pure-torch planner (`reference_planner.py`, wraps existing `hitting.py`) converts a hit-plane ball state into an end-effector reference at runtime. A reference-command manager (`reference_commands.py`) samples a serve per episode (synthetic source first; baked-real source later), runs the planner at reset to fill per-step `noisy`(actor)/`clean`(critic) reference streams + `tau_true`, and advances a cursor each control step. Reward = Gaussian time-gated position + velocity tracking error. A new env cfg (`hittrack_env_cfg.py`) at 100 Hz wires the new terms; Catch is untouched.

**Tech Stack:** Python, PyTorch, Isaac Lab `ManagerBasedRLEnv`, NumPy (baking), pytest.

## Global Constraints

- **Never modify** `table_tennis_sac/env_cfg.py` or anything Catch (`A1-TableTennis-SAC-Catch`) depends on. New task = new files + additive functions only. **No deletions** of existing shared functions.
- **Control rate 100 Hz**: HitTrack env cfg uses `decimation=2`, `sim.dt=0.005` (physics 200 Hz), `step_dt=0.01`. Catch stays 50 Hz.
- **Hit plane**: `HIT_PLANE_X = -1.37` (= `SAC_ROBOT_X`), `ROBOT_SIDE = -1`.
- **Frame convention**: all stored references and the racket center used in rewards are **env-local** (world minus `env.scene.env_origins`). `p_racket` for reward = `_racket_body_state(...)` center `- env.scene.env_origins`. Guard against the historical frame-flip bug — references and racket pos MUST be in the same env-local frame.
- **Isaac-free tests**: `hitting.py`, `tracking.py`, `reference_planner.py`, and the baking script import no `isaaclab`; test them by direct import. For env-facing functions (observations/rewards/events) that transitively import `isaaclab`, test the **pure kernel** (extracted into Isaac-free modules) and/or mirror against a `types.SimpleNamespace` fake env, per the existing pattern in `tests/test_sac_table_tennis_pipeline.py`.
- **Tracking reward numbers (v1, 100 Hz)**: `sigma_t=0.03 s`, `sigma_p=0.03 m`, `sigma_v=0.3 m/s`, `w_pos=20`, `w_vel=20`, `w_normal=0` (normal off). `sigma_n=12°` reserved for when normal is enabled.
- **Success thresholds (vs clean)**: `pos < 0.05 m` AND `vel < 0.2 m/s`.
- **Episode**: `max_prep ≈ 0.6 s`, `post_margin ≈ 0.12 s`, `episode_length_s ≈ 0.72 s` (~72 steps @100 Hz).
- **Reference planner reuse**: `mdp.hitting.ideal_racket_velocity` (`NEUTRAL_THETA=radians(28°)`, `PADDLE_RESTITUTION=0.75`, `drag_k=0.08`, `lin_damp=0.05`). Target = `(OPP_TABLE_CENTER_X, 0.0, TABLE_Z)`, with `OPP_TABLE_CENTER_X≈+0.685`, `TABLE_Z=0.76`.
- **Run tests with**: `cd /data/PPO-pingpong && python -m pytest tests/<file> -v`.
- **Commit policy**: This repo requires showing the commit message draft (per `~/.gitmessage`) and getting explicit user confirmation before each `git commit`. The "Commit" steps below mean "prepare draft, confirm, then commit". Messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Pure tracking-reward kernels

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/tracking.py`
- Test: `tests/test_hittrack_tracking.py`

**Interfaces:**
- Produces: `time_gate(tau: Tensor, sigma_t: float) -> Tensor`; `gaussian_score(error_norm: Tensor, sigma: float) -> Tensor`; `hit_track_terms(p_racket, v_racket, p_ref, v_ref, tau, *, sigma_t, sigma_p, sigma_v, w_pos, w_vel) -> tuple[Tensor pos_term, Tensor vel_term]` (each `[N]`, additive, time-gated, full-vector velocity).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hittrack_tracking.py
from __future__ import annotations
import math
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp.tracking import (
    time_gate, gaussian_score, hit_track_terms,
)

def test_time_gate_peaks_at_zero():
    tau = torch.tensor([0.0, 0.03, -0.03])
    g = time_gate(tau, sigma_t=0.03)
    assert torch.isclose(g[0], torch.tensor(1.0))
    assert torch.isclose(g[1], torch.tensor(math.exp(-0.5)), atol=1e-6)
    assert torch.isclose(g[1], g[2])  # symmetric

def test_gaussian_score_perfect_and_decay():
    assert torch.isclose(gaussian_score(torch.tensor(0.0), 0.03), torch.tensor(1.0))
    assert gaussian_score(torch.tensor(0.06), 0.03) < gaussian_score(torch.tensor(0.03), 0.03)

def test_hit_track_terms_perfect_hit():
    p = torch.zeros(1, 3); v = torch.zeros(1, 3)
    pos_term, vel_term = hit_track_terms(
        p, v, p_ref=p.clone(), v_ref=v.clone(), tau=torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=20.0, w_vel=20.0)
    assert torch.isclose(pos_term[0], torch.tensor(20.0))
    assert torch.isclose(vel_term[0], torch.tensor(20.0))

def test_hit_track_terms_velocity_is_full_vector():
    # tangential velocity error must be penalized (full-vector, D3=a)
    p = torch.zeros(1, 3)
    v = torch.tensor([[0.0, 1.0, 0.0]])
    v_ref = torch.zeros(1, 3)
    _, vel_term = hit_track_terms(p, v, p, v_ref, torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=20.0, w_vel=20.0)
    assert vel_term[0] < 20.0  # 1 m/s tangential error reduces the score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mdp.tracking`.

- [ ] **Step 3: Write minimal implementation**

```python
# mdp/tracking.py
"""Pure-torch hit-time reference-tracking reward kernels (no isaaclab import, unit-testable)."""
from __future__ import annotations
import torch

def time_gate(tau: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """Gaussian gate centered at the hit instant tau=0 (tau, sigma_t in seconds)."""
    return torch.exp(-0.5 * (tau / sigma_t) ** 2)

def gaussian_score(error_norm: torch.Tensor, sigma: float) -> torch.Tensor:
    """exp(-||e||^2 / (2 sigma^2)) given the L2 norm ||e||."""
    return torch.exp(-(error_norm ** 2) / (2.0 * sigma * sigma))

def hit_track_terms(
    p_racket: torch.Tensor, v_racket: torch.Tensor,
    p_ref: torch.Tensor, v_ref: torch.Tensor, tau: torch.Tensor,
    *, sigma_t: float, sigma_p: float, sigma_v: float, w_pos: float, w_vel: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive, time-gated position + (full-vector) velocity tracking terms. Each [N]."""
    gate = time_gate(tau, sigma_t)
    pos_err = torch.norm(p_racket - p_ref, dim=-1)
    vel_err = torch.norm(v_racket - v_ref, dim=-1)
    pos_term = w_pos * gate * gaussian_score(pos_err, sigma_p)
    vel_term = w_vel * gate * gaussian_score(vel_err, sigma_v)
    return pos_term, vel_term
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_tracking.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (draft → confirm → commit) `mdp/tracking.py` + `tests/test_hittrack_tracking.py`.

---

### Task 2: Reference planner (ball hit-state → end-effector reference)

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/reference_planner.py`
- Test: `tests/test_hittrack_reference_planner.py`

**Interfaces:**
- Consumes: `mdp.hitting.ideal_racket_velocity`, `NEUTRAL_THETA`, `PADDLE_RESTITUTION`.
- Produces: `plan_hit_reference(p_ball_hit: Tensor[N,3], v_ball_hit: Tensor[N,3], target: Tensor[N,3], *, theta=NEUTRAL_THETA, restitution=PADDLE_RESTITUTION, drag_k=0.08, lin_damp=0.05) -> tuple[Tensor p_ref[N,3], Tensor v_ref[N,3], Tensor n_ref[N,3]]`. `p_ref == p_ball_hit` (contact point); `v_ref` is the ideal paddle velocity; `n_ref` the blade normal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hittrack_reference_planner.py
from __future__ import annotations
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_planner import plan_hit_reference

def _case():
    # ball crosses hit plane x=-1.37 moving in +... wait: ROBOT_SIDE=-1, ball travels -x toward robot
    p = torch.tensor([[-1.37, 0.0, 1.0]])
    v_in = torch.tensor([[-4.0, 0.0, -0.5]])      # incoming, toward -x
    target = torch.tensor([[0.685, 0.0, 0.76]])    # opponent table center
    return p, v_in, target

def test_shapes_and_p_ref_identity():
    p, v_in, target = _case()
    p_ref, v_ref, n_ref = plan_hit_reference(p, v_in, target)
    assert p_ref.shape == (1, 3) and v_ref.shape == (1, 3) and n_ref.shape == (1, 3)
    assert torch.allclose(p_ref, p)

def test_v_ref_parallel_to_normal():
    # contact_inverse returns a normal-only paddle velocity → v_ref ∥ n_ref
    p, v_in, target = _case()
    _, v_ref, n_ref = plan_hit_reference(p, v_in, target)
    cross = torch.cross(v_ref, n_ref, dim=-1)
    assert torch.norm(cross) < 1e-4

def test_returns_ball_toward_opponent():
    # the planned outgoing direction must push the ball back toward +x (opponent)
    p, v_in, target = _case()
    _, v_ref, _ = plan_hit_reference(p, v_in, target)
    assert v_ref[0, 0] > 0.0  # paddle pushes in +x
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mdp.reference_planner`.

- [ ] **Step 3: Write minimal implementation**

```python
# mdp/reference_planner.py
"""Runtime hit-reference planner (shared with deployment). Pure torch, no isaaclab import.

Converts a hit-plane ball state into the end-effector reference the arm policy tracks:
target landing -> drag-aware launch solve + closed-form contact inversion (mdp.hitting).
"""
from __future__ import annotations
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp.hitting import (
    ideal_racket_velocity, NEUTRAL_THETA, PADDLE_RESTITUTION,
)

def plan_hit_reference(
    p_ball_hit: torch.Tensor, v_ball_hit: torch.Tensor, target: torch.Tensor,
    *, theta: float = NEUTRAL_THETA, restitution: float = PADDLE_RESTITUTION,
    drag_k: float = 0.08, lin_damp: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (p_ref, v_ref, n_ref). p_ref is the contact point; v_ref the ideal paddle
    velocity; n_ref the blade normal. All world/env-frame; origin offset cancels."""
    v_paddle, _v_out, n = ideal_racket_velocity(
        p_ball_hit, v_ball_hit, target,
        theta=theta, restitution=restitution, drag_k=drag_k, lin_damp=lin_damp,
    )
    return p_ball_hit, v_paddle, n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_planner.py -v`
Expected: PASS (3 tests). If `test_returns_ball_toward_opponent` fails, inspect the sign of `v_ref[0,0]` and confirm the incoming-velocity sign convention; do NOT flip frames — fix the test's expected sign to match the verified `hitting.py` physics.

- [ ] **Step 5: Commit** (draft → confirm) `mdp/reference_planner.py` + test.

---

### Task 3: Synthetic reference-source kernels + reachability

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/reference_source.py`
- Test: `tests/test_hittrack_reference_source.py`

**Interfaces:**
- Produces (all pure torch, Isaac-free):
  - `sample_hit_ball_states(n, box, hit_plane_x, *, device, gen=None) -> Tensor[n,6]` — `[x,y,z,vx,vy,vz]`, `x=hit_plane_x`, other dims uniform in `box` (dict with keys `y,z,vx,vy,vz`, each a `(lo,hi)` tuple).
  - `tau_streams(tau_initial: Tensor[n], n_steps: int, step_dt: float) -> Tensor[n, n_steps]` — `tau_true[k, s] = tau_initial[k] - s*step_dt`.
  - `phase_scaled_ball_noise(tau: Tensor[n,T], bias_unit: Tensor[n,5], *, bias_std, jitter_std, fixed_offset, far_tau) -> Tensor[n,T,5]` — per-episode-biased noise on `(y,z,vx,vy,vz)`, magnitude scaled by `clamp(tau/far_tau,0,1)`. Mirrors the calibrated KF model; **defaults to all-zero stds (noise off)**.
  - `is_reachable(p_hit: Tensor[n,3], y_range, z_range) -> Tensor[n] bool` — loose workspace box gate on the hit-plane `(y,z)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hittrack_reference_source.py
from __future__ import annotations
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_source import (
    sample_hit_ball_states, tau_streams, phase_scaled_ball_noise, is_reachable,
)

BOX = {"y": (-0.2, 0.3), "z": (0.9, 1.25), "vx": (-4.5, -3.0), "vy": (-0.3, 0.3), "vz": (-1.0, 0.5)}

def test_sample_fixes_x_and_respects_box():
    s = sample_hit_ball_states(64, BOX, hit_plane_x=-1.37, device="cpu", gen=torch.Generator().manual_seed(0))
    assert s.shape == (64, 6)
    assert torch.allclose(s[:, 0], torch.full((64,), -1.37))
    assert (s[:, 1] >= -0.2).all() and (s[:, 1] <= 0.3).all()
    assert (s[:, 3] <= -3.0).all() and (s[:, 3] >= -4.5).all()

def test_tau_streams_counts_down():
    tau0 = torch.tensor([0.5, 0.3])
    ts = tau_streams(tau0, n_steps=5, step_dt=0.01)
    assert ts.shape == (2, 5)
    assert torch.isclose(ts[0, 0], torch.tensor(0.5))
    assert torch.isclose(ts[0, 1], torch.tensor(0.49))

def test_noise_off_by_default_is_zero():
    tau = torch.full((3, 4), 0.4)
    bias = torch.randn(3, 5)
    noise = phase_scaled_ball_noise(tau, bias, bias_std=(0,0,0,0,0), jitter_std=(0,0,0,0,0),
                                    fixed_offset=(0,0,0,0,0), far_tau=(0.6,0.4,0.5,0.6,0.6))
    assert torch.allclose(noise, torch.zeros_like(noise))

def test_reachable_box():
    p = torch.tensor([[-1.37, 0.0, 1.0], [-1.37, 2.0, 1.0]])  # 2nd is far out in y
    ok = is_reachable(p, y_range=(-0.6, 0.6), z_range=(0.7, 1.5))
    assert bool(ok[0]) and not bool(ok[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_source.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mdp.reference_source`.

- [ ] **Step 3: Write minimal implementation**

```python
# mdp/reference_source.py
"""Pure-torch synthetic reference-source kernels (Isaac-free): sample hit-plane ball states,
build tau countdown streams, optional KF-style phase-scaled noise, loose reachability gate."""
from __future__ import annotations
import torch

def sample_hit_ball_states(n: int, box: dict, hit_plane_x: float, *, device, gen=None) -> torch.Tensor:
    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)
    y, z = u(*box["y"]), u(*box["z"])
    vx, vy, vz = u(*box["vx"]), u(*box["vy"]), u(*box["vz"])
    x = torch.full((n,), hit_plane_x, device=device)
    return torch.stack([x, y, z, vx, vy, vz], dim=-1)

def tau_streams(tau_initial: torch.Tensor, n_steps: int, step_dt: float) -> torch.Tensor:
    s = torch.arange(n_steps, device=tau_initial.device, dtype=tau_initial.dtype)
    return tau_initial.unsqueeze(1) - s.unsqueeze(0) * step_dt

def phase_scaled_ball_noise(tau, bias_unit, *, bias_std, jitter_std, fixed_offset, far_tau) -> torch.Tensor:
    device, dtype = tau.device, tau.dtype
    far = torch.tensor(far_tau, device=device, dtype=dtype).clamp(min=1e-6)
    phase = (tau.unsqueeze(-1) / far).clamp(0.0, 1.0)                       # [n,T,5]
    bs = torch.tensor(bias_std, device=device, dtype=dtype)
    js = torch.tensor(jitter_std, device=device, dtype=dtype)
    off = torch.tensor(fixed_offset, device=device, dtype=dtype)
    bias = bias_unit.unsqueeze(1) * (bs * phase)                            # [n,T,5]
    jitter = torch.randn(tau.shape[0], tau.shape[1], 5, device=device) * js
    return bias + jitter + off * phase

def is_reachable(p_hit: torch.Tensor, y_range, z_range) -> torch.Tensor:
    y, z = p_hit[:, 1], p_hit[:, 2]
    return (y >= y_range[0]) & (y <= y_range[1]) & (z >= z_range[0]) & (z <= z_range[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_source.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (draft → confirm) `mdp/reference_source.py` + test.

---

### Task 4: Reference-command manager — buffers, reset, per-step update (synthetic source)

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/reference_commands.py`
- Test: `tests/test_hittrack_reference_commands.py`

**Interfaces:**
- Consumes: `reference_planner.plan_hit_reference`; `reference_source.*`.
- Produces (env-facing, store buffers on `env` as `_ht_*`):
  - `reset_reference_command(env, env_ids, *, hit_plane_x, target_xyz, box, max_prep_s, post_margin_s, step_dt, reach_y_range, reach_z_range, noise_params=None)` — EventTerm `mode="reset"`.
  - `update_hit_track_state(env, env_ids, *, success_pos_thresh, success_vel_thresh)` — EventTerm `mode="interval"`.
  - Buffers read by observations/rewards/terminations: `_ht_p_ref_noisy[N,3]`, `_ht_v_ref_noisy[N,3]`, `_ht_n_ref_noisy[N,3]`, `_ht_tau_noisy[N]`, `_ht_p_ref_clean[N,3]`, `_ht_v_ref_clean[N,3]`, `_ht_n_ref_clean[N,3]`, `_ht_tau_true[N]`, `_ht_valid_len[N]`, `_ht_hit_done[N] bool`, `_ht_success[N] bool`, plus final-capture mirrors `_ht_final_*`.
- The planner runs once **at reset** over each env's full noisy ball stream `[k,T,5]` and clean ball state `[k,5]` (memoryless ⇒ identical to per-step). Streams stored as `_ht_p_ref_noisy_stream[N,T,3]` etc.; `update_hit_track_state` copies the cursor row into the current buffers.

- [ ] **Step 1: Write the failing test** (fake-env namespace, mirrors the env-buffer contract)

```python
# tests/test_hittrack_reference_commands.py
from __future__ import annotations
import types
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp import reference_commands as rc

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_commands.py -v`
Expected: FAIL (`ModuleNotFoundError` / attribute errors).

- [ ] **Step 3: Write minimal implementation**

Key design notes for the implementer:
- `update_hit_track_state` accepts optional `racket_pos`/`racket_vel` args (for Isaac-free testing); when `None`, it reads them from the sim via `observations._racket_body_state(env, RACKET_BODY_NAME)` and subtracts `env.scene.env_origins` (env-local frame). Import `RACKET_BODY_NAME` from the forehand env cfg.
- Cursor = `episode_length_buf.clamp(0, valid_len-1)`.
- `n_steps = round(max_prep_s/step_dt) + round(post_margin_s/step_dt) + 1`.
- Hit step per env = `round(tau_initial/step_dt)`; success is evaluated only on the step where `episode_length_buf == hit_step` (latch `_ht_hit_done`).

```python
# mdp/reference_commands.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_planner import plan_hit_reference
from unitree_rl_lab.tasks.table_tennis_sac.mdp.reference_source import (
    sample_hit_ball_states, tau_streams, phase_scaled_ball_noise, is_reachable,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def _ensure_ht_buffers(env, n_steps: int):
    n, device = env.num_envs, env.device
    if hasattr(env, "_ht_p_ref_clean") and env._ht_n_steps == n_steps:
        return
    env._ht_n_steps = n_steps
    z3 = lambda: torch.zeros(n, 3, device=device)
    zT3 = lambda: torch.zeros(n, n_steps, 3, device=device)
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

def reset_reference_command(env, env_ids, *, hit_plane_x, target_xyz, box, max_prep_s,
                            post_margin_s, step_dt, reach_y_range, reach_z_range, noise_params=None):
    n_steps = int(round(max_prep_s / step_dt)) + int(round(post_margin_s / step_dt)) + 1
    _ensure_ht_buffers(env, n_steps)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    k = ids.shape[0]
    device = env.device

    # sample reachable clean hit-plane ball states (resample the few unreachable ones)
    clean = sample_hit_ball_states(k, box, hit_plane_x, device=device)
    bad = ~is_reachable(clean[:, :3], reach_y_range, reach_z_range)
    for _ in range(8):
        if not bool(bad.any()):
            break
        clean[bad] = sample_hit_ball_states(int(bad.sum()), box, hit_plane_x, device=device)
        bad = ~is_reachable(clean[:, :3], reach_y_range, reach_z_range)

    tau_initial = torch.full((k,), float(max_prep_s), device=device)  # E1: capped synthetic horizon
    hit_step = torch.round(tau_initial / step_dt).long()
    tau_true_stream = tau_streams(tau_initial, n_steps, step_dt)       # [k,T]

    # noisy ball stream = clean repeated + optional phase-scaled noise on (y,z,vx,vy,vz)
    noisy_stream = clean.unsqueeze(1).repeat(1, n_steps, 1)            # [k,T,6]
    tau_noisy_stream = tau_true_stream.clone()
    if noise_params is not None:
        bias_unit = torch.nn.functional.normalize(torch.randn(k, 5, device=device), dim=-1)
        noise5 = phase_scaled_ball_noise(tau_true_stream.clamp(min=0.0), bias_unit, **noise_params)  # [k,T,5]
        noisy_stream[..., 1:6] += noise5

    target = torch.tensor(target_xyz, device=device).reshape(1, 3).repeat(k, 1)
    # plan clean (constant) and noisy (per-step) references at reset
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
    env._ht_valid_len[ids] = n_steps
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

def update_hit_track_state(env, env_ids, *, success_pos_thresh, success_vel_thresh,
                           racket_pos=None, racket_vel=None):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    step = env.episode_length_buf.to(torch.long)
    cursor = step.clamp(min=0)
    cursor = torch.minimum(cursor, env._ht_valid_len - 1)
    arange = torch.arange(env.num_envs, device=env.device)
    env._ht_p_ref_noisy = env._ht_p_ref_noisy_stream[arange, cursor]
    env._ht_v_ref_noisy = env._ht_v_ref_noisy_stream[arange, cursor]
    env._ht_n_ref_noisy = env._ht_n_ref_noisy_stream[arange, cursor]
    env._ht_tau_noisy = env._ht_tau_noisy_stream[arange, cursor]
    env._ht_tau_true = env._ht_tau_true_stream[arange, cursor]

    if racket_pos is None or racket_vel is None:
        from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state
        from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import RACKET_BODY_NAME
        center, center_vel, _ = _racket_body_state(env, RACKET_BODY_NAME)
        racket_pos = center - env.scene.env_origins
        racket_vel = center_vel

    at_hit = (step == env._ht_hit_step) & (~env._ht_hit_done)
    if bool(at_hit.any()):
        pe = torch.norm(racket_pos - env._ht_p_ref_clean, dim=-1)
        ve = torch.norm(racket_vel - env._ht_v_ref_clean, dim=-1)
        ok = (pe < success_pos_thresh) & (ve < success_vel_thresh)
        env._ht_pos_err_at_hit[at_hit] = pe[at_hit]
        env._ht_vel_err_at_hit[at_hit] = ve[at_hit]
        env._ht_success[at_hit] = ok[at_hit]
        env._ht_hit_done[at_hit] = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_reference_commands.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (draft → confirm) `mdp/reference_commands.py` + test.

---

### Task 5: Observations (additive — reference command + FK error; NO deletions)

**Files:**
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/observations.py` (append new funcs)
- Test: `tests/test_hittrack_observations.py`

**Interfaces:**
- Produces: `hit_reference_command(env) -> Tensor[N,10]` = `[p_ref_noisy(3), v_ref_noisy(3), n_ref_noisy(3), tau_noisy(1)]`; `hit_reference_command_clean(env) -> Tensor[N,10]` (clean variant + tau_true); `hit_ref_pos_error(env) -> Tensor[N,3]` = `racket_center_local - p_ref_noisy` (deployable, actor); `hit_ref_vel_error(env) -> Tensor[N,3]` = `racket_center_vel - v_ref_noisy` (privileged, critic-only).

- [ ] **Step 1: Write the failing test** (fake env carrying `_ht_*` buffers + a stub racket)

```python
# tests/test_hittrack_observations.py
from __future__ import annotations
import types
import torch
import pytest

obs = pytest.importorskip(
    "unitree_rl_lab.tasks.table_tennis_sac.mdp.observations",
    reason="isaaclab not installed")

def _fake_env(n=2):
    env = types.SimpleNamespace()
    env.num_envs = n
    env.device = "cpu"
    env._ht_p_ref_noisy = torch.zeros(n, 3); env._ht_v_ref_noisy = torch.ones(n, 3)
    env._ht_n_ref_noisy = torch.zeros(n, 3); env._ht_n_ref_noisy[:, 1] = 1.0
    env._ht_tau_noisy = torch.full((n,), 0.4)
    return env

def test_reference_command_is_10_dim_and_ordered():
    env = _fake_env()
    cmd = obs.hit_reference_command(env)
    assert cmd.shape == (2, 10)
    assert torch.allclose(cmd[:, 6:9], env._ht_n_ref_noisy)   # normal block
    assert torch.allclose(cmd[:, 9], env._ht_tau_noisy)        # tau last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_observations.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'hit_reference_command'` (or skip if isaaclab absent — then validate via the env smoke test in Task 8).

- [ ] **Step 3: Write minimal implementation** (append to `observations.py`)

```python
def hit_reference_command(env) -> torch.Tensor:
    """Actor reference command [p_ref, v_ref, n_ref, tau] (noisy, deployable). 10-dim."""
    return torch.cat([env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy,
                      env._ht_tau_noisy.unsqueeze(-1)], dim=-1)

def hit_reference_command_clean(env) -> torch.Tensor:
    """Critic reference command [p_ref, v_ref, n_ref, tau_true] (clean/privileged). 10-dim."""
    return torch.cat([env._ht_p_ref_clean, env._ht_v_ref_clean, env._ht_n_ref_clean,
                      env._ht_tau_true.unsqueeze(-1)], dim=-1)

def hit_ref_pos_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center (env-local) minus noisy p_ref. Deployable (FK) -> actor."""
    center, _, _ = _racket_body_state(env, racket_body_name)
    return (center - env.scene.env_origins) - env._ht_p_ref_noisy

def hit_ref_vel_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center velocity minus noisy v_ref. Privileged (sim vel) -> critic only."""
    _, center_vel, _ = _racket_body_state(env, racket_body_name)
    return center_vel - env._ht_v_ref_noisy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_observations.py -v`
Expected: PASS (or skipped if isaaclab absent; covered later by Task 8 smoke test).

- [ ] **Step 5: Commit** (draft → confirm) observation additions + test.

---

### Task 6: Rewards (additive — hit_ref_pos, hit_ref_vel; NO deletions)

**Files:**
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/rewards.py` (append; import `tracking`)
- Test: `tests/test_hittrack_rewards.py`

**Interfaces:**
- Consumes: `mdp.tracking.hit_track_terms`; reads `_ht_*` buffers + racket center.
- Produces: `hit_ref_pos(env, racket_body_name, *, sigma_t, sigma_p, w_pos=1.0) -> Tensor[N]`; `hit_ref_vel(env, racket_body_name, *, sigma_t, sigma_v, w_vel=1.0) -> Tensor[N]`. (Per-term `w_*` left at 1.0 here; the `RewardTermCfg.weight` carries the 20/20 magnitude, matching the repo convention.)

- [ ] **Step 1: Write the failing test** (Isaac-free, mirror the kernel like the existing test file)

```python
# tests/test_hittrack_rewards.py
from __future__ import annotations
import torch
from unitree_rl_lab.tasks.table_tennis_sac.mdp.tracking import hit_track_terms

def test_perfect_hit_gives_full_terms_at_tau_zero():
    p = torch.zeros(1, 3); v = torch.zeros(1, 3)
    pos, vel = hit_track_terms(p, v, p, v, torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=1.0, w_vel=1.0)
    assert torch.isclose(pos[0], torch.tensor(1.0)) and torch.isclose(vel[0], torch.tensor(1.0))

def test_far_from_hit_time_gates_to_zero():
    p = torch.zeros(1, 3); v = torch.zeros(1, 3)
    pos, vel = hit_track_terms(p, v, p, v, torch.full((1,), 0.3),  # tau=0.3s >> sigma_t
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=1.0, w_vel=1.0)
    assert pos[0] < 1e-6 and vel[0] < 1e-6
```

- [ ] **Step 2: Run test to verify it fails** — Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_rewards.py -v`. Expected: PASS only after Task 1 (kernel) exists; if Task 1 done it already passes — so first make the assertions reference the NOT-yet-added reward wrappers via a fake env to force red. Simpler: keep this test on the kernel (already covered) and instead add the wrapper smoke check in Task 8. **If executing strictly TDD per-file:** add a `tests/test_hittrack_rewards.py::test_reward_wrappers_exist` that imports `mdp.rewards` and asserts `hasattr(rewards, "hit_ref_pos")` (skipif isaaclab missing) — fails before Step 3.

- [ ] **Step 3: Write minimal implementation** (append to `rewards.py`)

```python
from unitree_rl_lab.tasks.table_tennis_sac.mdp.tracking import hit_track_terms
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

def hit_ref_pos(env, racket_body_name: str, *, sigma_t: float, sigma_p: float, w_pos: float = 1.0):
    center, center_vel, _ = _racket_body_state(env, racket_body_name)
    p_racket = center - env.scene.env_origins
    pos_term, _ = hit_track_terms(p_racket, center_vel, env._ht_p_ref_noisy, env._ht_v_ref_noisy,
        env._ht_tau_true, sigma_t=sigma_t, sigma_p=sigma_p, sigma_v=1.0, w_pos=w_pos, w_vel=0.0)
    return pos_term

def hit_ref_vel(env, racket_body_name: str, *, sigma_t: float, sigma_v: float, w_vel: float = 1.0):
    center, center_vel, _ = _racket_body_state(env, racket_body_name)
    p_racket = center - env.scene.env_origins
    _, vel_term = hit_track_terms(p_racket, center_vel, env._ht_p_ref_noisy, env._ht_v_ref_noisy,
        env._ht_tau_true, sigma_t=sigma_t, sigma_p=1.0, sigma_v=sigma_v, w_pos=0.0, w_vel=w_vel)
    return vel_term
```

Note: reward target is the **noisy** reference (D1=a) and the gate uses **`_ht_tau_true`** (privileged timing).

- [ ] **Step 4: Run test to verify it passes** — Run: `cd /data/PPO-pingpong && python -m pytest tests/test_hittrack_rewards.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** (draft → confirm) reward additions + test.

---

### Task 7: Termination — `hit_window_elapsed`

**Files:**
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/terminations.py` (append)
- Test: `tests/test_hittrack_terminations.py`

**Interfaces:**
- Produces: `hit_window_elapsed(env) -> Tensor[N] bool` — True once `episode_length_buf > hit_step + round(post_margin/step_dt)` (i.e. `tau_true < -post_margin`). Reuse existing `sac_time_out`, `joint_state_nan`.

- [ ] **Step 1: Write the failing test** (fake env)

```python
# tests/test_hittrack_terminations.py
from __future__ import annotations
import types, torch, pytest
term = pytest.importorskip("unitree_rl_lab.tasks.table_tennis_sac.mdp.terminations",
                           reason="isaaclab not installed")

def test_done_after_post_margin():
    env = types.SimpleNamespace(num_envs=2, device="cpu")
    env._ht_hit_step = torch.tensor([50, 50])
    env._ht_post_margin_steps = 12
    env.episode_length_buf = torch.tensor([55, 63])  # 55<=62 not done; 63>62 done
    done = term.hit_window_elapsed(env)
    assert not bool(done[0]) and bool(done[1])
```

- [ ] **Step 2: Run test to verify it fails** — Run: `python -m pytest tests/test_hittrack_terminations.py -v`. Expected: FAIL (`AttributeError hit_window_elapsed`) or skip if isaaclab absent.

- [ ] **Step 3: Write minimal implementation** (append to `terminations.py`)

```python
def hit_window_elapsed(env) -> torch.Tensor:
    """End the episode once the hit-time tracking window has fully closed."""
    margin = getattr(env, "_ht_post_margin_steps", 12)
    done = env.episode_length_buf.to(torch.long) > (env._ht_hit_step + margin)
    return done
```

`reset_reference_command` must also set `env._ht_post_margin_steps = round(post_margin_s/step_dt)`. Add that line to Task 4's `_ensure_ht_buffers` or reset (e.g. `env._ht_post_margin_steps = int(round(post_margin_s/step_dt))`). Update Task 4 accordingly when implementing this task.

- [ ] **Step 4: Run test to verify it passes** — Run: `python -m pytest tests/test_hittrack_terminations.py -v`. Expected: PASS (or skip).

- [ ] **Step 5: Commit** (draft → confirm) termination addition + test (+ the `_ht_post_margin_steps` line in `reference_commands.py`).

---

### Task 8: HitTrack env cfg + task registration (100 Hz)

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/hittrack_env_cfg.py`
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/__init__.py` (add `gym.register` for HitTrack)
- Test: `tests/test_hittrack_env_cfg.py`

**Interfaces:**
- Produces: `HitTrackEnvCfg`, `HitTrackPlayEnvCfg`; gym id `A1-TableTennis-SAC-HitTrack`.
- Constants (top of file): `HIT_PLANE_X=-1.37`, `HITTRACK_TARGET_XYZ=(OPP_TABLE_CENTER_X,0.0,TABLE_Z)`, `HITTRACK_BOX` (the synthetic `box` dict), `MAX_PREP_S=0.6`, `POST_MARGIN_S=0.12`, `SIGMA_T=0.03`, `SIGMA_P=0.03`, `SIGMA_V=0.3`, `W_POS=20.0`, `W_VEL=20.0`, `SUCCESS_POS=0.05`, `SUCCESS_VEL=0.2`, `REACH_Y=(-0.6,0.6)`, `REACH_Z=(0.7,1.5)`, `JOINT_POS_DELTA_HISTORY_LENGTH=5`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hittrack_env_cfg.py
from __future__ import annotations
import pytest
cfgmod = pytest.importorskip(
    "unitree_rl_lab.tasks.table_tennis_sac.hittrack_env_cfg", reason="isaaclab not installed")

def test_cfg_constructs_at_100hz_with_tracking_rewards():
    cfg = cfgmod.HitTrackEnvCfg()
    assert cfg.decimation == 2 and abs(cfg.sim.dt - 0.005) < 1e-9
    rew = cfg.rewards
    assert hasattr(rew, "hit_ref_pos") and hasattr(rew, "hit_ref_vel")
    # no ball-outcome reward terms wired
    for banned in ("hit_bonus", "return_cross_net", "bad_hit", "landing_placement"):
        assert not hasattr(rew, banned)

def test_task_registered():
    import gymnasium as gym
    import unitree_rl_lab.tasks.table_tennis_sac  # noqa: F401 triggers registration
    assert "A1-TableTennis-SAC-HitTrack" in gym.registry
```

- [ ] **Step 2: Run test to verify it fails** — Run: `python -m pytest tests/test_hittrack_env_cfg.py -v`. Expected: FAIL (module/attr missing) or skip if isaaclab absent.

- [ ] **Step 3: Write minimal implementation**

Build `hittrack_env_cfg.py` mirroring `env_cfg.py`'s class structure (`@configclass` for Scene/Actions/Observations/Rewards/Events/Terminations + `RobotEnvCfg`-style top cfg), but:
- **Scene**: reuse `A1TableTennisSacSceneCfg` (ball asset may stay in scene unused; the reset just won't launch it).
- **Actions**: same `mdp.JointDeltaTargetActionCfg(... action_scale=0.06 ...)` (100 Hz: halved from 0.12).
- **Observations.ActorCfg**: `joint_pos`, `joint_pos_delta_history(history_length=5)`, `hit_reference_command`, `racket_pos`, `racket_normal`, `hit_ref_pos_error(racket_body_name=RACKET_BODY_NAME)`, `last_action`; `enable_corruption=False`, `concatenate_terms=True`.
- **Observations.CriticCfg(ActorCfg)**: add `joint_vel`, `racket_vel`, `racket_ang_vel`, `racket_axes`, `hit_reference_command_clean`, `hit_ref_vel_error(racket_body_name=RACKET_BODY_NAME)`.
- **Rewards**: `hit_ref_pos = RewTerm(func=mdp.hit_ref_pos, weight=W_POS, params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_p": SIGMA_P})`; `hit_ref_vel = RewTerm(func=mdp.hit_ref_vel, weight=W_VEL, params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_v": SIGMA_V})`; plus the kept regularizers copied verbatim from `env_cfg.py` (`action_rate`, `joint_acc`, `joint_jerk`, `joint_limit` margin 0.05, `joint_effort_margin` margin_frac 0.85). **Do not wire any ball-outcome term.**
- **Events**: `reset_robot = EventTerm(func=mdp.reset_robot_to_ready_pose, mode="reset", params={... SAC_READY_JOINT_POS ...})`; `reset_reference = EventTerm(func=mdp.reset_reference_command, mode="reset", params={all Task-4 params})`; `update_ref = EventTerm(func=mdp.update_hit_track_state, mode="interval", interval_range_s=(0.01,0.01), params={"success_pos_thresh": SUCCESS_POS, "success_vel_thresh": SUCCESS_VEL})`. **No** `launch_ball`, `apply_air_drag`, `reset_sac_episode_state`, `track_episode`.
- **Terminations**: `time_out = DoneTerm(func=mdp.sac_time_out, time_out=True)`, `nan_state = DoneTerm(func=mdp.joint_state_nan)`, `hit_done = DoneTerm(func=mdp.hit_window_elapsed)`.
- **`__post_init__`**: `self.decimation = 2`, `self.episode_length_s = 0.72`, `self.sim.dt = 0.005`, `self.sim.render_interval = self.decimation`, physics/material lines copied from `env_cfg.py`, robot base pos/rot/lift copied from `env_cfg.py`.
- Add `mdp.reset_reference_command`, `update_hit_track_state`, `hit_window_elapsed`, `hit_ref_pos`, `hit_ref_vel`, `hit_reference_command`, etc. are reachable via `mdp` (they're imported through `mdp/__init__.py`'s `from .X import *`; confirm each new module is imported there — append `from .reference_commands import *`, `from .reference_planner import *`, `from .tracking import *`, `from .reference_source import *` to `mdp/__init__.py`).

Then register in `__init__.py`:

```python
gym.register(
    id="A1-TableTennis-SAC-HitTrack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackPlayEnvCfg",
    },
)
```

- [ ] **Step 4: Run test to verify it passes** — Run: `python -m pytest tests/test_hittrack_env_cfg.py -v`. Expected: PASS (or skip if isaaclab absent — then run a manual Isaac smoke step per Step 4b).

- [ ] **Step 4b: Isaac smoke (if a GPU/Isaac env is available)**: launch the play env for a handful of steps with random actions; assert no exceptions and that `env._ht_tau_true` decreases and `env._ht_success` populates by episode end. Document the exact launch command in `docs/command_lines.md`.

- [ ] **Step 5: Commit** (draft → confirm) `hittrack_env_cfg.py`, `__init__.py`, `mdp/__init__.py` exports, test.

---

### Task 9: Baking script — real recording → 100 Hz reference streams

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/sac_table_tennis/bake_hittrack_references.py` (alongside `create_serve_states.py`; adjust path to the actual `table_tennis_sac/` dir)
- Test: `tests/test_hittrack_baking.py`

**Interfaces:**
- Produces (pure numpy, Isaac-free): `load_recording(path) -> dict` of arrays `t, serve_id, mocap[N,6], kf[N,6], kf_pred[N,6]` (kf_pred = `y,z,vx,vy,vz,tau`); `true_crossing(mocap_traj, t, hit_plane_x) -> (state6, t_cross)` (reuse the `state_at_x` interpolation logic from `create_serve_states.py`); `resample_to_grid(t, values, grid_t) -> array` (linear); `bake(records, *, hit_plane_x, step_dt, reach_y_range, reach_z_range) -> dict` writing per-serve resampled `noisy_ball_stream[S,T,5]`, `clean_ball_state[S,5]`, `tau_true[S,T]`, `valid_len[S]`, `reachable[S]`. Output `hittrack_references.npz`.
- **Coordinate transform**: apply the same real→sim frame mapping (`ROBOT_SIDE`, axis offsets) used by `create_serve_states.py` before computing crossings.

- [ ] **Step 1: Write the failing test** (synthetic in-memory fixture; no real data needed)

```python
# tests/test_hittrack_baking.py
from __future__ import annotations
import numpy as np
from unitree_rl_lab.tasks.table_tennis_sac.bake_hittrack_references import (
    true_crossing, resample_to_grid,
)

def test_true_crossing_linear():
    # straight-line ball crossing x=-1.37 at t=0.5
    t = np.linspace(0.0, 1.0, 101)
    x = 1.0 - 4.0 * t          # crosses -1.37 at t=(1+1.37)/4=0.5925
    y = np.zeros_like(t); z = 1.0 + 0.0 * t
    traj = np.stack([x, y, z, -4*np.ones_like(t), 0*t, 0*t], axis=1)
    state, t_cross = true_crossing(traj, t, hit_plane_x=-1.37)
    assert abs(state[0] - (-1.37)) < 1e-6
    assert abs(t_cross - 0.5925) < 1e-3

def test_resample_linear_midpoint():
    t = np.array([0.0, 0.02]); v = np.array([[0.0], [2.0]])
    grid = np.array([0.01])
    out = resample_to_grid(t, v, grid)
    assert abs(out[0, 0] - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails** — Run: `python -m pytest tests/test_hittrack_baking.py -v`. Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — port `load_vrpn`/`state_at_x` patterns from `create_serve_states.py`; add `resample_to_grid` (np.interp per column), `true_crossing` (interpolated plane crossing), and `bake`/`main` writing the npz. Parse the new recording columns (`t, serve_id, mocap6, kf6, kf_pred6`). Filter to fly-through serves (or honor a `contact` column when present). `argparse` with `--data-dir`, `--out`, `--hit-plane-x -1.37`, `--step-dt 0.01`, reach-range flags. Provide concrete column indices in a docstring header.

- [ ] **Step 4: Run test to verify it passes** — Run: `python -m pytest tests/test_hittrack_baking.py -v`. Expected: PASS (2 tests).

- [ ] **Step 5: Commit** (draft → confirm) baking script + test.

> **Gating note:** running `bake_hittrack_references.py` on real serves is blocked on the user recording the new-format 120 Hz logs (mocap + deployment-KF + KF-pred). The script and its unit tests do NOT depend on that data; curriculum ①/② (Task 10) train without it.

---

### Task 10: Baked-real source wiring + curriculum switch

**Files:**
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/reference_commands.py` (add a baked-npz source branch)
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/hittrack_env_cfg.py` (add `HITTRACK_USE_BAKED` flag + path; build reset params from baked source when on)
- Test: `tests/test_hittrack_baked_source.py`

**Interfaces:**
- Produces: `load_baked_references(path, device) -> dict` of tensors; `reset_reference_command` gains an optional `baked` dict param — when present, per-env it **samples a serve index** and slices `noisy_ball_stream/clean_ball_state/tau_true/valid_len` from the baked tensors (no synthetic sampling, no synthetic noise), then runs the same planner-at-reset path. Curriculum: ① fixed (synthetic, single state in `box` collapsed to a point) → ② small-random (synthetic box + `noise_params`) → ③ baked real.

- [ ] **Step 1: Write the failing test** (tiny fixture npz written in-test)

```python
# tests/test_hittrack_baked_source.py
from __future__ import annotations
import types, torch, numpy as np, tempfile, os
from unitree_rl_lab.tasks.table_tennis_sac.mdp import reference_commands as rc

def _write_fixture(path, S=3, T=40):
    np.savez(path,
        noisy_ball_stream=np.zeros((S, T, 5), np.float32),
        clean_ball_state=np.tile(np.array([-1.37,0,1.0,-4,0,-0.3], np.float32), (S,1)),
        tau_true=np.tile(np.linspace(0.4, 0.4-(T-1)*0.01, T, dtype=np.float32), (S,1)),
        valid_len=np.full((S,), T, np.int64), reachable=np.ones((S,), bool))

def test_baked_source_loads_and_resets():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ref.npz"); _write_fixture(p)
        baked = rc.load_baked_references(p, device="cpu")
        env = types.SimpleNamespace(num_envs=2, device="cpu",
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            scene=types.SimpleNamespace(env_origins=torch.zeros(2,3)), step_dt=0.01)
        rc.reset_reference_command(env, None, hit_plane_x=-1.37, target_xyz=(0.685,0,0.76),
            box=None, max_prep_s=0.4, post_margin_s=0.0, step_dt=0.01,
            reach_y_range=(-0.6,0.6), reach_z_range=(0.7,1.5), baked=baked)
        assert env._ht_p_ref_clean.shape == (2, 3)
        assert (env._ht_tau_true > 0).all()
```

- [ ] **Step 2: Run test to verify it fails** — Run: `python -m pytest tests/test_hittrack_baked_source.py -v`. Expected: FAIL (`AttributeError load_baked_references` / unexpected `baked` kwarg).

- [ ] **Step 3: Write minimal implementation** — add `load_baked_references`; in `reset_reference_command`, when `baked is not None`, draw `idx = randint(0, S, (k,))`, set `clean = baked["clean_ball_state"][idx]`, `noisy_stream = baked["noisy_ball_stream"][idx]` (pad/truncate to `n_steps`), `tau_true_stream = baked["tau_true"][idx]`, `valid_len = baked["valid_len"][idx]`, then run the same planner-at-reset path (skip synthetic sampling/noise). Add `HITTRACK_USE_BAKED`/`HITTRACK_BAKED_PATH` to the env cfg and pass `baked` into the reset params when enabled.

- [ ] **Step 4: Run test to verify it passes** — Run: `python -m pytest tests/test_hittrack_baked_source.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** (draft → confirm) baked-source wiring + cfg flag + test.

---

## Validation Metrics (wire into the training logger when the trainer is selected)

Per spec §11, expose: `ref_pos_error_at_hit` / `ref_vel_error_at_hit` (vs clean via `_ht_pos_err_at_hit`/`_ht_vel_err_at_hit`, and a vs-noisy variant), `success_rate` (`_ht_success`), `hit_time_abs_error`, `reachable_reference_rate`, `joint_limit_violation_rate`, `effort_margin_mean`, per-`reward_terms/*`. Deployment dry-run + offline return-quality validation (`hitting.predict_landing_xy`/`predict_z_at_x` on realized `p/v/n`) stay **outside** the RL reward. The trainer choice (custom SAC in `sac.py` vs `rsl_rl` PPO) is deferred; the env is trainer-agnostic (asymmetric `policy`/`critic` obs groups already supported by `runtime.split_actor_critic_obs`).

## Deviations from the spec (flagged for review)

- **No hard deletion** of ball observation/reward/event functions: `observations.py`, `rewards.py`, `events.py` are shared with `A1-TableTennis-SAC-Catch`, which the spec keeps alive and untouched. Deleting would break Catch. The plan is purely additive; deletion is deferred to a future Catch-retirement cleanup task.
- Pure kernels were factored into Isaac-free modules (`tracking.py`, `reference_source.py`) to honor the repo's Isaac-free CI test convention (`tests/test_sac_table_tennis_pipeline.py`).

---

## Self-Review

**Spec coverage:** §2 noisy/clean → Tasks 4,5 (noisy actor / clean critic+success). §3 recording+baking → Task 9. §4 runtime planner → Tasks 2,4. §5 observations → Task 5 (Kd=5, 10-dim, vel-error critic-only). §6 action (JointDelta, scale 0.06) → Task 8. §7 reward (D1=a noisy, D2 gaussian gate tau_true, D3 full-vector, σ/w numbers, normal off) → Tasks 1,6,8. §8 episode/reset (E1 capped horizon, E2 0.72s, E3 soft+loose filter, E4 fixed pose, E5 success vs clean) → Tasks 3,4,7,8. §9 replay/trainer-agnostic → Validation section. §10 implementation slice → Tasks 1-10 (planner=runtime, baking=raw-only). §11 metrics → Validation section. §12 migration/curriculum → Tasks 8 (①②) + 10 (③). §13 rejected alternatives → encoded in design (no IK, no clean-target reward, gaussian gate not single-fire, full-vector, planner not baked, 100 Hz not 120). §14 open items → flagged in Tasks 8/9/10 params.

**Placeholder scan:** No TBD/TODO; every code step has concrete code or a concrete edit description with exact names. Task 6 Step 2 notes the kernel test already passes and gives the explicit alternative red test.

**Type consistency:** `_ht_*` buffer names, `plan_hit_reference` signature, `hit_track_terms` signature, `hit_reference_command` 10-dim ordering `[p(3),v(3),n(3),tau(1)]`, and `RACKET_BODY_NAME` usage are consistent across Tasks 2/4/5/6/8. `reset_reference_command` param set is consistent between Tasks 4, 8, and 10 (10 adds optional `baked`).
