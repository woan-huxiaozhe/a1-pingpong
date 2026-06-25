# Traditional-Controller Swing Analysis — Plan

**Central question:** Theory *and* the deployed traditional task-space controller both
return the ball with a paddle-tip cruise speed of ~1.3 m/s; the drag-aware
`ideal_racket_velocity` solve independently lands on the same ~1.3 m/s. So the physics
target is validated and provably reachable. **Yet SAC plateaus at a mis-aimed ~48° / ~3.1 m/s
lob (valid_return ~2%)** — while a good return *flickers in and out* during training (proof it
is kinematically reachable). The goal of this analysis is to characterize the traditional
hitting motion precisely (the existence proof) and from the gap determine **why RL fails to
converge to it.**

> Status: PLAN ONLY. No FK / velocity / orientation has been computed yet.

---

## 0. What we already know (inputs, not to re-derive)
- `ideal_racket_velocity` (sim, drag-aware, e=0.75) on in-distribution contacts returns:
  paddle tip ~1.2–1.4 m/s at ~20–28° elevation; ball launch ~4.7 m/s at the neutral 28°;
  blade face-normal elevation ~20–28°. **Most return energy is reflection of the incoming
  ball, not swing speed.**
- RL plateau (both projection and full-vector velocity reward): ball off the paddle at
  vx≈+2.05, vz≈+2.30 → ~48° / ~3.1 m/s, apex 1.33 m, lands ~0.30 m short. Blade orientation
  is currently **unrewarded**.
- Leading prior: ball direction is set by paddle velocity **and** the blade normal; RL only
  ever shaped velocity, so the rebound is mis-aimed. This analysis must confirm or refute that
  against the real motion, and weigh it against the other hypotheses below.

## 1. Data & tooling inventory
- **Target log:** `/home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_2.csv`
  (78-line `#` preamble incl. `kps`/`kds`; header line 79; ~1212 rows @ ~100 Hz ≈ 12 s).
  Cross-check siblings: `pingpong_log_1.csv` (larger, more swings), `100Hz_*`, `_3.csv`.
- **Columns:** `ros_time_s, swing_id, stage, ee_plan_pos/vel_{xyz}, q_plan_0..6, dq_plan_0..6,
  q_actual_0..6, dq_actual_0..6, tau_actual_0..6, tau_ff_0..6, orientation_mapper_*`.
- **Stages:** `tracking → to_hit → cruise → follow-through` (+ `recovery`/`idle`).
  **`cruise` = the contact / hitting window** (confirmed by tool + user).
- **FK:** pinocchio, URDF `robotbase/src/armcontrol/urdf/A1/a1_r.urdf`, frame `right_load`
  (paddle load). **Reuse `visualize_log.py`**: `load_log`, `segment_trajectories`,
  `compute_fk_series` (returns tip pos / rpy / rotation matrix / lin_vel / ang_vel),
  `report_cruise_orientation` (already computes EE-local-axis-vs-velocity angles during cruise —
  i.e. *which axis is the paddle-face normal and how it sits relative to the swing*).
- **Arm identity (re-verify, memory is 19 days old):** real A1 RIGHT arm == sim USD; joints 1:1
  `q_*_{i} ↔ joint_yb_{i+1}`, no sign flip; q_plan respects USD limits; q_actual overshoots +
  has a J5 oscillation → prefer `q_plan` for the reference motion.

### Prerequisite / risk
- **pinocchio is broken in the isaac conda env** (`No module named
  'pinocchio.pinocchio_pywrap_default'`). Before any FK: identify the python env the user runs
  `visualize_log.py` in (likely a robotbase_gripper venv/system python), or stand up a working
  pinocchio. Fallback: FK from the URDF with an independent parser, or via the sim's articulation.

## 2. Reference frames (keep comparisons valid)
- Traditional FK lives in the **real robot base frame**; the sim is Isaac **world** with the
  robot at `SAC_ROBOT_BASE_X`, rotated per `ROBOT_SIDE`.
- **Frame-invariant (no transform needed) — do these first:** tip speed `|v|`,
  angle(tip-velocity, paddle-normal), angle(paddle-normal, vertical), joint angles/velocities,
  cruise duration, per-joint wind-up excursion.
- **Frame-dependent (needs base→world transform):** absolute tip position vs serve geometry,
  launch *elevation in world*. Establish/borrow the base→world transform (reuse the backhand
  frame-alignment work if it exists) only where strictly needed.

## 3. Extraction steps (compute — NOT yet executed)
Run on `pingpong_log_2.csv`, cross-check on `_1.csv`. Headless (`--no-show`), dump per-swing
numbers to JSON.

1. **Enumerate swings.** Group by `swing_id`; per swing list the stage sequence, per-stage
   sample counts, and durations; keep only *complete* swings (`tracking→to_hit→cruise→
   follow-through`). Output: N usable strokes and their timing.
2. **Cruise tip kinematics (frame-invariant).** Per swing: `|v(t)|` profile and the value at
   contact; tip-velocity direction (elevation/azimuth in base frame); paddle-face normal (the
   `report_cruise_orientation` ~90° axis) and its angle to `v` and to vertical; tip angular
   velocity. **Cross-validate:** is cruise `|v| ≈ 1.3 m/s`? Does (v_in, v_out, normal) match the
   reflection geometry `ideal_racket_velocity` assumes (normal ≈ bisector of −v_in and v_out)?
3. **Approach / wind-up structure.** `to_hit` per-joint excursion and `dq` build-up into cruise;
   how far each joint travels; time budget (`to_hit` + `cruise` durations).
4. **Joint-space demands during cruise (KEY discriminator).** Peak `|dq_plan|` per joint and
   peak joint acceleration during cruise. Compare to (a) `A1_ARM_VELOCITY` limits and (b) what
   the RL action parameterization can physically produce: `action_scale=0.12` rad,
   `smoothing=0.5`, `step_dt=0.02 s`, `max_joint_velocity` clamp → effective reachable
   joint-delta and `dq` per control step over the cruise window.
5. **Contact configuration.** Tip pose (position + normal) at the cruise sample nearest contact;
   its relation to the intended ball/target geometry (the log may lack ball state → use the
   controller's planned target / `ee_plan_*`).
6. **Plan-vs-actual fidelity (secondary).** Does `q_actual` track `q_plan` through cruise
   (tracking error, J5 oscillation)? Bears on sim2real, not on the core RL question.

## 4. Build the RL-comparable reference
- Reduce the traditional cruise to "what RL should produce at contact": tip velocity vector
  (mag + direction), paddle normal, and the joint configuration + joint velocities — expressed
  frame-invariantly, and in sim world where the transform is known.
- (Optional, separate) Short sim rollout of the current SAC checkpoint to log the policy's
  actual contact state (tip vel + normal + joints) for a direct gap comparison.

## 5. Hypotheses for "why RL can't learn it" + discriminators
- **H1 — Orientation underspecified.** RL shapes tip velocity but not the blade normal.
  *Discriminator:* traditional cruise holds a tight, specific normal-vs-velocity angle (low std)
  → an unconstrained normal in RL explains the 48° rebound. *Fix:* add a normal-alignment reward
  toward the `n` already returned by `ideal_racket_velocity`.
- **H2 — Action parameterization too weak/slow.** Cruise needs joint speeds/timing beyond
  `action_scale·smoothing·velocity-cap` within the contact window. *Discriminator:* step 3.4
  demanded `dq` vs reachable `dq`. If demanded ≫ reachable → RL is capped. *Fix:* raise
  `action_scale` / reduce `smoothing` / adjust `decimation`, not the reward.
- **H3 — Reward credit / exploration.** The good contact is a narrow target; sparse terminal
  return reward + annealed entropy can't bootstrap; the traditional motion uses a *planned*
  wind-up RL never explores. *Discriminator:* cruise-window width (3.2) + wind-up excursion
  (3.3) → how precise the timing must be; consistent with the observed flicker-in/out. *Fix:*
  imitation warm-start from this very trajectory / curriculum / stronger terminal credit.
- **H4 — Frame/target or serve-distribution mismatch.** Sim serve distribution differs from the
  real serves these swings handled, or the ideal target differs. *Discriminator:* compare
  traditional contact height/incoming geometry to the sim serve config.
- **H5 — Ready-pose / wind-up affordance.** `SAC_READY_JOINT_POS` may not afford the `to_hit`
  wind-up the traditional swing relies on. *Discriminator:* compare sim ready pose to the
  traditional pre-`to_hit` configuration.

## 6. Deliverables
- Per-swing report (numbers + 2–3 plots): cruise tip speed/direction, normal-vs-v angle, joint
  `dq` peaks vs limits/reachable, cruise duration, wind-up excursion.
- One-line per-hypothesis verdict (supported / refuted / inconclusive) with the numbers.
- A recommended next RL change justified by the dominant supported hypothesis — candidates:
  add normal-alignment reward (H1), retune action parameterization (H2), imitation warm-start
  from a specific `log_2` swing (H3).

## 7. Execution order (once approved)
1. Stand up a working pinocchio env (or fallback FK).
2. Wrap the `visualize_log.py` functions in a headless stats script (extend
   `report_cruise_orientation` to dump per-swing JSON).
3. Run on `_2.csv` (+ `_1.csv` cross-check) → report.
4. (Optional) short SAC-checkpoint rollout to capture the RL contact state for the gap table.
5. Write the per-hypothesis verdict + recommended RL change.

## 8. Out of scope
- Changing RL reward/training (follows from the verdict).
- Sim2real tracking fidelity beyond the quick cruise-tracking sanity check.
