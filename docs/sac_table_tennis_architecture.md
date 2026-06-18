# A1 Table Tennis SAC Training Architecture

Last updated: 2026-06-17

This document is the architecture note for the independent SAC table-tennis pipeline. When the SAC task's observation, action, reward, replay, event logic, latency model, training loop, or TensorBoard logging changes, update this file in the same change.

## 1. Scope

The current pipeline trains a single SAC policy for `A1-TableTennis-SAC-Catch`.

Current goal:

- First make the policy learn contact and then `return` on a fixed, slow, middle incoming ball.
- Use off-policy SAC plus event replay to reuse rare `near_miss`, `hit`, `return`, and `valid_return` transitions.
- Keep the training path independent from the existing imitation/PPO/reference-motion stack.

Current non-goals:

- No expert imitation.
- No `ReferenceResidualJointAction`.
- No HER in active training yet. HER reward recomputation exists only as a scaffold.
- Actor now receives a deployment-style estimated hit command; critic additionally receives the clean simulator hit command.
- No serve randomization, spin, target-y curriculum, or real sensor latency model yet.

Main entry points:

- Training: `scripts/sac_table_tennis/train.py`
- Playback/evaluation: `scripts/sac_table_tennis/play.py`
- Task registration: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/__init__.py`
- Environment config: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/env_cfg.py`
- SAC implementation: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/sac.py`
- Replay/event tables: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/replay.py`
- MDP terms: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/`

## 2. Environment

Task id:

```text
A1-TableTennis-SAC-Catch
```

Robot and scene:

- Robot: A1 fixed base.
- Controlled joints: right arm, 7 DOF.
- Controlled joint names:
  - `joint_yb_1`
  - `joint_yb_2`
  - `joint_yb_3`
  - `joint_yb_4`
  - `joint_yb_5`
  - `joint_yb_6`
  - `joint_yb_7`
- Racket body: `Link_yb_paddle`.
- Robot side: `ROBOT_SIDE = -1`.
- Robot base x: `SAC_ROBOT_BASE_X = -(1.37 + 0.45) = -1.82`.
- Robot contact/evaluation x: `SAC_ROBOT_X = -1.47`.
- Own table x range: `[-1.37, 0.0]`.
- Opponent table x range: `[0.0, 1.37]`.
- Net x: `0.0`.

Current incoming ball:

```python
SAC_FIXED_MIDDLE_BALL = {
    "x_range": (1.25, 1.25),
    "y_range": (0.0, 0.0),
    "z_range": (1.05, 1.05),
    "vx_range": (-3.4, -3.4),
    "vy_range": (0.0, 0.0),
    "vz_range": (1.57, 1.57),
}
```

The ball is reset once per episode by `launch_ball`. It now starts near the opponent-side table edge instead of spawning close to the middle of the table, because the previous `x=0.35, z=1.10, vx=-2.4, vz=0.2` setup tended to land on the robot half too early and could show repeated bounces on the robot side before the arm interacted with it.
With the current fixed ball and `SAC_ROBOT_X = -1.47`, the analytic bounce model predicts a table bounce around `x = -0.284 m` and a strike-plane height of about `1.06 m` after `0.80 s`, slightly above the current ready-pose paddle-center height.

Robot reset:

- Every episode reset now writes the fixed-base robot root back to its default root state.
- The locked lift joint is initialized through `SAC_READY_LIFT_POS = -0.22`.
- Isaac reports the lift joint's legal range as `[-0.800, -0.050]`. This value sits between the original SAC/A1 default lift `-0.28` and the higher tested lift `-0.06`.
- A static USD joint-anchor FK check estimates the current ready-pose paddle-center height at about `1.016 m`; the fixed ball above is tuned to cross `x = -1.47` slightly above that height.
- The controlled right arm is reset to `SAC_READY_JOINT_POS`.
- Current ready pose:
  - `[1.53, -0.39, 1.60, -1.32, 0.0, 1.0, -1.845288]`
- Joint velocities are reset to zero.
- The `JointDeltaTargetAction` internal target is also reset so the first post-reset action is relative to the ready pose, not to the previous episode's target.

Timing:

- Physics dt: `0.005 s`.
- Decimation: `4`.
- Policy/env step: `0.02 s`, or 50 Hz.
- Episode length: `2.5 s`, about 125 policy steps.
- Event tracking interval: `0.02 s`.
- The same target action is held across 4 physics substeps.

## 3. Observation

The observation is kept close to real-machine feasibility. The actor receives a noisy deployment-style hit command (`p_hit`, `tau`) and recent joint-position deltas, but does not receive privileged clean intercept, clean joint velocity, or clean racket FK features.

Current groups:

- `policy`: actor observation.
- `critic`: critic observation.
- `policy` is the deployment-facing actor input and must stay compatible with real robot signals.
- `critic` is the training-only value input. It contains the full actor observation plus privileged simulator state. Actor is 51D, critic is 92D.

### Actor Obs (`policy`)

Actor dimension:

```text
7 + 3*7 + 4*3 + 4 + 7 = 51
```

Actor terms:

| Term | Dim | Meaning |
| --- | ---: | --- |
| `joint_pos` | 7 | Right-arm joint position relative term from IsaacLab MDP. |
| `joint_pos_delta_history` | 21 | Three recent deployable joint-position deltas: `[q_t-q_{t-1}, q_t-q_{t-2}, q_t-q_{t-3}]`. |
| `ball_pos_history` | 12 | `K=4` ball positions in env/world frame, newest sample first. |
| `estimated_hit_command` | 4 | Deployment-style `[p_hit_x, p_hit_y, p_hit_z, tau]` predicted at the robot strike plane with KF-like phase-dependent noise. |
| `last_action` | 7 | Previous normalized action from the action manager. |

Actor exclusions:

- No clean simulator `joint_vel`.
- No clean `racket_pos`, `racket_normal`, or racket velocity.
- No clean `predicted_hit_point` or clean `time_to_intercept`.
- No explicit `ball_vel`; the actor can infer ball velocity from `ball_pos_history`.

Actor details:

- `ball_pos_history` is sampled at policy-step cadence, so adjacent samples are 20 ms apart.
- The newest `ball_pos_history` entry is the current ball position. The history is not an explicit delay.
- Joint velocity is inferred from `joint_pos_delta_history`, using position differences instead of raw/clean velocity.
- The actor's `estimated_hit_command` uses cached noise per policy step so the actor and critic share the same deployment estimate when both observation groups are evaluated.
- `p_hit_x` is fixed at the robot strike plane; noise is injected into predicted y/z and `tau`. The default actor-side noise shrinks from `0.03 m` / `0.015 s` far from contact to `0.01 m` / `0.002 s` near contact. At `tau = 0.25 s`, this gives about `16 mm` per-axis y/z standard deviation, or about `20 mm` expected 2D position error, matching the current KF-level deployment estimate.
- No ball-history noise, dropout, or latency is injected now.

### Critic Obs (`critic`)

Critic dimension:

```text
51 actor dims + 41 privileged dims = 92
```

Critic includes every actor term above, plus:

| Term | Dim | Meaning |
| --- | ---: | --- |
| `joint_vel` | 7 | Clean simulator right-arm joint velocity. |
| `racket_pos` | 3 | Clean FK paddle blade-center position in env/world frame, minus env origin. |
| `racket_normal` | 3 | Clean FK racket local +Y normal rotated into world frame. |
| `ball_vel` | 3 | Clean simulator ball linear velocity. |
| `racket_vel` | 3 | Clean paddle blade-center linear velocity. |
| `racket_ang_vel` | 3 | Clean racket body angular velocity. |
| `ball_pos_rel_racket` | 3 | Clean ball position relative to the paddle blade center. |
| `ball_vel_rel_racket` | 3 | Clean ball velocity relative to the paddle blade center velocity. |
| `racket_axes` | 9 | Clean FK racket local X/Y/Z axes in world frame. |
| `groundtruth_hit_command` | 4 | Clean simulator `[p_hit_x, p_hit_y, p_hit_z, tau]` at the fixed strike plane. |

Critic details:

- Critic input is `obs_critic`; actor input is `obs_actor`.
- SAC stores and samples both observation groups separately in replay.
- Actor updates still sample actions from `obs_actor`; critics evaluate those actions using `obs_critic`.
- This realizes the target-command asymmetric stage: the critic sees both the actor's noisy deployment hit command and the clean simulator hit command, while the actor stays on deployable signals.
- A future step can additionally delay/noise the actor ball history while keeping critic observations clean.

Shared geometry note:

- The racket reference point used by `racket_pos`/`racket_normal` and all racket-based reward terms is the paddle blade center, i.e. the `Link_yb_paddle` body origin plus a local `+Z 0.045 m` offset (`observations.RACKET_OFFSET_Z`). See §5 for why this matters for center-vs-handle contact.

## 4. Action

Action term:

```text
JointDeltaTargetAction
```

Action space:

```text
action in [-1, 1]^7
```

Mapping:

```python
raw_target = q_current + clamp(action, -1, 1) * action_scale
target = previous_target + smoothing * (raw_target - previous_target)
target = rate_limit(target, previous_target, max_joint_velocity * step_dt)
q_target = clamp(target, soft_joint_lower + margin, soft_joint_upper - margin)
```

Current parameters:

- `action_scale = 0.12 rad`.
- `smoothing = 0.5`.
- `max_joint_velocity = [8, 8, 8, 20, 20, 20, 20] rad/s`, matching the A1 right-arm actuator velocity limits currently configured in `assets/robots/a1.py`.
- `step_dt = 0.02 s`.
- Target-space max delta per policy step from velocity limits: `[0.16, 0.16, 0.16, 0.40, 0.40, 0.40, 0.40] rad`.
- Because `action_scale = 0.12 rad`, the velocity limiter does not normally bind on a single raw action step; it is kept aligned with actuator limits for later larger action scales or target jumps.
- Limit margin inside the action term: `1e-3 rad`.

Control semantics:

- SAC outputs normalized joint deltas.
- The action term converts them into joint position targets.
- Isaac/robot low-level joint position control tracks those targets.
- The policy does not output torque, joint velocity targets, or a 14D `[q_target, qdot_target]` command.

Underlying A1 right-arm actuator limits:

| Joint | Velocity Limit | Effort Limit | Stiffness | Damping |
| --- | ---: | ---: | ---: | ---: |
| `joint_yb_1` | `8 rad/s` | `28 Nm` | `250` | `1.0` |
| `joint_yb_2` | `8 rad/s` | `28 Nm` | `250` | `1.0` |
| `joint_yb_3` | `8 rad/s` | `28 Nm` | `250` | `1.0` |
| `joint_yb_4` | `20 rad/s` | `8 Nm` | `120` | `0.5` |
| `joint_yb_5` | `20 rad/s` | `8 Nm` | `120` | `0.5` |
| `joint_yb_6` | `20 rad/s` | `8 Nm` | `120` | `0.5` |
| `joint_yb_7` | `20 rad/s` | `8 Nm` | `120` | `0.5` |

Joint targets are clamped by the action term. Joint-limit termination is currently disabled for SAC training because early runs learned to terminate episodes by driving `joint_yb_6` into the high limit before the ball reached the racket. Limit avoidance is handled by the `joint_limit` barrier reward instead.

## 5. Reward

IsaacLab reward terms are configured in `RewardsCfg`. In practice, the event weights were scaled by `1 / 0.02 = 50` so that after 20 ms reward integration they behave like the intended sparse bonuses:

- hit: about `+0.04`.
- quality hit: up to `+0.20`.
- centered hit: up to `+0.16`.
- return/cross-net: about `+0.30`.
- valid return: about `+0.50`.
- landing placement: up to `+0.50`.
- miss: about `-0.10`.
- bad hit: about `-0.06`.

Current reward terms:

| Term | Weight | Effective one-step scale at 20 ms | Meaning |
| --- | ---: | ---: | --- |
| `racket_ball_proximity` | `0.0` | disabled | Pre-hit dense racket-ball distance shaping. Kept only as an ablation hook because the 260k resume showed it correlated with misses rather than useful returns. |
| `racket_approach` | `0.0` | disabled | Pre-hit racket closing-speed shaping. Kept only as an ablation hook. |
| `racket_face_target` | `0.0` | disabled | Pre-hit racket-normal alignment shaping. Kept only as an ablation hook. |
| `racket_normal_swing` | `0.0` | disabled | Pre-hit racket-normal swing-speed shaping. Kept only as an ablation hook. |
| `hit` | `2.0` | `+0.04` | Step event reward when first racket contact is detected. Kept small so passive contact is not enough. |
| `quality_hit` | `20.0` | up to `+0.40` | First-hit basic outgoing-quality reward. It ramps outgoing x speed from `1.5 -> 3.5 m/s` and uses only a shallow vertical gate (`up_speed -0.2 -> +0.2 m/s`) so downward hits get little/no credit. The base score is multiplied by the per-hit centering gate (`center_sigma = 400`, `center_gate_floor = 0.15`) so a blade-edge contact cannot collect the full reward. |
| `hit_centered` | `8.0` | up to `+0.16` | First-hit event reward from the cached in-plane offset between ball and blade center (`exp(-400 * d_plane^2)`), separated from speed quality so center contact is not diluted by trajectory terms. |
| `return_cross_net` | `15.0` | `+0.30` | Step event reward when a hit ball crosses the net toward the opponent side above net height. |
| `valid_return` | `25.0` | `+0.50` | Step event reward when a hit ball lands on the opponent table. |
| `landing_placement` | `25.0` | up to `+0.50` | Valid-return event reward for actual landing near the opponent-half center (`sigma_x = 0.25`, `sigma_y = 0.30`), multiplied by the per-hit centering gate (`center_sigma = 400`, `center_gate_floor = 0.15`). This is the largest center-blind term an edge hit could farm, so gating it is the high-leverage half of the center-contact fix. |
| `miss` | `-5.0` | `-0.10` | Penalty when the ball passes the robot or falls before any hit. |
| `bad_hit` | `-3.0` | `-0.06` | Penalty when a hit ball lands on own table or goes out/falls after hit, or fails to cross the net within the hit-to-return timeout. |
| `post_hit_outgoing` | `0.0` | disabled | Kept as an ablation hook. Disabled because it duplicated `quality_hit` and `post_hit_landing_prediction` once valid returns became common. |
| `post_hit_net_progress` | `0.0` | disabled | Kept as an ablation hook; disabled because it rewarded shallow progress toward the net without placement quality. |
| `post_hit_net_clearance` | `1.0` | up to `0.02` per step | After hit, rewards predicted net clearance enough to preserve crossing gradient without dominating total reward. |
| `post_hit_landing_prediction` | `5.0` | up to `0.10` per step | After hit, predicts landing `x/y` at table height and scores proximity to opponent-half center (`sigma_x = 0.25`, `sigma_y = 0.30`) only if predicted landing is in bounds. |
| `post_hit_lob_penalty` | `0.0` | disabled | Kept as an ablation hook only. The 2026-06-16 resume showed direct high-arc penalty was too blunt: it lowered height slightly but also reduced return success. Prefer shaping target placement and center contact first. |
| `action_rate` | `-0.005` | regularizer | Penalizes action changes. |
| `joint_acc` | `-1e-6` | regularizer | Penalizes joint acceleration. |
| `joint_limit` | `-1.0` | barrier regularizer | Penalizes joints inside a `0.20 rad` margin before the hard limits. |

The earlier pre-hit `racket_forward_push` term was removed because fixed world-x end-effector push encouraged wrist-pitch limit-seeking. The later pre-hit shaping terms (`racket_ball_proximity`, `racket_approach`, `racket_face_target`, `racket_normal_swing`) are now disabled by default: after the policy can already contact the ball, the 260k resume showed these terms mostly rewarded chasing/posing near the ball and correlated with misses, not valid returns. Pure `hit` is intentionally small; `quality_hit` now only checks that first contact has enough outgoing velocity and is not hit downward. Center contact is handled by `hit_centered` plus a multiplicative centering gate on `quality_hit` and `landing_placement`; final placement is handled by `post_hit_landing_prediction` and `landing_placement`.

2026-06-11 retune (fix "hit but never return" / weak-swing / handle-contact). The previous run converged to `hit_rate = 1.0`, `return_rate = 0.0`, `bad_hit_rate = 1.0`: every episode hit the ball but gently (outgoing x speed parked at ~2.1 m/s) and was force-ended as `bad_hit` at the hit-to-return timeout. Three coupled causes were addressed:

- Speed targets were saturating below a returnable hit. `quality_hit.target_outgoing_speed` was `2.0` (saturated, so no incentive past 2 m/s), `post_hit_outgoing.target_speed` was `2.0`, and `racket_approach` was a Gaussian peaked at `1.0 m/s` that actively penalized faster swings. These are now `5.5`, `5.0`, and a monotonic approach reward (`target_vel = 3.0`). The score clamps still cap each term's magnitude, so the saturation point moved up without changing peak reward.
- Return was geometrically unreachable inside the old `0.40 s` hit-to-return timeout (the racket is ~1.5 m from the net; even at the rewarded 2 m/s the ball cannot cross the net before the episode is force-ended). The timeout is now `0.90 s`.
- Contact location was unrewarded: the racket reference point was the `Link_yb_paddle` body origin (~0.045 m below the blade center) and the `0.25 m` hit gate was larger than the whole paddle, so a handle/edge contact scored the same as a center contact. The reference point is now the blade center (`RACKET_OFFSET_Z = 0.045`, used by the rewards, observations, and the hit/near-miss distance gate alike), `quality_hit` is multiplied by a centeredness factor, and the hit distance gate was tightened to `0.15 m` (still ≥ the blade's ~0.13 m center-to-corner reach, so real contacts are not lost — the contact-force check already requires physical contact).

`racket_face_target` and `racket_normal_swing` were re-enabled in the same retune; they had been left at weight `0.0` in the converged run, which removed the only pre-hit incentive to orient and accelerate the paddle face toward the target.

2026-06-16 retune (center contact, deeper placement, flatter return). The `2026-06-15_17-24-45` run reached a usable valid-return rate but showed three deployment risks: mean hit distance was about `7.9 cm`, landing placement score implied returns still landing roughly half a meter from table center, and hit-time z velocity averaged about `2.6 m/s` with post-hit max height around `1.44 m`. The reward now makes center contact, actual landing placement, and predicted landing placement sharper, while reducing the broad post-hit dense terms that were dominating `reward/total_mean`. It also logs `episode/hit_center_offset_mean`, `episode/landing_x_mean`, and `episode/landing_y_mean` so the next run can be judged without relying only on video.

2026-06-16 follow-up after the 180k→260k resume. The direct `post_hit_lob_penalty` reduced hit-time up speed only modestly (`~2.53 -> ~2.38 m/s`) and post-hit max height only modestly (`~1.43 -> ~1.40 m`), while deterministic eval success dropped versus the previous best (`0.689 -> 0.588`). It was therefore disabled. The flatter-return preference is now expressed by target landing placement and by giving no extra quality-hit bonus to excessive upward speed, not by a dense negative lob penalty. A separate `hit_centered` event term was added because `quality_hit` alone did not improve center contact; the 260k tail still had `hit_center_offset_mean ~= 7.6 cm`.

2026-06-16 reward-boundary cleanup. `quality_hit` was simplified to avoid overlapping with center-contact and landing-placement terms. It no longer multiplies by centeredness and no longer uses a fixed `target_up_speed`; it only ramps outgoing x speed over a viable band and gates out downward hits. `post_hit_outgoing` was disabled because it duplicated the same x-speed semantics after contact; `post_hit_landing_prediction` is now the dense post-hit term responsible for whether the predicted trajectory actually reaches the opponent-side target.

2026-06-16 pre-hit shaping cleanup. The four dense pre-hit shaping terms (`racket_ball_proximity`, `racket_approach`, `racket_face_target`, `racket_normal_swing`) were set to `0.0` by default. In the 180k→260k resume tail they were strongly negatively correlated with `episode/valid_return_rate` and `episode/hit_rate`, strongly positively correlated with `episode/miss_rate`, and positively correlated with worse `episode/hit_center_offset_mean`. This suggests they had become stale scaffolding after the policy learned contact. They remain in the config only for ablation or early-curriculum experiments.

2026-06-18 center-contact re-gate. Edge/blade contact persisted under the additive-only `hit_centered` term, and the earlier `quality_hit`-only multiplicative gate had already plateaued at `hit_center_offset_mean ≈ 7.6 cm` (see the 2026-06-16 notes). Root cause: the dominant success rewards — `valid_return` (25), `landing_placement` (25), `return_cross_net` (15) — were entirely center-blind, so an edge hit that happened to land in-court collected the same ~65 points as a center hit; gating only `quality_hit` (20) put too little reward mass behind center contact to move the policy. Fix: a shared per-hit centering gate `floor + (1 - floor) * exp(-center_sigma * offset^2)` (`_center_gate_factor` in `rewards.py`, reading the cached `_sac_hit_center_offset`, which persists from the hit step through the `valid_return` step) now multiplies **both** `quality_hit` and `landing_placement` (`center_sigma = 400`, `center_gate_floor = 0.15`). `valid_return` and `return_cross_net` are deliberately left ungated as the unconditional "ball was returned" anchor, so the converged policy cannot forget returning while it learns to center. Sigma is calibrated to the real blade radius (~7.5 cm, 15 cm diameter): `center_sigma = 400` gives a 1/e width of ~5 cm and suppresses a rim hit to factor ~0.24, whereas the inherited `sigma = 220` (1/e width ~6.7 cm, nearly the whole blade) left a rim hit at ~0.43 — nearly flat across the blade, the likely reason both the additive `hit_centered` and the earlier `quality_hit`-only gate failed to pull contact inward. The same `center_sigma = 400` was applied to the additive `hit_centered` term for consistency, and all three terms share `SAC_CENTER_SIGMA` / `SAC_CENTER_GATE_FLOOR` in `env_cfg.py`. Net effect per success: centering from the rim (~7.6 cm) to center raises the gated reward mass (`quality_hit` + `landing_placement`, max 45) from ~11 to ~45, against an ungated return anchor of ~42. The `0.15` floor keeps the gate a gradient rather than a cliff, which the converged-but-oscillating policy is sensitive to (see `docs/sac_catch_oscillation_diagnosis.md`). If `hit_center_offset_mean` still does not fall, the next lever is to extend the gate to `return_cross_net`/`valid_return`, raise `center_sigma` further, or lower the floor; if it falls but `valid_return_rate` drops, raise `center_gate_floor` (or check whether center contact is even kinematically reachable for the fixed serve).

## 6. Event Logic

Event tags:

```text
near_miss, hit, bad_hit, return, valid_return, miss
```

Event tracking is in `mdp/events.py` and runs every 20 ms.

Main thresholds:

- Hit distance threshold: `0.15 m` (from the paddle blade center).
- Near-miss threshold: `0.25 m`.
- Contact force threshold: `0.1`.
- Hit-to-return timeout: `0.90 s`.
- Table z: `0.76 m`.
- Net z: `0.9125 m`.
- Table half y: `0.7625 m`.
- Out after hit if `z < 0.45`, `abs(x) > 3.0`, or `abs(y) > 1.4`.

Definitions:

- `near_miss`: no hit yet and episode min racket-ball distance <= `0.25`.
- `hit`: contact sensor force > `0.1` and racket-ball distance < `0.15`; only first hit is tagged.
- `return`: after hit, ball crosses net x toward opponent side with sufficient x velocity and `z > net_z`.
- `valid_return`: after hit, ball is near table height, over opponent table x range, inside y bounds, and moving down.
- `bad_hit`: after hit, ball lands on own table, goes out/falls before valid return, or fails to cross the net within `0.90 s` after hit.
- `miss`: no hit and ball has passed robot x by margin or dropped below `z_min`.

Episode termination:

- `valid_return`
- `bad_hit`
- `miss`
- timeout at `2.5 s`
- NaN joint state
- actual joint position limit violation

## 7. Replay And Event Tables

Uniform replay stores transition tensors:

- `obs_actor`
- `obs_critic`
- `action`
- `reward`
- `next_obs_actor`
- `next_obs_critic`
- `done`
- `episode_id`
- `step_index`
- `event_mask`

Default training capacity:

- Uniform replay size: `1_000_000`.
- Event table size: `250_000` per event.
- Batch size: `4096`.

Replay is stored on CPU; sampled batches are moved to the training device.

Episode trace storage:

- During an episode, `EpisodeTraceBuffer` stores replay slot index and slot version per vector env.
- At episode end, final event info is used to extract event windows.
- Event tables store replay indices plus slot versions, so overwritten replay slots can be rejected later.

Event windows:

| Event | Window |
| --- | --- |
| `near_miss` | closest step - 20 to closest step |
| `miss` | closest step - 20 to closest step |
| `hit` | hit step - 20 to hit step + 10 |
| `return` | hit step - 20 to return or valid-return step |
| `valid_return` | hit step - 20 to valid-return step |
| `bad_hit` | hit step - 20 to bad-hit step |

Default stratified sampling ratios:

| Source | Ratio |
| --- | ---: |
| `uniform` | `0.40` |
| `near_miss` | `0.25` |
| `hit` | `0.20` |
| `return` | `0.05` |
| `valid_return` | `0.05` |
| `miss` | `0.025` |
| `bad_hit` | `0.025` |

If an event table is empty or contains invalid overwritten slots, the missing portion falls back to uniform replay.

## 8. SAC Algorithm

Implementation: custom PyTorch SAC in `sac.py`.

Actor:

- Tanh-squashed Gaussian policy.
- Network outputs mean and log standard deviation.
- `log_std` is clamped to `[-5, 2]`.
- Stochastic action uses reparameterized sampling.
- Deterministic playback uses `tanh(mean)`.

Critic:

- Twin Q critics: `critic1`, `critic2`.
- Target critics: `target_critic1`, `target_critic2`.
- Critic input is `[obs_critic, action]`.
- Actor update uses the minimum of the two critics.

Core update:

```python
next_action, next_log_prob = actor.sample(next_obs_actor)
target_q = min(target_q1(next_obs_critic, next_action),
               target_q2(next_obs_critic, next_action))
backup = reward + (1 - done) * gamma * (target_q - alpha * next_log_prob)

critic_loss = mse(q1(obs_critic, action), backup) \
            + mse(q2(obs_critic, action), backup)

new_action, log_prob = actor.sample(obs_actor)
actor_loss = (alpha * log_prob - min(q1(obs_critic, new_action),
                                     q2(obs_critic, new_action))).mean()

alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
```

Default hyperparameters:

| Parameter | Value |
| --- | ---: |
| actor hidden dims | `[512, 256, 128]` |
| critic hidden dims | `[512, 256, 128]` |
| activation | `elu` |
| actor lr | `3e-4` |
| critic lr | `3e-4` |
| alpha lr | `3e-4` |
| gamma | `0.98` |
| tau | `0.005` |
| initial alpha | `0.02` |
| min alpha | `0.02` |
| target entropy | `-action_dim = -7` |

The `min_alpha` floor was added to avoid entropy collapsing too early while rare events are still sparse.

## 9. Training Loop

Typical command:

```bash
python scripts/sac_table_tennis/train.py \
  --headless \
  --task A1-TableTennis-SAC-Catch \
  --num_envs 64 \
  --max_updates 10000 \
  --log_interval 20 \
  --checkpoint_interval 1000
```

Important CLI defaults:

| Argument | Default |
| --- | ---: |
| `--seed` | `1` |
| `--max_updates` | `30000` |
| `--start_steps` | `20000` transitions |
| `--batch_size` | `4096` |
| `--replay_size` | `1000000` |
| `--event_table_size` | `250000` |
| `--sampler` | `stratified` |
| `--updates_per_step` | `4` |
| `--log_interval` | `100` |
| `--checkpoint_interval` | `1000` |

Loop behavior:

1. Reset vectorized env.
2. Use random normalized actions until `global_transitions >= start_steps`.
3. Then sample actions from the stochastic SAC actor.
4. Step Isaac env once.
5. Store transitions into uniform replay.
6. If any env finished, finalize its episode trace and add event windows to event tables.
7. Once replay and warmup are sufficient, sample a batch and run SAC updates.
8. Log losses, replay size, event-table sizes, batch composition, and episode metrics.
9. Save checkpoints periodically and at the end.

`max_updates` counts gradient updates, not environment steps. `train/transitions` increases roughly linearly by `num_envs` every env step; this is expected.

## 10. TensorBoard Cards

The SAC TensorBoard output is split by purpose. The card count is intentional: `episode/*` measures finished-episode outcomes, `event_table/*` measures replay indexing, `batch/*` measures sampler composition, `reward_terms/*` measures shaping and penalties, and `loss/*` measures SAC optimization health.

Loss cards:

- `loss/critic_loss`: twin critic Bellman MSE sum.
- `loss/actor_loss`: entropy-regularized actor objective.
- `loss/actor_grad_norm`: actor gradient L2 norm measured just before the optimizer step.
- `loss/critic_grad_norm`: combined twin-critic gradient L2 norm measured just before the optimizer step.
- `loss/alpha_loss`: temperature tuning loss.
- `loss/alpha`: current entropy temperature.
- `loss/q1_mean`, `loss/q2_mean`: average Q values on sampled batch.

Replay/train cards:

- `replay/size`: current uniform replay size.
- `train/transitions`: total environment transitions collected.
- `batch/uniform`: number of sampled transitions from uniform replay or fallback.
- `batch/near_miss`, `batch/hit`, `batch/return`, `batch/valid_return`, `batch/miss`, `batch/bad_hit`: sampled transitions from each event table.

Event table cards:

These are not success rates. They are the current number of valid replay slots indexed by each stratified event table. A nonzero `event_table/hit` means the replay has hit-window samples available for oversampling, even if the current policy's recent `episode/hit_rate` is low.

- `event_table/near_miss`: valid replay slots currently indexed by the near-miss table.
- `event_table/hit`: valid replay slots currently indexed by the hit table.
- `event_table/return`: valid replay slots currently indexed by the return table.
- `event_table/valid_return`: valid replay slots currently indexed by the valid-return table.
- `event_table/miss`: valid replay slots currently indexed by the miss table.
- `event_table/bad_hit`: valid replay slots currently indexed by the bad-hit table.

Episode cards:

These summarize episodes that ended inside the current logging window. They are not per-step rewards.

- `episode/count`: number of episodes summarized in the current logging window.
- `episode/near_miss_rate`: fraction of finished episodes with near miss.
- `episode/hit_rate`: fraction of finished episodes with hit.
- `episode/return_rate`: fraction of finished episodes where a hit ball crossed the net.
- `episode/valid_return_rate`: fraction of finished episodes where the ball landed on opponent table.
- `episode/miss_rate`: fraction of finished episodes with miss.
- `episode/bad_hit_rate`: fraction of finished episodes with own-table/out/fall after hit.
- `episode/min_dist_mean`: mean minimum racket-ball distance.
- `episode/landing_x_mean`: mean final recorded landing x for episodes with a valid/bad landing.
- `episode/landing_y_mean`: mean final recorded landing y for episodes with a valid/bad landing.
- `episode/hit_center_offset_mean`: mean in-plane ball offset from the paddle blade center at first hit.
- `episode/hit_outgoing_speed_mean`: outgoing x speed measured at first hit.
- `episode/hit_up_speed_mean`: z velocity measured at first hit.
- `episode/post_hit_max_outgoing_speed_mean`: max outgoing speed after hit before final outcome.
- `episode/post_hit_max_height_mean`: max ball height after hit before final outcome.

Reward cards:

These are weighted reward terms from IsaacLab's `RewardManager`, averaged over environment steps in the current logging window. They are useful for checking whether a shaping term is active and whether a penalty dominates the sparse event rewards.

- `reward/total_mean`: total weighted reward averaged over env steps.
- `reward_terms/racket_ball_proximity`: disabled pre-hit distance-shaping ablation hook.
- `reward_terms/racket_approach`: disabled pre-hit closing-speed ablation hook.
- `reward_terms/racket_face_target`: disabled pre-hit face-alignment ablation hook.
- `reward_terms/racket_normal_swing`: disabled pre-hit normal-swing ablation hook.
- `reward_terms/hit`: sparse contact reward.
- `reward_terms/quality_hit`: first-contact basic outgoing-quality reward from outgoing speed plus a shallow non-downward gate.
- `reward_terms/hit_centered`: first-contact centeredness reward from cached in-plane ball-to-blade-center offset.
- `reward_terms/return_cross_net`: sparse crossed-net reward after hit.
- `reward_terms/valid_return`: sparse valid-return reward.
- `reward_terms/landing_placement`: valid-return placement score at the actual landing point.
- `reward_terms/miss`: no-hit miss penalty.
- `reward_terms/bad_hit`: bad post-hit outcome penalty.
- `reward_terms/post_hit_outgoing`: disabled outgoing-velocity ablation hook.
- `reward_terms/post_hit_net_progress`: disabled post-hit net-progress hook.
- `reward_terms/post_hit_net_clearance`: predicted net-clearance shaping.
- `reward_terms/post_hit_landing_prediction`: predicted opponent-table center placement shaping.
- `reward_terms/post_hit_lob_penalty`: disabled high-arc penalty ablation hook.
- `reward_terms/action_rate`: action-change penalty.
- `reward_terms/joint_acc`: joint-acceleration penalty.
- `reward_terms/joint_limit`: joint-limit margin barrier penalty.

Primary cards to watch during reward tuning:

- `episode/hit_rate`
- `episode/miss_rate`
- `episode/bad_hit_rate`
- `episode/return_rate`
- `episode/min_dist_mean`
- `episode/hit_center_offset_mean`
- `episode/landing_x_mean`
- `episode/hit_up_speed_mean`
- `episode/post_hit_max_height_mean`
- `reward_terms/joint_limit`
- `reward_terms/hit`
- `reward_terms/quality_hit`
- `reward_terms/hit_centered`
- `reward_terms/landing_placement`
- `reward_terms/post_hit_landing_prediction`
- `reward_terms/post_hit_lob_penalty`
- `reward_terms/miss`

## 11. Latency Model

Current explicit latency model:

- No camera/vision latency is simulated.
- No actor-only observation delay is injected.
- KF-style noise is injected only into the actor-visible `estimated_hit_command`; no ball-position-history noise/dropout is injected.
- No command-transport latency is simulated.

Current timing effects:

- The policy only updates every 20 ms.
- The selected target is held across 4 physics substeps.
- Action smoothing and target rate limiting add target-side inertia, but this is not a measured communication delay.
- `ball_pos_history` provides recent samples at 20 ms intervals, but the newest sample is current state.

Real-machine implication:

- The current observation is intentionally closer to deployable signals than the previous predicted-intercept observation.
- It still assumes the current ball position history is available with no delay.
- The actor-visible `estimated_hit_command` is the training proxy for the KF/trajectory-prediction output that deployment can provide.
- The next latency-aware stage should add actor-only ball observation delay/noise/dropout while keeping critic observations true in simulation.

## 12. Evaluation

Playback command:

```bash
python scripts/sac_table_tennis/play.py \
  --headless \
  --task A1-TableTennis-SAC-Catch \
  --checkpoint logs/sac_table_tennis/A1-TableTennis-SAC-Catch/<run>/checkpoints/agent_final.pt \
  --num_envs 1 \
  --episodes 20
```

Default playback uses deterministic actor actions. Add `--stochastic` to sample from the actor distribution.

For GUI inspection, omit `--headless` and add `--real_time` or `--sleep_per_step 0.02`; otherwise Isaac may run faster than wall time and finish episodes too quickly to inspect.

For reset-pose and ball-trajectory inspection without a learned policy, use `--zero_action`; in that mode `--checkpoint` is not required and the script sends all-zero normalized actions.

Playback writes a per-step simulated joint CSV by default under:

```text
logs/sac_table_tennis/sim_logs/
```

The file stem is derived from the task id, checkpoint run folder, and checkpoint name, for example `A1-TableTennis-SAC-Catch__2026-06-10_17-13-09__agent_0010000.csv`. Use `--sim_log_dir <dir>` to redirect it or `--no_sim_log` to disable it. The CSV contains one row per env per play step with `global_step`, `time_s`, `env_id`, `episode`, `episode_step`, done flags, normalized `action_*`, processed `q_target_*`, simulated `q_sim_*`, `qd_sim_*`, and `tau_sim_*` for `joint_yb_1..7`. `q_target_*` is the `JointDeltaTargetAction` position target after scaling, smoothing, rate limiting, and clamping; `tau_sim_*` is read from IsaacLab `robot.data.applied_torque`.

Visualize a recorded joint CSV with `python scripts/sac_table_tennis/plot_sim_joint_log.py <csv> --env_id 0 --episode 1`. If no CSV path is provided, the script uses the newest file in `logs/sac_table_tennis/sim_logs/`. The plot layout is one row per joint and three columns: target-vs-sim joint position, simulated joint velocity, and simulated torque.

Printed evaluation output includes:

- episode index
- decoded event list
- landing x/y
- min racket-ball distance
- hit-center offset
- hit-time outgoing and upward speed
- post-hit max height
- final event counts
- metric means

## 13. Current Test Coverage

Unit tests live in `tests/test_sac_table_tennis_pipeline.py`.

Covered:

- joint-delta clamp and rate-limit helper
- replay insertion, overwrite, sampling shapes, slot-version validity
- event window extraction and trace finalization
- stratified sampler fallback and event-table sampling
- HER landing-y reward recomputation scaffold
- SAC update on synthetic replay batch
- SAC alpha lower bound

Common verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_sac_table_tennis_pipeline.py -q
python -m compileall scripts/sac_table_tennis source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac tests/test_sac_table_tennis_pipeline.py
```

Isaac smoke training should be run inside the Isaac conda/runtime environment.

## 14. Roadmap

Near-term focus:

1. Keep the current single fixed-ball setup until contact and return are stable.
2. Improve hit-to-return behavior before adding HER.
3. Watch post-hit diagnostics to see whether the ball has enough outgoing speed and height.
4. Add a small serve curriculum only after fixed-ball hit/return becomes reliable.
5. Tune post-hit shaping or action constraints only if hit exists but return stays near zero.

Serve curriculum recommendation:

- Stage 0: single fixed incoming ball.
- Stage 1: small y/z/v randomization around the fixed ball after stable hit/return.
- Stage 2: expand x/y/z/v ranges after the policy keeps acceptable return rate in Stage 1.
- Stage 3: add target-y and HER after nonzero return/landing events are present.

Deferred stages:

- HER with `target_y` only after hit/return episodes are nonzero.
- Actor-only delayed/noisy ball-position history.
- Serve randomization and curriculum.
- Spin.
- More realistic command and vision latency.
- Hardware-oriented policy export and safety filtering.
