# A1 Table Tennis SAC Training Architecture

Last updated: 2026-06-10

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
- No asymmetric/noisy actor yet. Actor and critic observations are currently identical.
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
- Robot base x: `-1.7`.
- Robot contact/evaluation x: `-1.5`.
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
    "vz_range": (0.0, 0.0),
}
```

The ball is reset once per episode by `launch_ball`. It now starts near the opponent-side table edge instead of spawning close to the middle of the table, because the previous `x=0.35, z=1.10, vx=-2.4, vz=0.2` setup tended to land on the robot half too early and could show repeated bounces on the robot side before the arm interacted with it.

Robot reset:

- Every episode reset now writes the fixed-base robot root back to its default root state.
- The locked lift joint is initialized through `SAC_READY_LIFT_POS = -0.17`.
- This is the midpoint between the original SAC/A1 default lift `-0.28` and the higher tested lift `-0.06`. Isaac reports the lift joint's legal range as `[-0.800, -0.050]`; an FK-only solve for exactly `1.16 m` paddle height with the current 7DOF ready pose would require about `+0.126`, which is outside the joint limit.
- The controlled right arm is reset to `SAC_READY_JOINT_POS`.
- Current user-provided backhand ready pose:
  - `[1.533406, -0.523925, 1.60474, -1.183103, -0.007649, 1.042375, -1.845288]`
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

The observation was simplified for real-machine feasibility. The current policy observation does not include privileged predicted intercept features or ball velocity in racket frame.

Current groups:

- `policy`: actor observation.
- `critic`: critic observation.
- These two groups currently have the same terms and same dimension.

Current dimension:

```text
7 + 7 + 3 + 3 + 4*3 + 7 = 39
```

Terms:

| Term | Dim | Meaning |
| --- | ---: | --- |
| `joint_pos` | 7 | Right-arm joint position relative term from IsaacLab MDP. |
| `joint_vel` | 7 | Right-arm joint velocity relative term from IsaacLab MDP. |
| `racket_pos` | 3 | Racket center position in env/world frame, minus env origin. |
| `racket_normal` | 3 | Racket local +Y normal rotated into world frame. |
| `ball_pos_history` | 12 | `K=4` ball positions in env/world frame, newest sample first. |
| `last_action` | 7 | Previous normalized action from the action manager. |

Important details:

- `ball_pos_history` is sampled at policy-step cadence, so adjacent samples are 20 ms apart.
- The newest `ball_pos_history` entry is the current ball position. The history is not an explicit delay.
- Ball velocity can be inferred by the network from position history; no explicit `ball_vel` observation is used now.
- No `predicted_hit_point` or `time_to_intercept` observation is used now.
- No racket velocity observation is used now.
- No joint position history is used now.
- No observation noise, dropout, or latency is injected now.

The actor/critic split is still kept in the storage and SAC API:

- Actor input: `obs_actor`.
- Critic input: `obs_critic`.
- Current stage: both are 39D and identical.
- Future asymmetric stage: actor can receive delayed/noisy ball history while critic keeps true simulator state.

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

Termination currently uses actual joint position limit violation with margin `0.005`, not the action term's raw target violation flag.

## 5. Reward

IsaacLab reward terms are configured in `RewardsCfg`. In practice, the event weights were scaled by `1 / 0.02 = 50` so that after 20 ms reward integration they behave like the intended sparse bonuses:

- hit: about `+1`.
- return/cross-net: about `+4`.
- valid return: about `+8`.
- miss: about `-1`.
- bad hit: about `-2`.

Current reward terms:

| Term | Weight | Effective one-step scale at 20 ms | Meaning |
| --- | ---: | ---: | --- |
| `racket_ball_proximity` | `0.4` | up to `0.008` | Dense `exp(-12 * distance^2)` between racket and ball. |
| `racket_approach` | `0.1` | up to `0.002` | Reward positive racket velocity toward the ball, centered near `1.0 m/s`. |
| `racket_forward_push` | `2.0` | up to `0.04` per step | Before hit, when the ball is near the racket, encourages racket velocity toward the opponent side. |
| `hit` | `50.0` | `+1.0` | Step event reward when first racket contact is detected. Lowered so passive contact is not enough. |
| `return_cross_net` | `200.0` | `+4.0` | Step event reward when a hit ball crosses the net toward the opponent side above net height. |
| `valid_return` | `400.0` | `+8.0` | Step event reward when a hit ball lands on the opponent table. |
| `miss` | `-50.0` | `-1.0` | Penalty when the ball passes the robot or falls before any hit. |
| `bad_hit` | `-100.0` | `-2.0` | Penalty when a hit ball lands on own table or goes out/falls after hit. |
| `post_hit_outgoing` | `3.0` | up to `0.06` per step | After hit, encourages positive outgoing x velocity toward the opponent side. |
| `post_hit_lift` | `1.0` | up to `0.02` per step | After hit and before return, encourages upward velocity. |
| `action_rate` | `-0.02` | regularizer | Penalizes action changes. |
| `joint_acc` | `-1e-6` | regularizer | Penalizes joint acceleration. |
| `joint_limit` | `-0.5` | regularizer | Penalizes proximity/violation of joint position limits. |

Pre-hit forward-push shaping was added after visual inspection showed the policy could get contact without a meaningful push. Post-hit shaping was added because training reached some `hit` episodes but did not yet produce `valid_return`. These terms try to make the first hits send the ball away from the robot and high enough to clear the net before the sparse `return` events become common.

## 6. Event Logic

Event tags:

```text
near_miss, hit, bad_hit, return, valid_return, miss
```

Event tracking is in `mdp/events.py` and runs every 20 ms.

Main thresholds:

- Hit distance threshold: `0.25 m`.
- Near-miss threshold: `0.25 m`.
- Contact force threshold: `0.1`.
- Hit-to-return timeout: `0.40 s`.
- Table z: `0.76 m`.
- Net z: `0.9125 m`.
- Table half y: `0.7625 m`.
- Out after hit if `z < 0.45`, `abs(x) > 3.0`, or `abs(y) > 1.4`.

Definitions:

- `near_miss`: no hit yet and episode min racket-ball distance <= `0.25`.
- `hit`: contact sensor force > `0.1` and racket-ball distance < `0.25`; only first hit is tagged.
- `return`: after hit, ball crosses net x toward opponent side with sufficient x velocity and `z > net_z`.
- `valid_return`: after hit, ball is near table height, over opponent table x range, inside y bounds, and moving down.
- `bad_hit`: after hit, ball lands on own table, goes out/falls before valid return, or fails to cross the net within `0.40 s` after hit.
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
| min alpha | `0.005` |
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
| `--max_updates` | `10000` |
| `--start_steps` | `20000` transitions |
| `--batch_size` | `4096` |
| `--replay_size` | `1000000` |
| `--event_table_size` | `250000` |
| `--sampler` | `stratified` |
| `--updates_per_step` | `1` |
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

Loss cards:

- `loss/critic_loss`: twin critic Bellman MSE sum.
- `loss/actor_loss`: entropy-regularized actor objective.
- `loss/alpha_loss`: temperature tuning loss.
- `loss/alpha`: current entropy temperature.
- `loss/q1_mean`, `loss/q2_mean`: average Q values on sampled batch.

Replay/train cards:

- `replay/size`: current uniform replay size.
- `train/transitions`: total environment transitions collected.
- `batch/uniform`: number of sampled transitions from uniform replay or fallback.
- `batch/near_miss`, `batch/hit`, `batch/return`, `batch/valid_return`, `batch/miss`, `batch/bad_hit`: sampled transitions from each event table.

Event table cards:

- `event_table/near_miss`: valid replay slots currently indexed by the near-miss table.
- `event_table/hit`: valid replay slots currently indexed by the hit table.
- `event_table/return`: valid replay slots currently indexed by the return table.
- `event_table/valid_return`: valid replay slots currently indexed by the valid-return table.
- `event_table/miss`: valid replay slots currently indexed by the miss table.
- `event_table/bad_hit`: valid replay slots currently indexed by the bad-hit table.

Episode cards:

- `episode/count`: number of episodes summarized in the current logging window.
- `episode/near_miss_rate`: fraction of finished episodes with near miss.
- `episode/hit_rate`: fraction of finished episodes with hit.
- `episode/return_rate`: fraction of finished episodes where a hit ball crossed the net.
- `episode/valid_return_rate`: fraction of finished episodes where the ball landed on opponent table.
- `episode/miss_rate`: fraction of finished episodes with miss.
- `episode/bad_hit_rate`: fraction of finished episodes with own-table/out/fall after hit.
- `episode/min_dist_mean`: mean minimum racket-ball distance.
- `episode/hit_outgoing_speed_mean`: outgoing x speed measured at first hit.
- `episode/hit_up_speed_mean`: z velocity measured at first hit.
- `episode/post_hit_max_outgoing_speed_mean`: max outgoing speed after hit before final outcome.
- `episode/post_hit_max_height_mean`: max ball height after hit before final outcome.

## 11. Latency Model

Current explicit latency model:

- No camera/vision latency is simulated.
- No actor-only observation delay is injected.
- No observation noise/dropout is injected.
- No command-transport latency is simulated.

Current timing effects:

- The policy only updates every 20 ms.
- The selected target is held across 4 physics substeps.
- Action smoothing and target rate limiting add target-side inertia, but this is not a measured communication delay.
- `ball_pos_history` provides recent samples at 20 ms intervals, but the newest sample is current state.

Real-machine implication:

- The current observation is intentionally closer to deployable signals than the previous predicted-intercept observation.
- It still assumes the current ball position is available with no delay.
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

Printed evaluation output includes:

- episode index
- decoded event list
- landing y
- final event counts

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
- Actor/critic asymmetry with delayed/noisy actor ball observations.
- Serve randomization and curriculum.
- Spin.
- More realistic command and vision latency.
- Hardware-oriented policy export and safety filtering.
