# Ace SAC Paper Comparison And A1 Optimization Notes

Last updated: 2026-06-18

Companion to `docs/sac_table_tennis_architecture.md`. This note summarizes the actionable parts of the Ace table-tennis SAC paper and supplementary material for the current `A1-TableTennis-SAC-Catch` pipeline. It is intentionally scoped to the current A1 arm constraints:

- Train one stable catch/return policy first.
- Ignore ball spin for now.
- Keep the deployed policy at about 50 Hz and interpolate to the real arm's 100 Hz command loop.
- Do not reproduce Ace's 32 ms FAOC/MPC plus 1 kHz low-level trajectory stack.
- Focus on observation design, asymmetric actor-critic, replay/event sampling, and reward cleanup.

---

## 1. Key Takeaway

Ace's reward is not powerful by itself. It works because the surrounding system removes many hard problems from the reward:

- The policy outputs a high-level action that is converted by FAOC/MPC into a smooth, feasible 32 ms trajectory segment.
- The robot executes at 1 kHz with a dynamics model trained for high-frequency transfer.
- The actor observes rich ball history with timestamps and a planned open-loop terminal robot/end-effector state.
- The critic receives cleaner ground-truth observations than the actor.
- The actor is regularized by an auxiliary reconstruction loss from noisy actor observations toward clean critic observations.
- Event tables oversample rare useful transitions.
- Training starts from realistic human/synthetic ball-state distributions rather than one fixed ball.
- Real-game deployment uses multiple policies, a policy sampler, and a prepare/reset policy.

For this repository, the useful lesson is not "make reward more complex". The useful lesson is:

1. Give the actor enough deployable state to understand racket geometry and timing.
2. Let the critic and auxiliary losses use privileged simulator truth.
3. Shift replay emphasis from "make contact" to "make clean valid returns" once contact is learned.
4. Keep reward terms event/outcome based and remove stale dense pre-hit scaffolding.

---

## 2. Paper Details Worth Keeping

### Observation

Ace policy state contains:

- Ball position history: `N=15` 3D samples.
- Ball spin history: `N=15` 3D samples.
- Timestamp/delay history for position and spin measurements.
- Robot state from the terminal point of the previous 32 ms trajectory segment:
  - joint position, velocity, acceleration,
  - end-effector position, compact orientation, linear velocity, angular velocity.
- Skill conditioning:
  - desired landing `y`,
  - position reward weight,
  - spin reward weight.

For this repository:

- Spin history and spin-conditioned rewards are intentionally out of scope.
- Skill conditioning is out of scope until one single return policy is stable.
- The useful part is the robot/end-effector state: Ace does not ask the policy to infer all racket geometry from joint position alone.

### Asymmetric Actor-Critic

The official Ace pseudocode uses different observations for actor and critic:

- Actor: noisy sensor estimate, deployable at runtime.
- Critic: ground-truth state, training-only.
- Replay stores `obs_actor`, `obs_critic`, `next_obs_actor`, and `next_obs_critic`.
- SAC actor actions are sampled from `obs_actor`, but Q-values are evaluated with `obs_critic`.
- The policy also has an auxiliary reconstruction loss that predicts clean/ground-truth observations from actor observations.

This repository already implements the main observation split. It does not yet implement the auxiliary reconstruction loss.

### Reward

Ace reward terms are mostly outcome/event terms:

- miss: reward inversely related to racket-ball distance at miss,
- hit without return: includes racket angular velocity penalty and distance-to-table terms,
- valid return: landing position, landing speed, net-crossing height, spin, bounce/robustness terms depending on the selected policy.

The important pattern is that reward is attached to meaningful events: miss, hit, return, landing, and quality at those events. It is not dominated by long pre-hit dense shaping.

For the current no-spin single-policy stage, the paper-compatible subset is:

- miss distance / near-miss information for exploration,
- hit event,
- hit quality,
- valid return,
- landing placement,
- racket angular velocity penalty at hit,
- smoothness/limit/effort regularization.

---

## 3. Current Repository State

Main paths:

- Task config: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/env_cfg.py`
- Observations: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/observations.py`
- Rewards: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/rewards.py`
- Events: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/events.py`
- Replay/event tables: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/replay.py`
- SAC implementation: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/sac.py`
- Training entrypoint: `scripts/sac_table_tennis/train.py`

Current actor observation:

- right-arm joint position,
- recent joint-position delta history,
- ball position history,
- noisy deployment-style hit command `[p_hit_x, p_hit_y, p_hit_z, tau]`,
- last action.

Current critic observation:

- all actor observations,
- clean joint velocity,
- clean racket position and normal,
- clean ball velocity,
- clean racket linear/angular velocity,
- clean ball-racket relative position/velocity,
- clean racket axes,
- clean ground-truth hit command.

Current reward is already close to the desired event/outcome structure:

- `hit`
- `quality_hit`
- `hit_centered`
- `return_cross_net`
- `valid_return`
- `landing_placement`
- `post_hit_net_clearance`
- `post_hit_landing_prediction`
- `miss`
- `bad_hit`
- `action_rate`
- `joint_acc`
- `joint_jerk`
- `joint_limit`
- `joint_effort_margin`

The pre-hit shaping terms are disabled by default, which is consistent with the current stage. Prior runs showed they correlated with misses after the policy already learned contact.

---

## 4. Main Gaps Versus Ace

### Gap A: Actor Has Too Little Deployable Robot Geometry

Ace gives the policy an end-effector state from the currently executing trajectory segment. The current actor does not directly observe racket pose, racket normal, or racket target state. It must infer FK and racket geometry from 7 joint positions and recent deltas.

This is likely expensive for a small MLP and directly relevant to the current failure signature: the policy can return, but hit center offset remains around the paddle edge region.

Recommended actor additions, all deployable from real robot encoder FK:

- `racket_pos`
- `racket_normal` or full `racket_axes`
- processed/current joint target if available in deployment,
- `q_target - q` or previous processed target delta,
- optional `ball_pos_relative_to_racket` if computed from measured/predicted ball position and FK.

Keep simulator-only velocity and clean ground-truth hit command critic-only.

### Gap B: No Auxiliary Reconstruction Loss

Ace trains the policy with an auxiliary loss to reconstruct clean/ground-truth observations from actor observations. This helps the actor learn latent velocity, delay, timing, and hidden state from noisy observations.

Minimal version for this repo:

- Add an actor trunk/head split in `sac.py`.
- Keep the SAC action head unchanged.
- Add a reconstruction head trained only during actor update.
- Reconstruct a selected privileged target, not the entire 92D critic observation at first.

Suggested target vector:

- clean `groundtruth_hit_command`,
- clean `ball_vel`,
- clean `racket_vel`,
- clean `ball_pos_rel_racket`,
- clean `ball_vel_rel_racket`.

Use normalized components and a small loss coefficient. Start with `aux_coef = 0.05` or `0.10`, then inspect actor loss scale.

### Gap C: Event Replay Still Looks Contact-Heavy

Current default event-table ratios are suitable for early learning:

```text
uniform      0.40
near_miss    0.25
hit          0.20
return       0.05
valid_return 0.05
miss         0.025
bad_hit      0.025
```

Once `hit_rate` and `valid_return_rate` are nonzero and stable, this can overemphasize "touch the ball" transitions. The next stage should bias replay toward clean successful returns and the transition window that produces them.

Recommended next-stage ratios:

```text
uniform      0.40
near_miss    0.10
hit          0.10
return       0.10
valid_return 0.20
miss         0.05
bad_hit      0.05
```

If center contact remains poor, add a new event table:

- `centered_valid_return`: valid return with first-hit center offset below a threshold, for example `<= 0.04 m` or `<= 0.05 m`.

This is more targeted than increasing the scalar `hit_centered` reward again.

### Gap D: No Hit-Time Racket Angular Velocity Penalty

Ace explicitly penalizes racket angular velocity at hitting time. This is useful for this repo because edge/handle contact and wrist-flick exploitation are sim-to-real risks.

Recommended reward term:

- fires only on the first `hit` event step,
- reads cached or current racket angular velocity magnitude,
- penalizes above a tolerant threshold,
- keep small compared with `valid_return` and `landing_placement`.

Example behavior:

```text
score = clamp((|omega| - omega_free) / omega_band, 0, 1)^2
reward weight = -1.0 to -3.0 initially
```

This is preferable to restoring broad pre-hit face/proximity rewards, because it acts only at the actual event that matters.

### Gap E: Training Distribution Is Still Much Narrower Than Ace

Ace samples initial ball states from human and synthetic KDE distributions with OOD bandwidth control. The current A1 setup intentionally starts from a narrow fixed-middle-ball distribution.

Because the real A1 catchable region is only about `40 cm x 50 cm`, widening the ball distribution should be based on reachability, not copied from Ace.

Recommended process:

1. Map reachable `p_hit_y`, `p_hit_z`, and `tau` for the current ready pose and joint/effort limits.
2. Define a conservative reachable curriculum inside that map.
3. Expand only after deterministic eval success and center-contact metrics hold.
4. Keep out-of-reach shots out of the training distribution until there is a prepare/reset strategy.

---

## 5. Proposed Implementation Order

### Phase 1: Observation Upgrade

Goal: give actor deployable racket geometry.

Changes:

- Add actor terms for `racket_pos` and `racket_normal` or `racket_axes`.
- Consider adding `ball_pos_relative_to_racket` if it can be computed from deployable FK and actor-visible ball estimate.
- Update `docs/sac_table_tennis_architecture.md` actor/critic dimensions after the code change.

Validation:

- Confirm actor observation dimension changes as expected.
- Start a fresh run or use actor-only warm start only if dimensions allow.
- Watch `episode/hit_center_offset_mean`, `eval/valid_return_rate`, and `episode/bad_hit_rate`.

### Phase 2: Auxiliary Reconstruction Loss

Goal: help actor infer clean timing and relative geometry from deployable/noisy observations.

Changes:

- Modify `TanhGaussianActor` to expose a trunk embedding.
- Add a reconstruction head.
- Store/derive selected privileged reconstruction targets from `obs_critic`.
- Add `aux_recon_loss` to actor update with a small coefficient.
- Log `loss/aux_recon_loss`.

Validation:

- Check that SAC actor loss scale is not overwhelmed.
- Compare valid return and center offset against the no-aux baseline.
- If center improves but valid return drops, lower `aux_coef`.

### Phase 3: Replay Ratio Shift

Goal: stop overtraining contact once return exists.

Changes:

- Add configurable event ratios in `train.py` or `replay.py`.
- Run success-heavy ratios for resumed or fresh training.
- Optionally add `centered_valid_return` event table.

Validation:

- Inspect `batch/*` TensorBoard cards to verify sampler composition.
- Compare `eval/valid_return_rate`, `episode/hit_center_offset_mean`, and `episode/landing_x_mean`.

### Phase 4: Hit-Time Angular Velocity Penalty

Goal: reduce wrist-flick / edge-contact solutions that may not transfer.

Changes:

- Cache racket angular velocity at first hit in `events.py`, or compute it in the reward term on hit step.
- Add a small `hit_racket_ang_vel` penalty in `RewardsCfg`.
- Keep it event-only, not dense pre-hit shaping.

Validation:

- Watch `reward_terms/hit_racket_ang_vel`, `episode/hit_center_offset_mean`, and `episode/hit_up_speed_mean`.
- If valid returns drop sharply, reduce weight first rather than adding dense positive shaping.

### Phase 5: Reachability-Based Curriculum

Goal: expand from fixed middle ball toward the real catchable region.

Changes:

- Add a small sweep/diagnostic script that samples candidate hit points and checks IK/FK/limit margins.
- Expand `SAC_FIXED_MIDDLE_BALL` ranges only inside reachable `p_hit_y/z/tau`.
- Gate expansion on deterministic eval metrics.

Validation:

- Do not judge by aggregate valid return alone.
- Track success by `p_hit_y/z` bins and hit-center offset bins.

---

## 6. What Not To Copy Yet

Do not implement these in the current phase:

- Full spin estimation or spin-aware ball dynamics.
- Multiple reward-conditioned policies.
- Policy sampler or strategy layer.
- Prepare policy for opponent-game tactics.
- FAOC/MPC at 1 kHz.
- Full Ace-scale human/KDE ball distribution.

Those are useful only after the single-policy A1 catch/return behavior is stable and the reachable workspace has been measured.

---

## 7. Practical Success Criteria

For the next meaningful run, prioritize these metrics:

- `eval/valid_return_rate`: should not regress materially.
- `episode/hit_center_offset_mean`: should move below the current edge-contact plateau.
- `episode/landing_x_mean` and `episode/landing_y_mean`: should remain near the target region.
- `episode/bad_hit_rate`: should not spike during center-contact improvements.
- `reward_terms/post_hit_landing_prediction`: should not dominate the total reward again.
- New, if added:
  - `loss/aux_recon_loss`,
  - `episode/centered_valid_return_rate`,
  - `reward_terms/hit_racket_ang_vel`.

A good short-term target is not full-table coverage. A realistic next target is:

- keep deterministic valid return around or above the current best run level,
- reduce mean hit center offset from roughly `7-8 cm` toward `4-5 cm`,
- then expand the launch/hit-point range inside the measured A1 reachable region.

