# A1 Backhand — Fine-Tuning Progress (2026-06-08)

Working note for the next fine-tuning round. Companion to the design spec /
plan in `docs/superpowers/{specs,plans}/2026-06-04-tabletennis-backhand-env-imitation*.md`.

## Goal

Sim-to-real fine-tuning of the A1 right-arm **backhand** table-tennis return.
Imitation → exploration curriculum. DR scope is deliberately narrow: **serve
distribution + action delay + ball-obs delay only** (no PD/torque/obs noise, no
ball/table physics randomization).

Task: `A1-TableTennis-Backhand`. Env at
`source/unitree_rl_lab/.../robots/a1/backhand/env_cfg.py` (reuses forehand
Scene/Obs/Rewards/Terminations).

## Symptom

Stage-2 RL plateaus at **~79% miss**, `ball_hit ≈ 0`, `ball_return = 0`. The
arm reproduces the reference swing but rarely contacts the ball, and when it
does the contact is glancing (no return).

## What we found (in order)

### 1. Root cause = lateral mis-centering of the reference (FIXED)

Diagnosed via instrumented pure-reference rollout (`play_pure_ref.py
--max_trials`, zero residual, calibrated serves). The miss is **lateral, not
timing**: timing was correct (phase@closest-approach 0.452 vs hit_phase 0.4643),
x/z aligned, but median `dy(ball−racket) = +0.156 m`. The single static
reference parks the paddle at `y≈−0.106` while serves span `y∈[−0.17,+0.24]`.

**Fix applied:** baked `joint_yb_2 += 0.329 rad` into `backhand_ref.npz` (all 126
frames; `.orig` backup kept). Result: dy bias `0.156 → 0.020 m`, min_gap
`0.186 → 0.129 m`, timing/x/z preserved.

### 2. Re-centering = necessary but insufficient

Floor improved `0.79 → 0.75`, miss briefly touched 0.57, but it did **not**
sustain contact on its own.

### 3. Contact-gate bug (FIXED)

The reward functions `ball_hit` / `ball_hit_speed` / `ball_hit_direction` gated
on **instantaneous** force `net_forces[:,0]`, which misses the 1–2 substep
ping-pong impact. The working detector `track_ball_hit` (sets `ball_was_hit`)
uses **max-force-over-history** + `dist<0.25`.

**Fix applied (backhand-only, backward-compatible):** added `use_max_force:
bool = False` to the three reward funcs in `mdp/rewards.py`; backhand Stage-2
opts in (`use_max_force=True`, `proximity_threshold 0.15 → 0.25`), aligning the
reward gate to `track_ball_hit`. Forehand untouched (default `False`).

### 4. Gate-fix-only clean test = FAILED

One-variable re-launch from `model_3000.pt` with only the gate fix (anchor 0.8 /
residual 0.3 unchanged): miss reverted to **0.83**, contact rewards stayed at the
sparse 1-step floor (~0.0003), `return = 0`. The dense gradient is
`pose_tracking (0.103)` + `racket_ball_proximity (0.120)` — a **hover
attractor**: proximity is maximized by parking at the ball, and hitting makes the
ball leave (dist↑ → proximity↓), so it actively discourages follow-through.

### 5. Forehand comparison → structural root cause (KEY)

Same reward weights, **forehand returns balls at residual 0.05**. So neither
reward weights nor residual reach is the bottleneck. The distinguishing factor:

| | Forehand (works) | Backhand (stuck) |
|---|---|---|
| `match_ball_direction` | **True** | False |
| `motion_files` | **multiple** (left/middle/right + variants), per-serve matched | **single** static ref |
| residual_scale | 0.05 | 0.3 |
| pure-ref min_gap | passes through ball | 0.129 m |

The backhand's single static reference cannot place the paddle on the ball's
path across the ±0.2 m serve spread; the residual alone is expected to cover the
remaining 0.129 m, which is marginal.

## Current uncommitted state

Nothing committed this session. All staged/untracked:

- **`backhand_ref.npz`** — re-centered (`yb_2 += 0.329`); `.orig` backup kept.
- **`mdp/rewards.py`** — `use_max_force` param on the 3 contact rewards.
- **`robots/a1/backhand/env_cfg.py`** — Stage-2 opt-in to the gate fix.
- Entire `robots/a1/backhand/` dir is **untracked** (`??`) — backhand env was
  never committed; commit will be a feature add, not just edits.
- Other modified shared files present (`actions.py`, `events.py`,
  `observations.py`, `delayed_env.py`, `play_pure_ref.py`, `train.py`,
  `rsl_rl_ppo_cfg.py`) — review scope before committing.
- `scripts/rsl_rl/_fk_jac_probe.py` — temp diagnostic, removable.

> Commit protocol: paste full `~/.gitmessage`-format draft and get explicit
> confirmation **before** any `git commit`.

## Next-round options (not yet decided)

1. **Mirror the forehand design (recommended):** give the backhand multiple
   per-serve-matched references + `match_ball_direction=True`. Generate/augment
   backhand refs at several lateral positions so a reference always passes
   through the incoming ball. This addresses the structural root cause directly.
2. **Tame the hover attractor:** reshape/decay `racket_ball_proximity` so it
   stops competing with follow-through (forehand's "Fix A–K" campaign that
   backhand never got).
3. **Bump residual reach** 0.3 → 0.4 — comparison suggests this is *not* the
   bottleneck; low priority on its own.

The gate fix and re-centering stay in place as accepted improvements regardless
of which option is chosen.

## Ops / environment

- Conda env `isaac`; launch via
  `source /home/woan/miniforge3/etc/profile.d/conda.sh && conda activate isaac`
  (bare `bin/python` fails — `ModuleNotFoundError: isaacsim`).
- Launch: `python -u scripts/rsl_rl/train.py --task A1-TableTennis-Backhand
  --headless --num_envs 1024 --resume --load_run <ts> --checkpoint model_3000.pt`.
- Latest runs under `logs/rsl_rl/a1_tabletennis_backhand/`: gate-fix run
  `2026-06-08_10-44-44` (model_3000, model_3500). Warm-start base was
  `2026-06-05_15-08-05/model_3000.pt`.
- **Safety:** never `kill -9` Isaac Sim (corrupts nvidia_uvm → CUDA "unknown
  error"). SIGTERM/SIGINT only. Recovery: `sudo rmmod nvidia_uvm && sudo
  modprobe nvidia_uvm`.
- TensorBoard already serving this logdir on :6007. Training process currently
  exited; GPU free (~1.3 GB = TB only).

## Session 2026-06-08 PM — reward-reshape campaign (user chose option 2)

User decision: pursue **reward reshape** (option 2), keep the single reference —
*not* multi-lateral refs / serve curriculum. Goal restated: generalize so the
policy **returns** the ball across the serve spread, not just swings. Stage-1
imitation is done. All runs warm-start from the same Stage-1 base
`2026-06-05_15-08-05/model_3000.pt` (residual 0.05, swing intact) for a clean A/B/C.

### R1 — starve the hover, keep the swing anchor (FAILED, informative)

Change vs failed-run: demote `racket_ball_proximity` 0.60→0.10 **and**
`gate_pre_contact=True` (proximity pays 0 once `ball_was_hit`, killing the
follow-through penalty). Run `2026-06-08_15-08-03`, ~1450 iters past base.

TB trend (smoothed, iters 3000→4446):

| signal | failed baseline | R1 best (~3600) | R1 final (4446) | read |
|---|---|---|---|---|
| `ball_missed_paddle` | 0.813 | **0.663** | **0.838** | dipped then reverted *worse* |
| `racket_ball_proximity` | 0.102 | 0.016 | 0.017 | ✅ hover killed (lever worked) |
| `racket_face_target` | 0.055 | 0.055 | **0.079** | ❌ new contact-free attractor |
| `swing_timing` | ~0 | ~0 | ~0 | ❌ timed swing never re-executed |
| `ball_hit` | 0.0002 | 0.0008 | 0.0002 | ❌ no sustained contact |
| `ball_return`/`land_opponent` | 0 | 0 | 0 | ❌ zero returns |

**Verdict:** demoting proximity *did* break the hover (0.102→0.017) and miss
briefly improved to 0.66 — but with the hover gone there was **no dense gradient
pulling the paddle onto the ball's path**, so the policy drifted to gaming the
next contact-free term it could reach (`racket_face_target`, orientation-only)
and abandoned the timed swing. Miss reverted to 0.84. This is the fallback-ladder
**F2 case**: *miss stays high / swing not executing.* R1's proximity demotion +
gate are kept (correctly killed the hover); F2 adds the missing position gradient
on top.

### F2 — add the forward-looking ballistic-intercept gradient (RUNNING)

Change vs R1: add `racket_at_predicted_hit` (rewards.py:533) to
`BackhandStage2RewardsCfg` only (added as an instance attr in `__post_init__`;
`RewardManager._prepare_terms` iterates `cfg.__dict__`, so it registers; forehand
`RewardsCfg` has no such term → untouched). It ballistically predicts where the
ball crosses `x=ROBOT_X`, rewards the paddle being at `(pred_y, pred_z)` with
`urgency` ramping 0→1 as arrival nears — the dense "be where the ball *will* be"
signal R1 lacked. Backhand-specific params: `weight=1.0`, `z_offset=0.0`
(center-hit, not forehand under-swing), `robot_side=ROBOT_SIDE=-1` (so incoming
balls `vx<0` pass the validity mask), `sigma=20`, `urgency_window=0.5`.

Verified at startup: `Episode_Reward/racket_at_predicted_hit` logs ~0.018 (term
fires). Run `2026-06-08_15-45-36`, PID tracked in `/tmp/f2_train.pid`, ~1.25s/iter.

**Decision criteria at ~model_4500** (same point R1 had clearly reverted):
- `ball_missed_paddle` must drop **and sustain** (R1's failure was non-sustain)
- `racket_at_predicted_hit` must **rise** (policy using the new gradient)
- `ball_hit`/`swing_timing` must rise; `ball_return`/`land_opponent` must clear 0
- If hits rise but returns stay flat → next is F1 (raise `ball_hit_direction`
  0.5→1.5, wire `ball_toward_target_after_hit`). If still no contact / swing not
  executing → reconsider structural option 1 (multi-ref / `match_ball_direction`).

### F2 verdict — FAILED, same wall as R1 (and the decisive finding)

F2 trend (iters 3000→4523): `ball_missed_paddle` 0.765 → dipped 0.523 @3300 →
**reverted to 0.892** (worse than R1 0.797 and baseline 0.813).
`racket_at_predicted_hit` *was* optimized (0.010→0.029) — but miss got **worse**
while it rose, i.e. the policy maximized the intercept reward by **parking the
paddle at the predicted point without swinging through it** — the identical hover
pathology, relocated from "the ball now" to "where the ball will be."
`swing_timing ≈ 0` and `ball_return = 0` in all three runs.

**The decisive finding (checked the Stage-1 base `2026-06-05_15-08-05`):** even at
residual 0.05, Stage-1 has `pose_tracking` 0.33↑ (swing motion reproduced) but
`ball_hit = 0.0000` and `ball_missed_paddle` 0.77→0.83 the whole time.
**"Imitation done" = the swing *motion* is copied, NOT that it contacts/returns
the ball.** The reference passes ~0.13 m from the ball (`min_gap`) and never
connects; `swing_timing_reward` (pays only when paddle <0.6 m during phase
0.30–0.55) stays ~0 because swing and ball are never co-located.

## Session 2026-06-08 PM (cont.) — Multi-reference structural fix (chosen direction)

After the reward-reshape path was shown exhausted, user chose **option 1: forehand-style
multi-reference per-serve matching** (over the contact-gated bootstrap and the
diagnose-first alternatives). Plan: `~/.claude/plans/fuzzy-fluttering-starlight.md`.

**Mechanism (verified in `mdp/commands.py`):** `_assign_motion_by_ball` (270-288) predicts
ball-y at `robot_x` and assigns `motion_ids` 0=middle / 1=left(+y) / 2=right(−y); engages
only when `num_motions ≥ 3` (else random). Proven template = `x1/forehand`
(`motion_files=[middle,left,right]`, `match_ball_direction=True`, `ball_y_threshold=0.05`).
The current a1 forehand cfg pins a single middle ref (deterministic pure-imitation), but the
machinery + a1 lateral refs already exist.

**Implemented:**
- `create_backhand_variants.py` (new): loads the re-centered `backhand_ref.npz` as middle,
  writes `backhand_ref_{middle,left,right}.npz` by offsetting **only column 1 (`joint_yb_2`)**
  — middle +0.0, left +0.35, right −0.27 rad (gain ≈0.41 m/rad; bucket centers +0.145/−0.11 m).
  All within `yb_2` limit (left max +0.229 < +0.26). Real-demo swing dynamics preserved.
- `env_cfg.py`: `motion_files=[middle,left,right]`, `match_ball_direction=True`,
  `ball_y_threshold=0.05`; `residual_scale 0.3→0.05` (contact now on the imitation manifold,
  so the residual-0.3 parking pathology cannot recur); new `BackhandStage3RewardsCfg`
  (pose-dominant 1.5, returns at 2.0, contact-gate `use_max_force`/0.25 retained, R1/F2 hacks
  dropped). py_compile OK.

**Phase B gate (DONE, 200 trials, multi-ref selection active):**
- median `min_gap` = **0.109 m** (strict gate wanted < ~0.08 → fail, but informative);
  44.5% of serves < 0.10 m, 33.5% < 0.08 m (vs old single-ref ~0% contact in training).
- per-axis: `|dy|` median **0.042** (lateral — multi-ref fixed it), but `|dz|` median
  **0.065** is now the *largest* component (`|dx|` tiny 0.027). The limiter shifted from
  lateral to **serve arrival height (dz)** — a second structural axis lateral refs can't fix.
- **Decision (per plan contingency):** the residual gap is x/z-dominant, so bump
  `residual_scale 0.05 → 0.10` (plan cap) and proceed. Rationale the pure-ref gate
  *understates*: (1) residual fine-tunes lateral+height onto the ball; (2) `phase_speed`
  (0.85–1.15) lets the policy pick *which swing frame* contacts → a z-timing lever pure-ref
  (phase_speed=1.0) never exercises. With matched lateral refs, parking is no longer easier
  than swinging, so 0.10 is safe where 0.3 was not.

**Phase D training (RUNNING):** warm-start `2026-06-05_15-08-05/model_3000.pt`, run
`2026-06-08_19-50-39`, PID in `/tmp/mr_train.pid`. First iter (3017) already beats all prior
runs: `error_joint_pos` 0.67 (vs ~1.36 at residual 0.3 — tracks matched ref tightly),
`ball_missed_paddle` **0.676** (vs 0.81–0.89 failed / 0.77–0.83 Stage-1 base),
`ball_on_own_table` 0.317 (contact up). `ball_hit`/`ball_return` still at floor — needs
iterations. Decision point ~`model_4500`: `ball_hit` off floor, miss drops+sustains,
`ball_return`/`ball_land_opponent` clear 0. If returns stay 0 → next structural step is
height-matched refs (3 lateral × N height, needs `_assign_motion_by_ball` change). Safety:
SIGINT/SIGTERM only.

### Multi-ref 4500 gate verdict — FAILED to return, but failure now isolated to height

Trailing-mean(±30) to the gate (latest logged iter 4541):

| signal | it3300 | it4000 | **it4500** | read |
|---|---|---|---|---|
| `error_joint_pos` | 0.479 | 0.395 | **0.389** | ✅ rides matched ref tightly (R1/F2 collapsed to 1.0–1.6) |
| `swing_timing` | 0.011 | 0.013 | **0.018** | ✅ timed swing fires + sustains (~10× failed runs) |
| `racket_ball_proximity` | 0.070 | 0.074 | **0.075** | paddle consistently *near* ball |
| `ball_missed_paddle` | 0.707 | 0.783 | **0.752** | ❌ never drops+sustains (0.71–0.79 all run) |
| `ball_hit` | 0.0003 | 0.0002 | **0.0003** | ❌ flat on floor |
| `ball_return` | 0.0001 | 0 | **0.0001** | ❌ peak 0.0036 @3948 = flicker, not trend |
| `ball_land_opponent` | 0 | 0 | **0** | ❌ peak 0.0020 @3262, transient |

**None of the plan's go criteria passed** (ball_hit off floor / miss drops+sustains /
returns clear 0). **But the failure mode changed decisively:** the two pathologies in every
prior run — tracking collapse (`errJpos`→1.0+) and abandoned swing (`swingT`→0) — are *gone*.
The policy now rides the laterally-matched reference and executes the timed swing on-target,
yet still does not contact. This isolates the remaining gap to a **single axis (height)** —
consistent with the Phase-B gate's `|dz|` median 0.065 m / `min_gap` 0.109 m.

**Two competing hypotheses for zero contact (must disambiguate before building height refs):**
1. *Height gap* (gate read): paddle passes ~0.065 m off in z → needs height-bucketed refs.
2. *Residual-wander*: residual 0.10 + `action_std` 0.25, contact never rewarded (sparse) →
   trained policy may jitter *off* the pure-ref contact line; its real min_gap could be
   *worse* than the pure-ref 0.109 m.

**Next step (evidence before escalation):** SIGINT the plateaued run, then run the
instrumented diagnostic (`play_pure_ref` geometry probe) on the **trained `model_4500`** with
residual active. dz-dominant ≈0.065 → confirm height-matched refs (3 lateral × N height,
modify `_assign_motion_by_ball` to select on predicted-z; height likely baked via the
shoulder-pitch joint or fixed `joint_lift` per bucket — needs `_fk_jac_probe.py` to find the
paddle-z gain). min_gap degraded vs 0.109 → it's residual-wander; fix is tighten residual /
add on-manifold contact bootstrap instead. Run dir `2026-06-08_19-50-39`, model_4500 saved.

### Trained-policy probe verdict — REFUTES height-first; it's PHASE/RESIDUAL WANDER (decisive)

Ran `probe_policy_geom.py` (NEW; trained policy, residual+phase_speed ACTIVE, multi-ref
selection live) on `2026-06-08_19-50-39/model_4500.pt`, 120 serves:

```
median min_gap        = 0.130 m   (pure-ref baseline 0.109 — TRAINED IS WORSE)
frac min_gap<0.08     = 6.7%      (pure-ref 33.5% — collapsed)
median phase@min_gap  = 0.407     (hit_phase target 0.4643 — EARLY)
median SIGNED offset(ball-racket) dx=+0.004 dy=+0.109 dz=+0.026
median |offset| per axis           |dx|=0.015 |dy|=0.106 |dz|=0.054   <-- dy dominates, NOT dz
```

Per-bucket (motion_id 0=middle 1=left 2=right):

| bucket | trials | ball_y got | signed dy | dz | min_gap |
|---|---|---|---|---|---|
| id0 middle | 31% | +0.019 | +0.114 | +0.026 | 0.137 |
| id1 left   | 20% | +0.077 | +0.083 | +0.072 | 0.143 |
| id2 right  | 49% | −0.057 | +0.105 | +0.017 | 0.119 |

**Reads, in order of what they rule out:**
1. **NOT a selection bug.** All 3 buckets fire; ball_y routing is monotone (middle +0.019 /
   left +0.077 / right −0.057) — selection correctly sorts serves by predicted-y. ✓
2. **NOT primarily a calibration bug.** Bucket *spread* works (paddle clusters at y≈−0.006 /
   −0.095 / −0.162 for left/mid/right — moves the right direction). But all three carry the
   **same uniform +0.08–0.11 signed dy** — a per-bucket miscalibration would differ in
   sign/size per bucket, not sit at one uniform offset.
3. **IT IS POLICY/PHASE WANDER.** `min_gap` 0.130 is **worse than the policy's own
   zero-residual reference (0.109)** — the trained policy degraded its initialization. The
   uniform ~−0.1 m racket-y offset coincides with `phase@min_gap = 0.407` (early vs 0.4643):
   the swing is a lateral sweep, so contacting *early* samples a frame where the paddle hasn't
   yet carried to +y → systematic −y miss. residual-0.10 on yb_2 (gain 0.41) only buys ±0.041 m
   lateral; the extra ~0.064 m beyond pure-ref must come from **phase_speed shifting the
   closest-approach to an earlier, laterally-incomplete frame**. `phase_speed_reg = −0.0005`
   ≈ no regularization, so phase_speed has free rein.

**Corrected diagnosis (supersedes the "isolated to height" line above):** the dominant
trained-policy error is **lateral (dy 0.106), self-inflicted by phase_speed/residual**, not
height. Building height-matched refs now would fix the *wrong axis* on top of a policy that is
already worse than its own reference. The height gap (Phase-B clean |dz| 0.065) is real but
**downstream** — it only becomes the limiter *after* the policy is constrained back onto the
laterally-good reference.

**Recommended fix sequence (no GPU spend yet — awaiting user direction):**
1. **Rein in phase_speed** (primary): tighten range 0.85–1.15 → ~0.95–1.05 and/or raise
   `phase_speed_reg`, and/or add a reward for contacting *at* hit_phase. Removes the early-
   contact lateral wander.
2. **Revert residual_scale 0.10 → 0.05** (secondary): the Phase-B bump to 0.10 (intended for
   height fine-tune) bought lateral wander instead. Hold the policy near the reference.
3. **Re-probe.** If min_gap recovers to ~0.109 with |dz| as the clean limiter but still
   won't contact (<0.06 needed), THEN — and only then — close the residual *height* gap via a
   directed z lever (per-bucket lift/z_offset or height-matched refs). NB: pure-ref min_gap
   0.109 on a *real successful return* swing implies a sim serve-vs-demo calibration offset —
   moving the ball (serve calib) may be cheaper than moving the paddle (height refs). See
   memory `ball-serve-calibration-0602`.

## Session 2026-06-09 — both levers applied (user: "Both levers" + "re-probe first")

Applied both levers to `env_cfg.py` and re-trained (clean A/B, same Stage-1 base
`2026-06-05_15-08-05/model_3000.pt`, only the two knobs changed):
- `phase_speed` range **0.85–1.15 → 0.95–1.05** (env_cfg.py:219-220)
- `residual_scale` **0.10 → 0.05** (env_cfg.py:212)
- New run `2026-06-09_16-45-34`, SIGINT-stopped at `model_4500` (the standard gate).

### TB gate (trailing-mean ±30, vs prior multi-ref at same iters)

| signal | MULTI(res.10/ph.85-1.15) @4500 | NEW(res.05/ph.95-1.05) @4500 | read |
|---|---|---|---|
| `ball_missed_paddle` | 0.752 | **0.605** | ✅ −0.15 **and sustained** (best sustained miss of any run; R1/F2 reverted to 0.83) |
| `error_joint_pos` | 0.389 | **0.347** | ✅ held tighter to matched ref |
| `swing_timing` | 0.018 | 0.021 | ✅ sustained |
| `ball_hit` | 0.0003 | 0.0006 | ↑ 2× but still floor |
| `ball_return` | 0.0001 | 0.0000 | ❌ still zero |

### Trained-policy geometry probe on NEW model_4500 (120 trials) — fix VALIDATED, partial

| metric | prior model_4500 | **NEW model_4500** | pure-ref |
|---|---|---|---|
| **phase@min_gap** | 0.407 (early) | **0.459** ✅ on target | ~0.452 |
| median min_gap | 0.130 | **0.121** | 0.109 |
| frac<0.08 | 6.7% | **21.7%** (3×) | 33.5% |
| signed dy / \|dy\| | +0.109 / 0.106 | **+0.076 / 0.080** | – / 0.042 |
| \|dz\| | 0.054 | 0.063 | 0.065 |

Per-bucket (NEW): id0 middle 42% dy **+0.100** gap 0.139 | id1 left 38% dy **+0.013** gap
0.103 (**lateral-solved**) | id2 right 19% dy +0.072 gap 0.106.

**Verdict:** the phase fix is a clean win — closest-approach now lands AT hit_phase (0.459 vs
0.4643), early-contact wander gone, min_gap recovered most of the way to pure-ref, frac<0.08
tripled. **Remaining lateral gap is no longer uniform**: previously all 3 buckets carried a
uniform +0.10 dy (uniform wander); now the **left bucket is solved (dy +0.013)** while
**middle is worst (+0.100)**. That divergence is the signature of *per-bucket reference
miscalibration* — but could be residual wander concentrated in middle/right. `|dz|` (0.063) is
now co-dominant with `|dy|` (0.080). Contact/return still on floor (expected — even the solved
left bucket's min_gap 0.103 > 0.06 contact threshold).

**Disambiguation in flight:** added per-bucket breakdown to `play_pure_ref.py` (motion_id
tracking + aggregation, mirrors `probe_policy_geom.py`); running it (residual 0, phase 1.0,
natural serves) on the real 3 refs. If pure-ref middle-bucket dy ≈ +0.10 → middle REFERENCE
miscalibrated (cheap yb_2 offset fix, no retrain). If pure-ref middle ≈ 0 → trained +0.10 is
residual wander → tighten residual instead. Result pending; decides calibrate-refs vs
constrain-policy before the (still-deferred) height decision.

### Pure-ref per-bucket result (120 trials) — refs are GOOD, policy WANDERS, left-band too wide

| bucket | trials | **pure-ref** dy / dz / min_gap | **trained m4500** dy / min_gap |
|---|---|---|---|
| middle | 23%/42% | +0.020 / +0.025 / **0.069** ← near contact | +0.100 / 0.139 |
| left   | 52%/38% | +0.036 / +0.003 / **0.143** ← worst, majority | +0.013 / 0.103 |
| right  | 24%/19% | −0.001 / +0.021 / 0.099 | +0.072 / 0.106 |
| overall | 120 | dy −0.021, dz +0.017, min_gap 0.112, phase 0.479 | dy +0.076, min_gap 0.121, phase 0.459 |

**Three conclusions:**
1. **Calibration is OFF the table.** Pure-ref per-bucket dy all near zero (+0.020/+0.036/−0.001);
   the middle ref at residual 0 reaches min_gap **0.069 ≈ contact threshold**. Nudging yb_2
   offsets would *break* good refs.
2. **Trained policy WANDERS — the immediate enemy.** It turns middle's 0.069 → 0.139 (+0.08 dy)
   and right's 0.099 → 0.106. residual-0.05 on yb_2 alone buys only ±0.02 m, so the drift is
   **multi-joint residual FK + residual phase earliness** (trained 0.459 vs ref-natural 0.479,
   still slightly early within the 0.95–1.05 range).
3. **Left bucket (52% of serves, majority) is structurally under-covered.** Laterally centered
   (dy +0.036) and height-fine (dz +0.003), yet pure-ref min_gap **0.143** — because the left
   band pred_y ∈ [+0.05, +0.24] (0.19 m wide) is covered by ONE reference. Within-bucket serve
   spread, not calibration/height. The structural root (one ref can't span the serve spread)
   resurfaces at finer resolution: 3 buckets is too coarse for the +y majority band.

Height (dz) is NOT the limiter (per-bucket dz ≤ 0.025) — the earlier Phase-B |dz| 0.065 was
absolute-deviation with sign cancellation; signed/per-bucket dz is small. **The deferred
height question is answered: not height.**

**Next-move fork (awaiting user direction):**
- **A — kill the wander (cheap, no code):** tighten `phase_speed` 0.95–1.05 → ~0.98–1.02 (or
  freeze 1.0) + `residual_scale` 0.05 → ~0.03. Holds policy on the refs where middle/right
  nearly contact (0.069/0.099). Retrain+re-probe. Validates "faithful tracking → contact" and
  could yield the first returns on the ~47% middle+right serves. Does NOT fix the 52% left band.
- **B — finer lateral coverage (code + pure-ref gate):** replace 3 discrete buckets with a
  **continuous** pred_y-driven yb_2 shift on a single ref (paddle lands on pred_y for ANY
  serve, collapsing the left-band spread), or split into ≥5 buckets. The real generalization
  fix for the majority bucket. Wander-kill levers from A still needed on top.
- **C — both at once:** tighten levers AND continuous shift in one retrain (fewer GPU cycles,
  muddier attribution).
