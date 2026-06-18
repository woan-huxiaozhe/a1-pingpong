# A1 Table Tennis SAC — `valid_return_rate` Oscillation Diagnosis & Reward-Stability Ablations

Last updated: 2026-06-15

Companion to `docs/sac_table_tennis_architecture.md`. This note records the root-cause diagnosis of the persistent `valid_return_rate` oscillation in `A1-TableTennis-SAC-Catch`, the four ablation experiments run on 2026-06-15, their results, and the open decision on what to try next. It is written to be self-contained so the investigation can resume in a fresh session.

---

## 1. The problem

Run under analysis: `logs/sac_table_tennis/A1-TableTennis-SAC-Catch/2026-06-12_18-59-20` (the converged 0612 run, 0→200k updates, num_envs=1024).

Symptom: the policy **can catch and return** the ball (hit_rate ≈ 0.99, valid_return_rate mean ≈ 0.85), but `reward/total_mean` and `episode/valid_return_rate` **never settle** — they oscillate persistently around a converged mean instead of flattening. User's initial hypothesis: two rewards are mutually exclusive (互斥).

## 2. Root-cause diagnosis (data-backed, holds)

The oscillation is a **limit cycle on a flat reward ridge**, not a two-reward conflict.

- `hit_rate` ≈ 0.99 essentially constant. `valid_return` and `bad_hit` form a near-binary partition of hits (Pearson r ≈ −0.975): a hit resolves as either a valid return or a bad hit.
- `reward/total_mean` is dominated by **dense per-step `post_hit_*` shaping** (r = 0.994 with `post_hit_landing_prediction`), and that shaping is nearly flat across the normal operating band → **weak restoring gradient**. The sparse terminal signal that actually distinguishes good vs bad outcomes (`valid_return` +25, `bad_hit` −3) is tiny per-step after averaging.
- Amplified by: (a) `min_alpha` pinned at its floor → near-deterministic policy; (b) **deterministic serve** (`SAC_FIXED_MIDDLE_BALL`, all ranges degenerate: x=1.25, vx=−3.4, vz=2.0); (c) a **single shared policy across 1024 envs** → when the policy drifts, all envs flip together, producing synchronized "bad-hit storms".
- The storms are **under-hits** (lower outgoing speed, ball fails to cross the net), not overshoots.

## 3. Experiment infrastructure (reusable)

**Conda env** (the only one that has isaaclab/isaacsim/unitree_rl_lab):
```
source /home/woan/miniforge3/etc/profile.d/conda.sh; conda activate isaac
PY=/data/miniforge3/envs/isaac/bin/python    # py3.11, torch 2.7.0+cu128, isaacsim 5.1.0, isaaclab 0.54.3
```
(conda *base* is `/home/woan/miniforge3`; `/data/miniforge3` is only an envs dir. The `lerobot` env is py3.10 and does NOT have isaaclab — a false-positive trap.)

**Two CLI flags added to `scripts/sac_table_tennis/train.py`** for ablations (do not change defaults):
- `--reward_weight_override "term=w,term2=w2,..."` — overrides `env_cfg.rewards.<term>.weight` **before** env creation (and before `dump_yaml`, so it's recorded in `params/env.yaml`). Unknown term → hard error.
- `--min_alpha_override <float>` — overrides `agent.config.min_alpha` **after** agent creation/resume, recomputes `_min_log_alpha`, and clamps. Needed because `SACAgent.load` restores `min_alpha` from the checkpoint, so editing the `SACConfig` default would NOT take effect on resume.

**Ablation protocol:** resume from the 200k checkpoint
(`.../2026-06-12_18-59-20/checkpoints/agent_final.pt`), run 50k updates (200k→250k), compare the **last 348 log-points** (equal window, ~last 35k updates) of `episode/valid_return_rate`. log_interval=100 → 500 points per resume run; last 348 skips the ~150-pt resume transient.

Example launch (the cut_posthit+alpha005 run):
```
$PY scripts/sac_table_tennis/train.py --headless --task A1-TableTennis-SAC-Catch \
  --num_envs 1024 --max_updates 250000 --resume_checkpoint "$CKPT" \
  --reward_weight_override "post_hit_net_clearance=1,post_hit_outgoing=0,post_hit_net_progress=0" \
  --min_alpha_override 0.005 \
  --log_dir logs/.../EXP_cut_posthit_alpha005 --checkpoint_interval 5000
```

**Analysis script:** `/tmp/compare4.py` (pure-Python TFRecord+protobuf parser via `/tmp/parse_tb.py`, no TB deps). Equal-window stats + alpha trajectory + per-term tail means. Event files per run:
- baseline: `.../2026-06-12_18-59-20/events.out.tfevents.1781261960...`
- zero_approach: `.../EXP_zero_approach_prox/events.out.tfevents.1781498161...`
- cut_posthit: `.../EXP_cut_posthit/events.out.tfevents.1781503802...`
- cut+alpha005: `.../EXP_cut_posthit_alpha005/events.out.tfevents.1781507990...`

## 4. Results — four runs, equal 348-pt steady-state window

| run | overrides | hit | valid_mean | **valid_std** | valid_min | bad_mean | bad_max | storms>.3 | storms>.4 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | none (0612 config) | 0.991 | 0.847 | 0.094 | 0.369 | 0.141 | 0.564 | 22 | 12 |
| zero_approach | `racket_approach=0, racket_ball_proximity=0` | 0.993 | 0.846 | 0.076 | 0.478 | 0.144 | 0.519 | 18 | 4 |
| **cut_posthit** | `post_hit_net_clearance 4→1, post_hit_outgoing→0, post_hit_net_progress→0` | 0.994 | 0.841 | **0.066** ✅ | 0.445 | 0.151 | 0.555 | 14 | 7 |
| **cut+alpha005** | cut_posthit **+** `min_alpha 0.02→0.005` | 0.919 | 0.737 | **0.243** ❌ | 0.000 | 0.182 | 0.744 | 29 | 18 |

`valid_std` reduction vs baseline: zero_approach −19.6%, **cut_posthit −30.1%** (best), cut+alpha005 **+158.6%** (catastrophic).

### Experiment-by-experiment

1. **zero_approach** (zero the approach/proximity dense shaping): std 0.094→0.076. Partially refuted the "approach/proximity conflict" idea — it only trims catastrophic storms (storms>.4: 12→4), not routine oscillation.
2. **cut_posthit** (cut the dense post-hit shaping): std →0.066, the biggest reduction, **mean and hit_rate preserved**. Confirms dense `post_hit` shaping is the largest oscillation lever.
3. **cut+alpha005** (additionally lower the entropy floor): **refuted the hypothesis** that the floor causes the residual oscillation. Floor change verified (alpha descended 0.0200→0.0050, pinned at floor for ~45k updates, entire window at 0.0050). Effect was **intermittent total collapses**, not larger ripples:
   ```
   update  valid   bad
   220300  0.744  0.256
   224300  0.048  0.010   ← collapse: policy stops making good contact
   228300  0.845  0.155   ← recovers
   236300  0.908  0.092   ← highest peak of ANY run
   244300  0.417  0.484   ← second collapse (48% bad-hit storm)
   248300  0.833  0.167   ← recovers
   ```

## 5. Conclusions

1. **Dense `post_hit` shaping is the oscillation lever.** Cutting it (cut_posthit) takes `valid_std` 0.094→0.066 with no loss of mean or hit_rate. Clean, attributable (same protocol as the other resume runs).
2. **The entropy floor `min_alpha=0.02` is a STABILIZER, not the oscillation source** — counterintuitive but decisive. Lowering it to 0.005 reproduced the exact failure the 0612 fix raised it (0.005→0.02) to cure: near-deterministic shared policy → synchronized collapse. **Keep 0.02. Do not lower it.**
3. **The residual 0.066 oscillation is structural** — a flat reward ridge, not entropy-injected. With the other post-hit terms cut, **`post_hit_landing_prediction` is now the dominant dense term** (tail mean 1.48/step, even higher than baseline's 1.33, ≈7× the sparse `valid_return` 0.22). It is the last flat-ridge contributor still on.

### Per-term tail means (cut+alpha005 vs baseline) — confirms the cut took effect
```
post_hit_net_clearance       0.26   (baseline 1.29)
post_hit_outgoing            0.00   (baseline 0.32)
post_hit_net_progress        0.00   (baseline 0.15)
post_hit_landing_prediction  1.48   (baseline 1.33)   ← now dominant
valid_return                 0.22   (baseline 0.25)
reward/total_mean            0.046  (baseline 0.074)
```

### Caveats
- **Resume confound:** all ablations resume from the same 200k checkpoint with an empty replay buffer, so absolute std values carry a resume transient. The clean, attributable finding is the **relative ordering under identical protocol**: cut_posthit < zero_approach < baseline ≪ cut+alpha005. To pin absolute numbers, a from-scratch control is needed.
- **Single seed.** No seed-variance estimate yet.

## 6. Open decision — next experiment (not yet chosen)

cut_posthit (std 0.066) is the best config so far; the entropy floor is settled (stays 0.02). The menu proposed:

- **A. Cut/sharpen `post_hit_landing_prediction`** (the now-dominant dense ridge term, weight 5.0), keep `min_alpha=0.02`. Tests whether removing the final flat-ridge driver pushes `valid_std` below 0.066 toward <0.04. Caveat: 0612 added this term for the `bad_hit→valid_return` gradient, so zeroing may slow learning — prefer **sharpening** (steep falloff so only near-target landings are rewarded) over zeroing.
- **B. Serve randomization.** Attack the synchronization root: the deterministic serve makes all 1024 shared-policy envs flip in lockstep. De-degenerate the serve range (within the already-defined DR scope, calibrated 0602 numbers) so storms desynchronize and average out. Also advances the sim-to-real goal.
- **C. Adopt cut_posthit + from-scratch confirm.** Bake the override weights into `env_cfg.py`, retrain fresh 0→250k (no resume) to remove the confound and lock the absolute std, then proceed toward sim-to-real fine-tuning.
- **D. Raise terminal-reward magnitude.** Instead of cutting dense shaping further, raise `valid_return`/`bad_hit` so the sparse outcome signal dominates the flat dense shaping per-step — reshape the ridge from the other side while keeping the learning gradient.

Recommendation at time of writing: **A (sharpen, not zero)**, because it directly targets the identified residual driver while keeping `min_alpha=0.02`; B is the better move if the goal pivots toward sim-to-real robustness.

## 7. Related
- `docs/sac_table_tennis_architecture.md` — pipeline architecture (update in the same change as code).
- Memory: `sac-catch-reward-stability-fix.md` (0612 fix + this 0615 ablation summary), `tabletennis-env-tuning-scope`, `ball-serve-calibration-0602`.
