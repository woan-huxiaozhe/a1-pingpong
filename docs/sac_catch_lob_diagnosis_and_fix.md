# SAC Catch — Lob Plateau: 诊断、两轮 reward-shaping 修复实测、根因结论

Companion to `traditional_swing_analysis_plan.md`. Branch: `feat/sony-ace`.
Runs referenced:
- **baseline** `2026-06-24_17-38-32` (~65.7k upd) — the original plateau.
- **fix-v1** `2026-06-24_20-44-47` (~241k upd) — added normal + predicted-landing shaping.
- **fix-v2** `2026-06-25_10-44-41` (~22.7k upd) — tightened normal tol 45→30, raised alpha floor 0.02→0.05.

Geometry (`ROBOT_SIDE=-1`): net at x=0; opponent table x∈(0,1.37), **center 0.685**; own table x∈(−1.37,0); table height 0.76 m; interception plane ≈ robot_x −1.47 (≈2.1 m from target).

---

## 0. TL;DR

- **The target is real and provably reachable.** The deployed traditional task-space controller returns the ball at a paddle-tip cruise speed of **~1.3 m/s**, and the drag-aware analytic `ideal_racket_velocity` independently lands on the same **~1.3–1.5 m/s paddle / ~28° / ~4.7 m/s ball, face-normal elevation ~20–28°**. **Most return energy is reflection of the incoming ball, not swing speed.** A good return *flickers in and out* during training → kinematically reachable.
- **Yet SAC parks on a serve-independent ~49° / ~3.4 m/s lob** (valid_return 2–4 %), landing ~0.3 m short on its **own** side.
- **Two reward-shaping fixes did not break the lob** (§3). fix-v1's new dense terms "engaged" by the watchlist but plateaued at a level the lob already satisfies; fix-v2 tightened the gradient + unfroze exploration and the lob still reformed.
- **Root conclusion (§4): the dense PRE-CONTACT approach-window shaping is fundamentally misaligned.** Its maximum is "hold the ideal face/velocity *during the approach window*", which is decoupled from the single **contact instant** — so the policy Goodharts it instead of returning the ball. fix-v2 is the proof: the dense velocity/normal terms peak early (normal_match→0.99) while valid_return≈0, then **negatively track** valid_return as the policy must trade proxy score for real returns.
- **Path forward (§5): judge stroke quality AT the contact instant** (the `hit` event, using the cached contact normal+velocity), and demote the dense window terms to a bootstrap-only weight.

---

## 1. The target & existence proof (traditional scheme / physics)

| source | paddle-tip speed | ball launch | face-normal elevation |
|---|---|---|---|
| deployed traditional controller (cruise) | **~1.3 m/s** | — | tight, low-std (held) |
| analytic `ideal_racket_velocity` (drag-aware, e=0.75) | ~1.2–1.4 m/s | ~4.7 m/s @ 28° | ~20–28° |

Key physics: from the deep interception plane the return is **reflection-dominated** — the paddle only adds ~1.3 m/s along the face normal; the ball's outgoing speed comes mostly from reflecting the incoming serve. So **the blade FACE NORMAL at contact is the dominant control variable**, not swing speed. Counterfactual (relaunch each captured contact, drag-aware, vary only the launch):

| relaunch | median landing x | % on opponent table |
|---|---|---|
| actual (49° / 3.4 m/s) | −0.12 | 4.4 % |
| flatten to 28°, **same** speed | −0.09 | 16.9 % |
| flatten to 20°, same speed | −0.18 | 0 % |
| full ideal (28° / **4.7 m/s**) | +0.73 | **100 %** |

→ **flat AND fast together** is the only returnable stroke; flattening alone or speeding alone both fail.

---

## 2. The plateau & the contact-state gap (baseline `17-38-32`, 296-contact rollout)

Pre-contact paddle state at every first contact vs the per-serve analytic ideal
(`rollout_contact_state.py` → `analyze_contact_state.py`):

| quantity (median) | RL actual | analytic ideal | gap |
|---|---|---|---|
| face-normal angle vs ideal `n` | — | — | **11.3°** |
| face-normal elevation | **31.7°** | 21.5° | **+10° too steep** |
| \|v_paddle\| (body-origin) | **1.00 m/s** | 1.48 m/s | **−32 % under-swing** |
| ball launch angle | **49°** | 28° | +21° (lob) |
| ball launch speed | 3.4 m/s | 4.7 m/s | too slow |
| contact center offset | 0.085 m | 0 | edge contact |

Reflection from the **actual** normal+velocity predicts a 48.4° launch ≈ the 49° measured → the lob is fully explained by the **~10°-too-steep face** (launch angle ≈ 2× face elevation, so the bias is amplified). **The discriminator that ISN'T:** valid_return vs bad_hit contacts have *identical* normal/elevation/paddle-speed/launch — they differ only in **incoming serve speed**. I.e. **one stereotyped open-loop lob applied to every serve**; the 2–4 % "successes" are just the serves that came in fast enough to carry the same lob over the net.

---

## 3. Two reward-shaping fixes — tried & measured

Active dense pre-contact terms (all gated to the *approach window*: dist<0.45, ball incoming, not yet hit/missed; clean privileged ball):
- `racket_ideal_velocity_match` (w10): full-vector match of body-origin paddle velocity to the per-step analytic `v_paddle_ideal`.
- `racket_ideal_normal_match` (w8): face-normal vs analytic `n_ideal`, linear-in-angle `clamp(1−angle/tol)`.
- `racket_predicted_landing` (w12): reflect ball off current blade pose+velocity → drag-aware `predict_landing_xy` → Gaussian on opponent center.

### 3.1 fix-v1 `20-44-47` — add normal_match(tol **45**) + predicted_landing, keep velocity_match

By the watchlist the shaping "engaged": `normal_match`≈0.78, `predicted_landing`≈0.17. **But it plateaued at a level the lob satisfies:**
- tol=45° → the lob's 10°-steep face scores `1−10/45 ≈ 0.78` (near-saturated) → almost no gradient to flatten the last 10°.
- `predicted_landing`≈0.17 **coexists with actual landing −0.30 m** → the per-step "if you hit now" hypothetical (blade-center velocity, transient approach pose) is farmable **without** the actual contact aiming at center. Falsifies the assumption that this term ≈ "this stroke is a real return".
- **Exploration frozen:** `alpha` pinned at the 0.02 floor the entire run (alpha_loss ≈ −41) → near-deterministic policy locked in the lob basin from early on.
- Outcome: valid_return 2.8 %→4.7 % (eval best), landing −0.30, apex 1.36, outgoing 2.0 — **same lob**, edge contact even worse.
- **"reward slowly declines, not rises":** decomposing `reward/total_mean` (first→last 10 %): dense terms **erode** (velocity_match −0.17, normal_match −0.13 raw) while sparse terminal **grows** (return_bonus +0.02 @w50, valid_return 0.3 %→4.3 %). The near-deterministic policy drifts *within* the lob basin (apex rose, edge worsened); dense erosion > sparse gain → total drifts down. Not divergence (critic/q stable).

### 3.2 fix-v2 `10-44-41` — tol 45→**30**, alpha floor 0.02→**0.05**

Fresh run; config confirmed (`angle_tol_deg=30.0`, `alpha`≡0.05). At 22.7k upd it is **early** (hit_rate ramped 0.09→0.65), but leading indicators show the lob **reforming** (apex→1.33, landing −0.34, valid_return≈0, eval@20k 0.006) and exposed the root problem:

- **Negative correlation = Goodhart proof.** `normal_match` shoots to **0.99 by ~4k upd while valid_return=0**, then declines to ~0.55 as valid_return ticks up. Over the run r(valid_return, normal_match)≈−0.05, velocity_match≈+0.03 (≈0 overall — the *visible* anti-correlation is the early proxy over-optimization). To get a real return the policy must **leave** the "hold ideal pose" configuration → it gives up proxy score, so the proxy and the objective compete. **Tightening tol to 30 made the proxy a *stronger* early attractor** (normal_match hit 0.99 faster/higher than v1).
- **`miss_approach` r = −0.28.** Two layers: (a) definitional — it fires only on the `miss` event, mutually exclusive with a return; (b) it pays a **positive reward for near-missing** (`exp(−8d²)`, w5), an attractor toward "hover, don't commit". Now near-zero as hit_rate rose; should be cut.
- **The 10k-step oscillation is a measurement artifact, not instability.** `eval_interval`=10k → after each greedy eval the loop does a full `env.reset()` + clears `episode_stats`/`reward_term_stats` (`train.py:456-461`, deliberately, to keep eval out of the replay buffer). The resulting **synchronized fresh episodes + near-empty stat windows** make the per-window means swing wildly for ~1–2k upd then re-converge (e.g. `normal_match` logs **1.95**, above its own [0,1] clamp → proves it's a near-empty-denominator artifact). Losses (q1, actor, alpha) are smooth across the boundary. Mild real cost: lockstep envs briefly reduce replay diversity. Acceptable; if it bothers curve-reading, raise `eval_interval` or ignore the post-eval windows.

---

## 4. Root conclusion: dense pre-contact shaping is the wrong tool

The three dense terms are **per-step approach-window** rewards. Their optimum is *"during the window, hold the paddle at the ideal face/velocity / make the hypothetical land at center"* — a behavior that **does not require completing a committed swing through the ball**, and is evaluated at **different instants** than the single frame of actual contact. Therefore:

1. The optimum of the proxy ≠ the task success → the policy maximizes the proxy and lobs (satisfiable without success). This is the same disease across baseline, v1, and v2.
2. Strengthening the proxy gradient (tol 45→30) or unfreezing exploration (alpha 0.02→0.05) does **not** help, because the gradient points to the proxy optimum, not the goal — v2 made the early over-commitment worse.
3. The physics says the **contact-instant face normal** is the dominant variable (reflection-dominated, §1), yet it is shaped only *before* contact and never *bound to* the contact frame.

---

## 5. Next step (structural, not another dense knob)

1. **Contact-instant quality reward.** On the `hit` event, score the cached contact normal + paddle velocity + incoming ball (`v_out = v_in − (1+e)((v_in−v_p)·n)n` → `predict_landing_xy`) against opponent center. Binds the reward to the frame that physically determines the return; removes the "farm at a non-contact instant" channel.
2. **Cut the dense window terms by default** (`velocity_match`/`normal_match`/`predicted_landing`). If they return, make it an explicit ablation or annealed bootstrap, not the main learning signal.
3. **Cut `miss_approach`** (near-zero, pays for a non-goal).
4. Raise alpha floor to 0.10 for the sparse/event-heavy pass; consider a serve/return-distance curriculum if the open-loop stereotype persists.
5. **Verify definitively** (not just aggregates): rerun `rollout_contact_state.py` on the new `agent_best.pt` and check face elevation→21°, \|v_paddle\|→1.5, launch→28°, and that valid vs bad contacts finally **differ** in normal/launch (aiming, not lobbing-and-hoping).

---

## 6. Dense reward audit and sparse/event-heavy simplification

Current conclusion: for the next run, treat exploration/replay as the search mechanism and let **event rewards** decide credit. The dense terms below either optimize the wrong frame, reward a non-goal, or are only hardware/smoothness regularizers.

| term | fires | weight before 0625 sparse pass | decision | reason |
|---|---|---:|---|---|
| `racket_ideal_velocity_match` | every approach-window step | `10.0` | **cut by default** | Full-vector velocity at non-contact instants can be farmed by holding an ideal-looking pose; v1/v2 showed it does not force a committed swing through contact. |
| `racket_ideal_normal_match` | every approach-window step | `8.0` | **cut by default** | v2 made this proxy score ~0.99 while `valid_return≈0`; stronger normal matching pulled into the proxy optimum, not the return objective. |
| `racket_predicted_landing` | every approach-window step | `12.0` | **cut by default** | "If hit now" counterfactual can score while the real contact still lobs short; the scored instant and actual contact frame are decoupled. |
| `miss_approach` | `miss` event | `5.0` | **cut by default** | It pays positive reward for near-missing, and is mutually exclusive with return. This creates a hover/chase attractor. |
| `table_proximity` | `bad_hit` event | `10.0` | **keep small** (`5.0`) | It is terminal/event-shaped, not per-step dense. Keep only as a signed bridge for bad hits that nearly cross the net; do not let it dominate actual return. |
| `landing_placement` | `valid_return` event | `25.0` | **keep** | Scores actual landing on opponent table; tied to the real outcome frame. |
| `flat_return` | `valid_return` event | `10.0` | **keep** | Only pays after a valid return; discourages high lobs without adding a negative dense penalty. |
| `racket_spin_penalty` | `hit` event | `-5.0` | **keep smaller** (`-2.0`) | Still useful for clean contact, but too much early spin pressure can suppress exploratory swings before valid returns are common. |
| `action_rate`, `joint_acc`, `joint_jerk` | every step | tiny negative | **keep** | These are sim-to-real smoothness regularizers, not task-shaping. They are already low. |
| `joint_limit`, `joint_effort_margin` | every step | `-3.0`, `-0.5` | **keep** | Safety/hardware barriers. `joint_effort_margin` is already relaxed so it should not kill the swing. |

Default config was changed to the sparse/event-heavy pass:

| term | new weight | effective event value (`*0.02`) |
|---|---:|---:|
| `hit_bonus` | `20.0` | `+0.40` |
| `return_cross_net` | `40.0` | `+0.80` |
| `return_bonus` (`valid_return`) | `100.0` | `+2.00` |
| `landing_placement` | `25.0` | `0 -> +0.50` |
| `flat_return` | `10.0` | `0 -> +0.20` |
| `table_proximity` | `5.0` | `-0.10 -> +0.10` |
| `racket_spin_penalty` | `-2.0` | `0 -> -0.04` |

Exploration was changed in two places:
- `initial_alpha = min_alpha = 0.10` so the stochastic actor keeps broader coverage after dense proxies are removed.
- `--start_steps` default is `128000` transitions. With `--num_envs 1024`, that is about one full 2.5 s episode of random actions before SAC updates begin; the old `20000` transitions were only ~20 policy steps and often ended before any meaningful random outcome existed.

Expected TensorBoard signature:
- Early run: `reward_terms/racket_ideal_*` and `reward_terms/racket_predicted_landing` should disappear; `reward_terms/return_cross_net` should become the bridge between `hit_rate` and `valid_return_rate`.
- If exploration works, `episode/hit_rate` may rise slower than the dense-proxy run, but `episode/return_rate` should separate from `bad_hit_rate` instead of both tracking the same lob.
- Success criterion is not just `valid_return_rate`; rerun contact-state analysis and require face elevation, launch angle, and paddle speed to move toward the analytic contact-frame target.

---

## 7. Reproduce / tooling

Env (calling the env python directly fails to import `isaacsim`; the `activate.d` hooks are required):
```bash
source /home/woan/miniforge3/etc/profile.d/conda.sh && conda activate /data/miniforge3/envs/isaac
```
- Contact-state rollout: `python scripts/sac_table_tennis/rollout_contact_state.py --headless --num_envs 128 --episodes 400 --checkpoint <agent_best.pt> --out <json>`
- Offline gap analysis (no Isaac): `python scripts/sac_table_tennis/analyze_contact_state.py <json>`
- Reward fire/finiteness sanity: `python scripts/sac_table_tennis/verify_new_rewards.py --headless`
- Standalone physics tests: `pytest tests/test_ideal_racket_velocity.py -q`

Open / out of scope: **traditional-swing FK reference** (the full existence proof, `traditional_swing_analysis_plan.md` §3) not yet computed (pinocchio lives in the oneroarm env, not isaac); the ~1.3 m/s / 28° figures above are the deployed-controller cruise speed + the analytic ideal, which agree. **H5 ready-pose affordance:** whether a flat 21° face at the deep contact is reachable from `SAC_READY_JOINT_POS` (vs a joint limit) is unverified — FK pass deferred unless the contact-instant reward underperforms.
