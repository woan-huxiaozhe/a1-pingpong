# SAC Real Serve State Pipeline

This note records the current pipeline for turning recorded ball trajectories
into a `serve_states_x1.npz` reset table for `A1-TableTennis-SAC-Catch`, plus
the IsaacSim validation errors measured before regenerating the final table.

## Goal

Use real serve trajectories instead of the independent uniform box in
`SAC_FIXED_MIDDLE_BALL`.  At each episode reset, the simulator should sample a
physically plausible incoming ball state from a table of states measured around
`x = +1.0 m`, then filter out combinations that fail before reaching the A1 hit
plane at `SAC_ROBOT_X = -1.47`.

The reset table stores rows in simulation/world coordinates:

```text
states[M, 6] = x, y, z, vx, vy, vz
```

`launch_ball` currently sets angular velocity to zero when launching from this
table, so spin is not represented in the current `npz` format.

## Source Data

Current source directories:

```text
/home/woan/kalman_filter_pingpong/data/0611_data_vel
/home/woan/kalman_filter_pingpong/data/0617_traj_data
```

Each trajectory is a VRPN-style text file.  The loader accepts at least
`t x y z`; when columns `vx vy vz` exist, `--velocity-source file` interpolates
those velocities.  `--velocity-source fit` instead fits local position windows
around the crossing time.

Coordinate conversion:

```text
z_sim = z_data + 0.714
table_z = 0.760
height_above_table = z_sim - table_z
```

The current table USD places the table top center at `z = 0.735` with thickness
`0.05`, so the top surface is `0.760 m`.

## Raw Trajectory Filter

The raw-data gate is applied at the real trajectory crossing of
`x = -1.47`, before using that trajectory as a source for `x = +1.0` reset
states.

Current gate:

```text
birth_x = +1.0
robot_x = -1.47
-0.10 < y(robot_x) < 0.30
0.00 < z_sim(robot_x) - table_z < 0.80
```

Counts from the two current datasets:

| Dataset | Files | Has birth and robot crossing | Rejected by y gate | Valid |
| --- | ---: | ---: | ---: | ---: |
| `0611_data_vel` | 59 | 58 | 21 | 37 |
| `0617_traj_data` | 53 | 52 | 13 | 39 |
| Combined | 112 | 110 | 34 | 76 |

Missing crossings:

```text
0611_data_vel: no_robot_x = 1
0617_traj_data: no_birth_x = 1
```

No source was rejected by the current height gate after passing the y gate.

## Implemented Scripts

`source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/create_serve_states.py`
builds a reset table from measured `x = +1.0` states.  It supports:

- velocity source: `file` or local polynomial `fit`
- sampler: `empirical`, `kde`, or `gaussian`
- jitter and percentile clipping
- simplified offline rollout rejection:
  - net clearance
  - table bounds
  - own-side/opponent-side bounce counts
  - arrival window at `robot_x`

`scripts/sac_table_tennis/validate_real_serve_isaac.py` validates real
trajectories directly in IsaacSim:

1. extract real state at `x = +1.0`
2. apply the raw-data `x = -1.47` y/z gate
3. reset a ball and table scene from the real `x = +1.0` state
4. step IsaacSim until the simulated ball crosses `x = -1.47`
5. report `sim - real` errors in `y, z, vx, vy, vz, tau`

`scripts/sac_table_tennis/scan_real_serve_drag.py` runs the validator for a grid
of `linear_damping` and `drag_k` values and writes `summary.csv/json` logs.

`create_serve_states.py` now applies this raw-data `x = -1.47` y/z source
prefilter by default.  Use `--no-source-hit-filter` only for diagnostic runs
that intentionally include the unfiltered source trajectories.

## Current Physics Parameters

Current table material:

```text
table restitution = 0.95
table dynamic_friction = 0.35
table static_friction = 0.40
table restitution_combine_mode = max
```

Current ball material:

```text
ball radius = 0.020
ball mass = 0.0027
ball material restitution = 0.80
ball dynamic_friction = 0.25
ball static_friction = 0.35
ball angular velocity at reset = 0
```

The validation script can apply:

```text
linear_damping: PhysX rigid-body linear damping
drag_k: quadratic drag, a_drag = -drag_k * |v| * v
```

The current SAC branch applies the lowest combined-error model from the 0.95
restitution scan:

```text
SAC_BALL_LINEAR_DAMPING = 0.05
SAC_BALL_DRAG_K = 0.08
```

`RobotEnvCfg.__post_init__` writes the linear damping into the ball rigid-body
properties, and `EventCfg.ball_air_drag` applies `mdp.apply_air_drag` every
`0.02 s` using `k = 0.08`.

## Measured Validation Error

All rows below use the 76 raw-filtered trajectories and current table
restitution `0.95`.

No drag or damping:

| Velocity source | Crossed | dz mean | dvx mean | dtau mean | yz p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fit` | 70/76 | -0.060 m | -0.914 m/s | -0.0966 s | 0.078 m |
| `file` | 76/76 | -0.089 m | -0.767 m/s | -0.0761 s | 0.103 m |

With the applied model, `linear_damping = 0.05`, `drag_k = 0.08`:

| Velocity source | Crossed | dz mean | dvx mean | dtau mean | yz p50 | yz p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 71/76 | -0.047 m | -0.104 m/s | -0.0012 s | 0.043 m | 0.245 m |
| `file` | 75/76 | -0.027 m | -0.043 m/s | +0.0084 s | 0.046 m | 0.173 m |

Best local candidates after the `0.95` restitution change:

| Candidate | Fit result | File result |
| --- | --- | --- |
| `linear_damping=0.05`, `drag_k=0.08` | crossed 71/76, dz -0.047 m, dvx -0.104 m/s, dtau -0.001 s, yz p50 0.043 m | crossed 75/76, dz -0.027 m, dvx -0.043 m/s, dtau +0.008 s, yz p50 0.046 m |
| `linear_damping=0.10`, `drag_k=0.06` | crossed 71/76, dz -0.037 m, dvx -0.158 m/s, dtau -0.011 s, yz p50 0.046 m | crossed 75/76, dz -0.032 m, dvx -0.088 m/s, dtau +0.003 s, yz p50 0.050 m |
| `linear_damping=0.15`, `drag_k=0.06` | crossed 71/76, dz -0.045 m, dvx -0.063 m/s, dtau +0.006 s, yz p50 0.040 m | crossed 75/76, dz -0.031 m, dvx +0.000 m/s, dtau +0.016 s, yz p50 0.047 m |

Interpretation:

- Increasing table restitution from `0.85` to `0.95` moved the z residual in the
  right direction.
- Drag/damping mainly fixes `dtau` and `dvx`; without it the simulated ball is
  still too fast and too early.
- Remaining `yz` error is likely dominated by missing spin and imperfect
  tangential contact modeling, not by normal restitution alone.

## Recommended Generation Flow

1. Keep table restitution at `0.95` for now.
2. Use the raw source filter at `x = -1.47`:

   ```text
   -0.10 < y < 0.30
   0.00 < z_sim - 0.76 < 0.80
   ```

3. Prefer `velocity-source file` when the velocity columns are trusted; use
   `fit` as a cross-check because it is less dependent on the file velocity
   estimator.
4. Use `empirical` sampling first.  KDE/Gaussian can be used later for
   robustness once the base distribution is stable.
5. Keep full-serve trajectories by default.  Do not force
   `--post-bounce-only` for this serve pipeline; the pre-bounce branch represents
   the real first serve that should bounce on the robot-side table before
   reaching the hit plane.

Example generation command after the raw source prefilter is applied:

```bash
python3 source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/create_serve_states.py \
  --data-dir /home/woan/kalman_filter_pingpong/data/0611_data_vel \
             /home/woan/kalman_filter_pingpong/data/0617_traj_data \
  --out source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/serve_states_x1.npz \
  --birth-x 1.0 \
  --velocity-source file \
  --sampler empirical \
  --jitter-frac 0.05 \
  --clip-percentiles 2 98 \
  --drag-k 0.08 \
  --max-opponent-bounces 1 \
  --min-own-bounces 1 \
  --max-own-bounces 1 \
  --hit-y-range -0.25 0.35 \
  --hit-z-range 0.90 1.25 \
  --hit-time-range 0.62 0.90 \
  --num-states 5000
```

Current generated table:

```text
path: source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/serve_states_x1.npz
states: (5000, 6), float32
source_states: (76, 6)
accepted_hit: (5000, 6)
ball_arrive_time_est: 0.692 s
source_hit_filter: true
velocity_source: file
drag_k: 0.08
keep_rate: 0.544
```

The output includes:

```text
states                  sampled reset states, shape (M, 6)
ball_arrive_time_est    median estimated hit-plane arrival time
source_states           measured source states at x = +1.0
accepted_hit            simplified-rollout hit-plane rows
accepted_own_bounce     simplified-rollout own-side bounce rows
clip_lo, clip_hi        sampler clipping bounds
metadata                JSON arguments and source file list
```

## Training Sampling

`env_cfg.py` points SAC to:

```text
SAC_SERVE_STATES_PATH = .../serve_states_x1.npz
SAC_USE_SERVE_STATES = True
SAC_BALL_LINEAR_DAMPING = 0.05
SAC_BALL_DRAG_K = 0.08
```

When `SAC_USE_SERVE_STATES` is true, reset calls:

```text
mdp.launch_ball(..., serve_states_path=SAC_SERVE_STATES_PATH)
```

`launch_ball` loads the table once, samples rows uniformly, applies
`env.scene.env_origins`, and writes ball root state.  The sampled state replaces
the independent uniform ranges from `SAC_FIXED_MIDDLE_BALL`.

The reset still sets:

```text
quat = [1, 0, 0, 0]
angular_velocity = [0, 0, 0]
```

So this table is a 6D translational state distribution, not a 9D
translation-plus-spin distribution.

The air-drag event runs independently of the reset table.  It affects both
states sampled from `serve_states_x1.npz` and any fallback fixed-range launch.

## Validation Commands

Direct validation of the recommended robust setting:

```bash
python3 scripts/sac_table_tennis/scan_real_serve_drag.py \
  --velocity-source file \
  --linear-damping-values 0.05 \
  --drag-k-values 0.08 \
  --out-dir logs/sac_table_tennis/real_serve_recheck
```

Small local scan around the current best region:

```bash
python3 scripts/sac_table_tennis/scan_real_serve_drag.py \
  --velocity-source fit \
  --linear-damping-values 0.05,0.10,0.15 \
  --drag-k-values 0.04,0.06,0.08 \
  --out-dir logs/sac_table_tennis/real_serve_fit_scan
```

The scan wrapper launches IsaacSim Python internally and writes a full log per
parameter pair plus `summary.csv` and `summary.json`.

## Known Limitations

- The simplified offline rollout in `create_serve_states.py` is not the same as
  IsaacSim/PhysX.  It is useful as a fast rejection gate, but the final
  generated distribution should be spot-checked with IsaacSim validation.
- The current `npz` format has no spin.  Real serves likely contain topspin,
  backspin, or sidespin, which changes tangential velocity through table contact.
- Restitution mainly controls normal bounce.  Horizontal speed loss should be
  matched with drag, damping, friction, and eventually spin, not by lowering
  restitution alone.
- Current scans match `dtau` and `dvx` reasonably well, but `yz` still has a
  several-centimeter residual and a long tail.

## Next Calibration Step

Before further tuning restitution, measure real and simulated table-bounce
states directly:

```text
vx_after / vx_before
vy_after - vy_before
vz_after / -vz_before
bounce x/y
```

That separates normal restitution error from tangential friction/spin error and
is the right next step if the `yz` residual matters for training.
