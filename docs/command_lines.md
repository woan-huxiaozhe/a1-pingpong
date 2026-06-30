# End2end

Start training with the following command:
```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
    --headless \
    --task A1-TableTennis-SAC-Catch \
    --num_envs 1024 \
    --start_steps 128000 \
    --max_updates 500000 \
    --log_interval 100 \
    --checkpoint_interval 20000
```

Continue training with the following command:
```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
    --headless \
    --task A1-TableTennis-SAC-Catch \
    --num_envs 1024 \
    --max_updates 300000 \
    --log_interval 100 \
    --checkpoint_interval 20000 \
    --resume_checkpoint logs/sac_table_tennis/A1-TableTennis-SAC-Catch/EXP_clean_reward_from_0615_best/checkpoints/agent_0200000.pt \
```

```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
    --headless \
    --task A1-TableTennis-SAC-Catch \
    --log_interval 100 \
    --checkpoint_interval 20000 \
    --max_updates 800000 \
    --resume_checkpoint logs/sac_table_tennis/A1-TableTennis-SAC-Catch/EXP_clean_reward_from_0615_best/checkpoints/agent_0200000.pt \
```

zero agent
```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/play.py \
    --task A1-TableTennis-SAC-Catch \
    --zero_action \
    --num_envs 1 \
    --episodes 20 \
    --max_steps 500 \
    --real_time
```

Play with a trained agent:
```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/play.py \
    --task A1-TableTennis-SAC-Catch \
    --num_envs 1 \
    --episodes 20 \
    --max_steps 500 \
    --real_time \
    --checkpoint logs/sac_table_tennis/A1-TableTennis-SAC-Catch/2026-06-23_11-22-47/checkpoints/agent_best.pt
```

```
/data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/plot_sim_joint_log.py  --show
```

logs/sac_table_tennis/sim_logs/A1-TableTennis-SAC-Catch__2026-06-17_20-33-03__agent_best.csv


# Hit Track  (PPO via rsl_rl; NOT the SAC trainer)
# train: rsl_rl/train.py uses --max_iterations (no --max_updates/--checkpoint_interval; save_interval
# lives in HitTrackPPORunnerCfg). Logs to logs/rsl_rl/a1_tabletennis_hittrack/<timestamp>/.
python scripts/rsl_rl/train.py --task A1-Pingpong-HitTrack \
    --headless \
    --num_envs 2048 \
    --max_iterations 20000 \

# play: prints per-episode tracking error + aggregate. Headless avoids the laptop-GPU Vulkan crash.
# Omitting --checkpoint auto-picks the latest model_*.pt of the latest run. Recording is ON by
# default -> writes a per-step/per-env CSV next to the ckpt (--record PATH to override; --no_record off).
python scripts/rsl_rl/play_hittrack.py --task A1-Pingpong-HitTrack \
    --headless \
    --episodes 50 \
    --checkpoint logs/rsl_rl/a1_tabletennis_hittrack/<run>/model_xxxx.pt   # or omit to auto-pick latest
#   (drop --headless and add --real-time to watch a window, GPU permitting)
#   zero-action baseline (no ckpt): python scripts/rsl_rl/play_hittrack.py --zero_action --headless --episodes 1

# visualize a record CSV: fig1 = per-joint pos(target vs actual)/vel/torque; fig2 = end-effector
# pos/vel/face-normal (actual vs target). No Isaac dep (numpy+matplotlib). With no path it reads the
# newest record CSV under logs/; default backend Agg saves PNGs next to the CSV (--show for a window).
python scripts/rsl_rl/plot_hittrack_record.py --env 0            # auto-pick first episode with a hit
python scripts/rsl_rl/plot_hittrack_record.py <csv> --list       # list (env, ep) pairs + hit/success
python scripts/rsl_rl/plot_hittrack_record.py <csv> --ep 3 --show

python scripts/rsl_rl/train.py --task A1-Pingpong-HitTrack \
    --headless \
    --num_envs 2048 \
    --max_iterations 20000 \
    --resume \
    --load_run 2026-06-30_10-58-36 \
    --checkpoint model_2000.pt


tensorboard --logdir logs/rsl_rl/a1_tabletennis_hittrack
