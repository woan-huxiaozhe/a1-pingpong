Start training with the following command:
```
  /data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
    --headless \
    --task A1-TableTennis-SAC-Catch \
    --num_envs 1024 \
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
    --checkpoint logs/sac_table_tennis/A1-TableTennis-SAC-Catch/2026-06-10_17-13-09/checkpoints/agent_0010000.pt \
```

```
/data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/plot_sim_joint_log.py  --show
```

logs/sac_table_tennis/sim_logs/A1-TableTennis-SAC-Catch__2026-06-17_20-33-03__agent_best.csv