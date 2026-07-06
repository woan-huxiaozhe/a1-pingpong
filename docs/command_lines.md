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
    --num_envs 4096 \
    --max_iterations 10000

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


# 阿里云 (Aliyun DSW)

## 训练 (实例上 GPU 更强, num_envs 拉到 8192)
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task A1-Pingpong-HitTrack \
    --headless \
    --num_envs 8192 \
    --max_iterations 10000


## 日志持久化: 本地写 + 定期同步到 OSS
# 背景: /mnt/data 是 OSS(ossfs)挂载。直接把 logs 软链到 OSS, 训练的高频 append + TensorBoard
# 并发读会全部穿透 fuse -> 卡顿, 且 events 文件报 CRC 错 (BadLengthCrc, want=0x00000000, 读到半写记录)。
# 需求只是"关机后能从 OSS 下载日志", 所以让训练写本地盘, 后台定期批量拷回 OSS, 即可绕开 fuse 的坑。

# 1) 去掉指向 OSS 的软链, 让 logs 直接成为本地真实目录
#    (repo 和日志都在 /root 本地盘, 无需再套一层软链中转)
rm /root/a1-pingpong/logs
mkdir -p /root/a1-pingpong/logs

# 2) 把 OSS 上已有的 run 拉回一次 (不丢历史)
cp -a /mnt/data/a1-pingpong-logs/. /root/a1-pingpong/logs/

# 3) 后台定期把本地增量同步到 OSS 挂载点 (在 tmux 里跑, 终端断开不影响)
tmux new -s tbsync
while true; do
  cp -ru /root/a1-pingpong/logs/. /mnt/data/a1-pingpong-logs/
  sleep 600
done

# TensorBoard 只读本地, 不碰 OSS
tensorboard --logdir /root/a1-pingpong/logs/rsl_rl/a1_tabletennis_hittrack --reload_interval=30

# 注意:
# - 最长约 10 分钟窗口未上 OSS: 实例在两次 sync 之间挂掉会丢最近这几分钟日志; 想更保险把 sleep 600 调小。
# - 每次 sync 整份重传变化的文件 (ossfs 无法真正增量 append); events 几十 MB、checkpoint 仅保存时变, 10 分钟一次扛得住。
# - /root/a1-pingpong/logs 在本地盘, 关机清空, 无妨 —— 下载用 OSS 上的副本。仅"关机后断点续训"才需开机先 `cp -a /mnt/data/a1-pingpong-logs/. /root/a1-pingpong/logs/` 拉回本地。
# - 更规范 (绕过 fuse, 增量判断更准): 用 `ossutil sync /root/a1-pingpong/logs/ oss://<bucket>/<前缀>/ --update`, 但需先配 ossutil 的 AK/SK。