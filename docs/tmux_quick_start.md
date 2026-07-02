# tmux 简明指南

tmux 用来创建可以断开、恢复的终端会话。适合跑训练、仿真、TensorBoard、日志同步等长任务；SSH 断开后，tmux 里的进程仍会继续运行。

## 基本流程

新建一个会话：

```bash
tmux new -s train
```

在 tmux 里运行任务，例如：

```bash
cd /data/PPO-pingpong
/data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
  --headless \
  --task A1-TableTennis-SAC-Catch \
  --num_envs 1024
```

断开会话但不停止任务：

```text
Ctrl-b d
```

查看已有会话：

```bash
tmux ls
```

重新进入会话：

```bash
tmux attach -t train
```

结束指定会话：

```bash
tmux kill-session -t train
```

## 常用快捷键

所有快捷键都先按 `Ctrl-b`，松开后再按后面的键。

| 快捷键 | 作用 |
|---|---|
| `Ctrl-b d` | 断开当前会话 |
| `Ctrl-b c` | 新建窗口 |
| `Ctrl-b n` | 切到下一个窗口 |
| `Ctrl-b p` | 切到上一个窗口 |
| `Ctrl-b ,` | 重命名当前窗口 |
| `Ctrl-b %` | 左右分屏 |
| `Ctrl-b "` | 上下分屏 |
| `Ctrl-b 方向键` | 切换分屏 |
| `Ctrl-b x` | 关闭当前分屏 |
| `Ctrl-b [` | 进入滚动模式 |
| `q` | 退出滚动模式 |

## 推荐用法

训练时建议把输出写入日志，方便断开后回看：

```bash
mkdir -p logs/manual
/data/miniforge3/envs/isaac/bin/python scripts/sac_table_tennis/train.py \
  --headless \
  --task A1-TableTennis-SAC-Catch \
  --num_envs 1024 \
  2>&1 | tee logs/manual/train_$(date +%Y%m%d_%H%M%S).log
```

TensorBoard 可以单独开一个会话：

```bash
tmux new -s tensorboard
tensorboard --logdir logs
```

如果只想临时回到 tmux 里看一下任务状态：

```bash
tmux attach -t train
```

看完后用 `Ctrl-b d` 退出，不要直接 `Ctrl-c`，否则会中断正在运行的任务。

## 常见问题

### 提示 duplicate session

说明同名会话已经存在。可以换一个名字：

```bash
tmux new -s train2
```

或者进入已有会话：

```bash
tmux attach -t train
```

### 不小心关掉 SSH

重新 SSH 到机器后执行：

```bash
tmux ls
tmux attach -t train
```

只要会话还在，任务通常仍在运行。

### 滚动查看历史输出

进入滚动模式：

```text
Ctrl-b [
```

用方向键或 PageUp/PageDown 滚动，按 `q` 退出。

