# IsaacLab / rsl-rl 版本对齐坑

> 记录 2026-07-01 多机训练（本地 Legion + aliyun DSW）时踩的 `rsl-rl` / `isaaclab_rl` 版本不匹配问题。

## 一句话结论

`rsl-rl-lib` 和 `isaaclab_rl` **必须成对**。只把 `rsl-rl-lib` 升到 5.x 而 `isaaclab_rl` 仍是旧版（缺 `handle_deprecated_rsl_rl_cfg`），PPO 启动会崩。

## 正确版本组合（以本地为准）

| 组件 | 版本 |
|---|---|
| rsl-rl-lib | 5.0.1 |
| isaaclab_rl | 0.5.0（含 `isaaclab_rl/rsl_rl/utils.py` 里的 `handle_deprecated_rsl_rl_cfg`） |
| isaaclab (core) | 0.54.3 |

## 症状与根因

### 症状 1：TensorBoard 缺 `hittrack/` 栏

- `scripts/rsl_rl/train.py` 用自定义 hook 包住 `runner.logger.log` 写 `hittrack/*` 遥测。
- `runner.logger` 是 rsl-rl **5.x** 才有的接口。旧 rsl-rl 的 runner 没有 `.logger` → hook 走 `else` 分支打印 WARN 跳过 → 没有 hittrack 曲线（训练本身正常）。

### 症状 2：升级 rsl-rl 后 `KeyError: 'actor'` 崩溃

```
OnPolicyRunner(...) → construct_algorithm → cfg["actor"].pop("class_name")  →  KeyError: 'actor'
```

- 仓库 PPO cfg 用**旧 schema**（`policy=RslRlPpoActorCriticCfg`）。
- `train.py` 运行时调用 `handle_deprecated_rsl_rl_cfg()` 把它迁移成 rsl-rl 5.x 的 `actor`/`critic` + `distribution_cfg` schema，5.x runner 才认得 `cfg["actor"]`。
- 该函数来自 **isaaclab_rl**（不是 rsl-rl）。`train.py` 的 import 带了 `except ImportError` **空操作兜底**。
- 当 isaaclab_rl 旧、没有这个函数 → 走空操作兜底 → cfg 没迁移 → 崩在 `KeyError: 'actor'`。

**本质**：rsl-rl 5.x 需要迁移后的 cfg，而迁移逻辑在 isaaclab_rl。两者版本必须配套；两台机器 Isaac Lab 版本不一致就会出现"一台正常一台崩"。

## 诊断命令

```bash
# rsl-rl-lib 版本
python -m pip show rsl-rl-lib | grep -i version
# isaaclab / isaaclab_rl 版本 + 安装位置
python -m pip show isaaclab isaaclab_rl | grep -iE "Version|Location|Editable"
# isaaclab_rl 是否含迁移函数（直接 grep 源码）
grep -rl "def handle_deprecated_rsl_rl_cfg" <isaaclab_rl 安装路径>
```

> 注意：`from isaaclab_rl.rsl_rl import ...` 需要 Isaac Sim app 起来后才有 `pxr`；用普通 `python` 直接 import 会报 `No module named 'pxr'`（**假信号**）。判断 shim 在不在要 **grep 源码**，别靠 import。

## 解决方法：同步 isaaclab_rl 成配套版本

两台都是 editable 安装；isaaclab_rl 是**纯 Python**、对 core **无版本约束**（`config/extension.toml` 里 `"isaaclab" = {}`），只钉 `rsl-rl-lib==5.0.1`。所以直接同步 isaaclab_rl 这一个包即可，不用动 rsl-rl-lib、也不用整棵 IsaacLab 搬。

```bash
# ① 在版本正确的机器上打包（排除缓存）
cd <IsaacLab>/source
tar czf /tmp/isaaclab_rl.tgz --exclude='__pycache__' --exclude='*.egg-info' isaaclab_rl

# ② 传到目标机后替换 + 重装（editable 安装，base 装不含 rsl-rl-lib，5.0.1 不受影响）
mv <IsaacLab>/source/isaaclab_rl <IsaacLab>/source/isaaclab_rl.bak
tar xzf /tmp/isaaclab_rl.tgz -C <IsaacLab>/source/
<IsaacLab>/isaaclab.sh -p -m pip install -e <IsaacLab>/source/isaaclab_rl

# ③ 验证
grep -c "def handle_deprecated_rsl_rl_cfg" <IsaacLab>/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py  # 期望 1
<IsaacLab>/isaaclab.sh -p -m pip show isaaclab_rl | grep -i version   # 期望 0.5.0
```

同步后：rsl-rl 5.0.1 有 `runner.logger`，`KeyError: 'actor'` 消失，`hittrack/` 栏也自动回来（无需改代码）。

## 备注

- **workaround（不推荐）**：若无法同步版本、只想让旧 rsl-rl 出 hittrack，可改 `train.py` 的 hook 兼容旧 API —— `runner.logger` 不存在时回退去包 `runner.log` + `runner.writer`。但这样两台仍跑不同 stack，结果不可比。
- core isaaclab 差一个 patch（0.54.2 vs 0.54.3）**不影响本问题**；若要求物理/sim 严格逐位可比，再单独同步 core。
- 传文件时若 `scp` 报 `OpenSSL version mismatch`：是 `LD_LIBRARY_PATH` 里的 ROS/gazebo 库盖住了系统 openssl，用 `LD_LIBRARY_PATH= scp ...` 临时清空即可（或直接用 DSW 网页上传）。
