# serve_gen — 物理逼真发球生成器(自足工具)

把 76→数千条**物理逼真、无 sim2real gap** 的发球烘焙数据造出来,喂给 HitTrack 训练,修复"仅 72 条离散发球态"的过拟合/覆盖问题。KF 已 vendored 进本目录(方案 B,自足)。

## 核心思路(drag/bounce 模型)

1. **辨识 faithful 球模型**:对每条真实发球(`data/0629_RL_traj/`)拟合(初速 v0 + 每条球的 `air_drag_coeff`/`bounce_alpha_z`/`bounce_alpha_xy`),用 KF 自己的物理(**Magnus 关**,和部署一致)复现真实轨迹。实测:全程 RMSE **2.06cm**、落点 **1.70cm**。→ 满足"放出的轨迹匹配真实"。
2. **拟合分布采样**:对 (pos0,v0,drag,alpha_z,alpha_xy) 拟合多元高斯(`envelope_scale` 可拓宽,采样裁剪到物理边界),采样大量新发球。
3. **过同一套 KF**:合成球(逐球 drag/bounce)喂进 vendored KF(**用固定 tuned 参数**)→ KF/KF_pred。因为真实球的阻力/反弹**逐球偏离 KF 的单一 tuned 值**、KF 追不上,**KF_pred 误差自然复现真机曲线**(纯物理缺口 2.6–7.4cm → 现在 1.5–2.6cm,形状正确,无需手动注入,也不引入 Magnus)。
4. **复用 bake**:输出 recording-schema csv,`bake_hittrack_references.py` 原样烘焙成 npz。

> **为什么不用自旋/Magnus**:实验 3 证明位置-only 数据下,"自旋"与"逐球阻力/反弹偏差"**退化等价**(复现误差都 ~1.8cm)。自旋需要假设一个 Magnus 系数(而部署 KF 特意把 `magnus_coeff=0`),且绝对自旋不可辨识;drag/bounce 留在 KF 模型族内、无假设系数,更站得住脚。发现:逐球变动主要在**反弹**(alpha_z 0.879→0.933、alpha_xy 0.73→0.77),drag 几乎不变——物理上合理(反弹最不确定)。

## 关键事实 / 验证

- KF = 纯 Python(vendored 自 `/home/woan/kalman_filter_pingpong` @ `61c17ba`);配置 == `tuned_params_0617.json`。
- **V1 移植保真**:vendored KF 复现录像 `KF_*`/`KF_pred_*` → 滤波态 0.65cm、预测 2.01cm(velocity_fit OFF,与部署一致,`validate_deploy_pred.py` 确认)。
- **物理保真**:无扰动单点滚动误差 == 真机 KF_pred 同 tau 误差(内在预测难度,非 bug)。
- **实验 3**:自旋 vs 阻力/反弹复现真实轨迹**等价**(1.85 vs 1.73cm)→ 选 drag/bounce。
- **faithful 拟合**:v0+drag/bounce → 2.06cm 复现,留 66/80。
- **曲线自然复现**:faithful 合成过 KF → KF_pred 误差-tau 曲线贴合真机 ~1.5-2.6cm(形状对)。
- **V3 覆盖**:合成过网态包络真实(填补 72 点空洞)。

## 文件

| 文件 | 职责 |
|---|---|
| `kalman.py` / `serve_gated_kalman.py` | vendored KF(逐行拷贝) |
| `ball_physics.py` | KF-physics 前向 rollout;`default_config`(KF/预测侧)/`ball_config(drag,alpha_z,alpha_xy)`(生成侧,magnus off)/`gen_config`(旧自旋,仅实验);`predict_to_plane` |
| `recording_io.py` | 按 serve_id 加载录像 |
| `mocap_noise.py` | 从真实 (mocap−KF) 残差稳健标定测量噪声(~1-2mm) |
| `serve_fit.py` | **per-serve 拟合**(默认 `--model dragbounce`;`--model spin` 仅对比);`--save`+质量门控 |
| `serve_state_dist.py` | (pos0,v0,drag,alpha_z,alpha_xy) 多元高斯 + `envelope_scale`(采样裁剪到物理边界) |
| `generate_serves.py` | **主生成器**:采样→逐球 `ball_config` 物理→加噪→KF→recording csv |
| `deploy_predict.py` | 复现部署 KF_pred(确认部署用 velocity_fit OFF) |
| `experiment3_dragbounce.py` | 判别实验:自旋 vs 阻力/反弹 |
| `validate_physics_match.py` / `validate_kf_port.py` / `diagnose_pred_error.py` / `compare_synthetic_curve.py` / `verify_curve_from_fit.py` | 验证/诊断 |

## 用法

```bash
cd serve_gen
# 1) 辨识 faithful 模型(一次性,存参数)
python3 serve_fit.py --data-dir /data/PPO-pingpong/data/0629_RL_traj --save fitted_serves.npz
# 2) 生成 N 条合成发球(recording schema)
python3 generate_serves.py --fit fitted_serves.npz --n 2000 \
    --out-dir /data/PPO-pingpong/data/synth_serves --envelope-scale 1.0
# 3) 烘焙(复用真实 bake,不改)
python3 ../bake_hittrack_references.py --data-dir /data/PPO-pingpong/data/synth_serves \
    --out /data/PPO-pingpong/data/synth_serves/hittrack_references_synth.npz --hit-plane-x -1.44
# 4) 训练指到新 npz:env_cfg.py 的 HITTRACK_BAKED_PATH 改成合成 npz(或与真实混合)
```

## 待办 / 已知

- V4 下游:训练 + held-out 真实发球成功率(Isaac,主训练量,交用户跑)。
- 残差:合成 KF_pred 略低于真机(未复刻 HOLD→release 全瞬态);可接受。
- `envelope_scale=1.0` 已略宽于真实(包络,利于覆盖);vz/z 尾偏宽、vz 均值略下偏;如需严格同分布可把采样裁剪到真实范围。
- drag/bounce 与自旋在位置-only 下退化等价;此处的 drag/bounce 是"等效逐球偏差",复现轨迹+KF_pred 结构,非反演真实空气动力学。
