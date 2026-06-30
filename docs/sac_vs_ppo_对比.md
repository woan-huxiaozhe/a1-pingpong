# SAC vs PPO 对比文档

> 针对本项目(Isaac Lab + Unitree A1 乒乓球)训练器选型的对比参考。
> 结论先行,再从数学原理展开,最后给出选型建议与验证方案。
> 相关任务:`A1-TableTennis-SAC`(Catch,现用 SAC)、`A1-Pingpong-HitTrack`(新,trainer-agnostic)、
> `table_tennis` forehand/backhand(现用 rsl_rl PPO)。

---

## 0. 一句话核心差异

> **PPO** 优化标准期望回报,用**似然比(score-function)梯度**,并把每步更新**约束在上一版策略附近**;
> **SAC** 优化**"回报 + 熵"**的最大熵目标,用**重参数化(pathwise)梯度穿过一个学到的 Q 网络**,并把策略**拉向均匀分布**。
>
> 其余区别(on/off-policy、要不要 critic、如何稳定)几乎都从这两点推导而来。

---

## 1. 目标函数(最根本的分歧)

记 MDP 为 $(\mathcal S,\mathcal A,P,r,\gamma)$,策略 $\pi_\theta(a\mid s)$。

**PPO —— 标准 RL 目标:**

$$J_{\text{PPO}}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\, r(s_t,a_t)\Big]$$

**SAC —— 最大熵目标(奖励里加策略熵):**

$$J_{\text{SAC}}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\big(r(s_t,a_t)+\alpha\,\mathcal H(\pi_\theta(\cdot\mid s_t))\big)\Big],\qquad \mathcal H(\pi(\cdot\mid s))=-\mathbb E_{a\sim\pi}[\log\pi(a\mid s)]$$

熵项不是"附加探索奖励",它**改变了最优解的形式**。对最大熵目标求解,最优策略是关于软 Q 值的 Boltzmann 分布:

$$\boxed{\;\pi^*(a\mid s)\;\propto\;\exp\!\Big(\tfrac1\alpha\,Q^*_{\text{soft}}(s,a)\Big)\;}$$

$\alpha\to0$ 时退化为标准 RL 的确定性贪心最优。**PPO 的最优解始终是确定性贪心;SAC 的最优解永久保留随机性**,$\alpha$ 控制"随机性 vs 最优性"的权衡。

---

## 2. 软值函数(目标函数的直接后果)

熵项使 SAC 的 Bellman 方程变"软":

$$Q_{\text{soft}}(s_t,a_t)=r(s_t,a_t)+\gamma\,\mathbb E_{s_{t+1}}\big[V_{\text{soft}}(s_{t+1})\big]$$
$$V_{\text{soft}}(s)=\mathbb E_{a\sim\pi}\big[Q_{\text{soft}}(s,a)-\alpha\log\pi(a\mid s)\big]$$

软 Bellman 算子仍是压缩映射,soft policy iteration 在表格情形下收敛。
PPO 不学软值;它只用一个普通状态值 $V(s)$ **当基线**降方差,从不学 $Q$。

---

## 3. 梯度估计方式(机制上的核心)

### 3.1 PPO:似然比 / score-function 梯度

策略梯度定理:

$$\nabla_\theta J=\mathbb E_{\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,A^\pi(s,a)\big]$$

只要求 $\log\pi$ 对 $\theta$ 可导,优势 $A$ 是个**标量**——**不需要对动作可导的 critic,也不需要对环境求导**。优势用 GAE 估计:

$$\hat A_t^{\text{GAE}}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\qquad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$$

原始策略梯度方差大、步子一大就崩,PPO 用**截断代理目标**约束步长。重要性比

$$\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\qquad
L^{\text{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]$$

这是 TRPO 信赖域 $\max_\theta\mathbb E[\rho_t\hat A_t]\ \text{s.t.}\ \mathbb E[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}\|\pi_\theta)]\le\delta$ 的一阶近似。
**关键**:$\rho_t$ 只有在 $\pi_\theta\approx\pi_{\theta_{\text{old}}}$ 时低方差、代理才准 —— 这是 **PPO 必须(近似)on-policy** 的数学根源。

### 3.2 SAC:重参数化 / pathwise 梯度穿过 critic

策略改进 = 逼近上面的 Boltzmann 分布,即最小化 KL:

$$\pi_{\text{new}}=\arg\min_\pi D_{\mathrm{KL}}\!\Big(\pi(\cdot\mid s)\,\Big\|\,\tfrac{1}{Z(s)}\exp(\tfrac1\alpha Q_\phi(s,\cdot))\Big)
\;\Longleftrightarrow\;
J_\pi(\theta)=\mathbb E_{s\sim\mathcal D}\,\mathbb E_{a\sim\pi_\theta}\big[\alpha\log\pi_\theta(a\mid s)-Q_\phi(s,a)\big]$$

为低方差求梯度,用**重参数化**把随机性挪到外部噪声:$a=f_\theta(\epsilon;s)=\tanh\!\big(\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon\big),\ \epsilon\sim\mathcal N(0,I)$:

$$\nabla_\theta J_\pi=\mathbb E_{s,\epsilon}\Big[\nabla_\theta\big(\alpha\log\pi_\theta(a\mid s)\big)+\underbrace{\big(\nabla_a\alpha\log\pi_\theta(a\mid s)-\nabla_a Q_\phi(s,a)\big)\nabla_\theta f_\theta(\epsilon;s)}_{\text{穿过 critic 的 pathwise 项}}\Big]_{a=f_\theta}$$

那一项 $\nabla_a Q_\phi\,\nabla_\theta f_\theta$ 把 **critic 对动作的梯度反传进策略**。pathwise 估计通常比 score-function 方差低得多,**代价是必须有对动作可导的 $Q_\phi(s,a)$**。

$Q_\phi$ 用 off-policy TD 学(replay buffer $\mathcal D$ + 双 Q 抗高估 + target 网):

$$J_Q(\phi)=\mathbb E_{(s,a,r,s')\sim\mathcal D}\Big[\tfrac12\big(Q_\phi(s,a)-y\big)^2\Big],\qquad
y=r+\gamma\Big(\min_{i=1,2}Q_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\Big),\ \ a'\sim\pi_\theta(\cdot\mid s')$$

Bellman 回填**不关心数据是哪个旧策略产生的**(只要目标里的 $a'$ 来自当前策略)→ 可从 replay 复用历史数据,这是 **SAC 能 off-policy** 的根源。PPO 做不到:其梯度估计器要求动作来自(接近)当前策略。

### 3.3 温度自调(SAC 特有)

$$J(\alpha)=\mathbb E_{a\sim\pi}\big[-\alpha\big(\log\pi_\theta(a\mid s)+\bar{\mathcal H}\big)\big]$$

把 $\alpha$ 调到命中目标熵 $\bar{\mathcal H}$(常取 $-\dim\mathcal A$)。PPO 有时加固定 $+\beta\mathcal H$ 探索奖励,但**不进入目标的不动点**。

---

## 4. 稳定化机制:同一套 KL 正则,锚点不同

两者都可写成"带 KL 正则的策略优化",区别只在**把策略拉向谁**:

- **PPO 锚向上一版策略**:约束 $D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\le\delta$。proximal/信赖域项,只约束**每一步**(稳定、单调改进),**不改变不动点**——仍收敛到标准贪心最优。
- **SAC 锚向均匀分布**:由 $\mathcal H(\pi)=\log|\mathcal A|-D_{\mathrm{KL}}(\pi\|\mathrm{Uniform})$,最大化熵 $\Leftrightarrow$ 最小化与均匀分布的 KL。它写进**目标本身**,**永久改变不动点**——最优策略是 Boltzmann 分布,天然鲁棒。

> 一句话:**PPO 的 KL 是"别离刚才太远"(稳);SAC 的 KL 是"别离均匀太远"(探索/鲁棒,且改写最优解)。**

---

## 5. 数学维度汇总

| 数学维度 | PPO | SAC |
|---|---|---|
| 目标 | $\mathbb E[\sum\gamma^t r]$ | $\mathbb E[\sum\gamma^t(r+\alpha\mathcal H)]$ |
| 最优策略 | 确定性贪心($\alpha\to0$ 极限) | Boltzmann $\propto e^{Q/\alpha}$ |
| 梯度估计 | score-function $\ \nabla\log\pi\cdot A$ | reparam pathwise $\ \nabla_aQ\cdot\nabla_\theta f$ |
| 需对动作可导的 critic | 否(只用 $V$ 基线) | 是(必须学 $Q_\phi$) |
| 数据使用 | on-policy(重要性比仅在 $\pi\approx\pi_{\text{old}}$ 有效) | off-policy(replay + TD 回填) |
| 稳定 / 正则 | 截断代理 / 信赖域,锚 = $\pi_{\text{old}}$ | 熵正则 + target 网 + 双 Q min,锚 = 均匀 |
| 方差–偏差 | 方差高、偏差低(无 bootstrap) | 方差低、偏差来自 bootstrap / Q 高估 |
| 活动部件 | actor + V;clip/GAE/KL 超参 | actor + 双 Q + target 网 + α;replay buffer |

---

## 6. 工程层面优劣(本项目语境)

### 6.1 为什么"样本效率"直觉在这里要打折

教科书说"SAC 样本效率高、PPO 低",前提是**采样昂贵**(真机、慢仿真)。但在 Isaac Lab 里跑 **2048 并行 env**,采样近乎免费,瓶颈是**墙钟时间和稳定性**:

- PPO 的"样本低效"被并行度抵消 —— 这正是几乎所有 Isaac Lab 发表结果(ANYmal、各类 manipulation)都用 PPO 的原因。
- SAC 的 replay buffer + off-policy 更新在数千并行 env 上**不易吃满并行度**,其调参传统面向较小 env 数。

### 6.2 RSL-RL PPO

**优势**
- **生态原生**:RSL-RL 是 Isaac Lab 事实标准 PPO,向量化 rollout、GPU 常驻数据、runner、ONNX/JIT 导出全现成。
- **统一管线**:本项目 forehand/backhand 已在用它(含 `handle_deprecated_rsl_rl_cfg` 兼容修复),HitTrack 接入只需加一份 runner cfg,**部署/导出路径完全复用**。
- **稠密奖励天然搭档**:HitTrack 是时间门控高斯跟踪,稠密、shaped 良好,PPO 又稳又好调。
- **活动部件少**:无 replay / target / α,sim-to-real 更可预测,社区可复现资料多。

**劣势**
- 名义样本效率低(并行 sim 下不重要)。
- 对奖励 scale / GAE / clip / KL 敏感。

### 6.3 当前的 SAC

**优势**
- **已实现且 Catch 在用**,切换有迁移成本。
- **稀疏奖励探索强**:Catch 是端到端 + 稀疏事件奖励(触球、落点、Ace 阶梯),最大熵探索确实帮得上 —— 这是当初选它的合理理由。
- 采样若变贵 / 想跨任务共享 replay 时,off-policy 有回旋余地。

**劣势**
- **仓库内自维护**实现,维护面大、社区支持少。
- 大规模并行下并行度利用不如 PPO,且要额外接 agent 到新 env。
- **核心优势(稀疏探索)在 HitTrack 上用不到** —— 做 HitTrack 的动机就是把问题从稀疏改稠密,等于主动放弃了选 SAC 的理由。

---

## 7. 选型建议

| 任务 | 奖励性质 | 建议训练器 | 理由 |
|---|---|---|---|
| **HitTrack** | 稠密(高斯跟踪) | **RSL-RL PPO** | 契合度/风险/维护三方面更优;与 forehand/backhand 管线合流 |
| **Catch** | 稀疏(事件阶梯) | 暂留 **SAC** | 稀疏奖励是 SAC 主场;贸然换 PPO 可能退化 |

- **HitTrack → PPO**:低风险默认。被 spec 标为"暂缓"的训练器选择,可在此解封。
- **Catch → 暂留 SAC**:除非用"稠密参考"的 HitTrack 思路逐步替代 SAC-Catch,届时再整体退役 SAC。
- **零成本验证**:HitTrack 的 env 本就是 trainer-agnostic 设计。给个小预算**两边各跑一遍**,比较:
  1. **墙钟时间到成功阈值**(成功判据:击球步位置误差 < 0.05 m 且速度误差 < 0.2 m/s);
  2. **导出策略的关节平滑度 / sim-to-real 表现**。

  让数据拍板,而非教科书。

---

## 8. 一句话总结

差异不只是工程偏好:**HitTrack 的稠密奖励 + 大规模并行,恰好落在 PPO 数学优势(score-function 估计器无需可导 critic、on-policy 在并行下不吃亏、机制简单易稳)的那一侧;而 SAC 的最大熵探索 + 样本复用红利在这个任务上用不到。**
