# U0 Paired REPA：实现、机制与创新空间分析

更新日期：2026-10-15

## 1. 文档目的

U0 paired REPA 是当前项目中最稳定、最有效的方法族。本文件回答四个问题：

1. U0 在代码中究竟做了什么；
2. 它为什么可以被理解为 trajectory-invariant representation learning；
3. 当前结果实际支持哪些结论、不支持哪些结论；
4. 如何从 U0 出发形成比“多视图 + 相同 teacher target”更有创新性的下一方法。

对应训练脚本为：

```text
scripts/dlc_tfcr_jobs/65_u0_paired_repa_ratio1_400k.sh
```

核心配置位于：

```text
scripts/tfcr_ablation.sh -> u0_paired_repa
```

## 2. U0 的实验设置

### 2.1 基础训练配置

| 项目 | 设置 |
|---|---|
| 模型 | SiT-B/2 |
| 训练步数 | 400k |
| source batch size | 256 |
| prediction | velocity `v` |
| path | linear interpolant |
| weighting | uniform |
| teacher | frozen DINOv2 ViT-B |
| teacher token width | 768 |
| student alignment depth | SiT block 8 |
| REPA coefficient | 0.5 |
| paired source ratio | 1.0 |
| independent-noise probability | 0.5 |
| timestep gap | `[0.15, 0.7]` |
| CFG dropout | 两个 view 共享同一个 mask |
| sampling/eval | 400k、50K、SDE 250 steps、CFG 1.0 |

`factor_batch_ratio=1.0` 表示每个 source image 都构造两个轨迹状态。若 source batch 为 (B)，主干一次处理 (2B) 个 trajectory states，但同一 source 的两项 denoising loss 会先平均，因此最终仍形成 (B) 个 source-level loss。U0 的计算量应与 A3 paired-view control 比较，不能只与单视图 baseline 比较。

### 2.2 两个轨迹视图如何构造

给定 clean latent (x_0)，先采样间隔：

\[
\Delta t \sim U(0.15, 0.7).
\]

再在合法范围内采样一低一高的两个 timestep，并以 0.5 概率交换顺序：

\[
|t_a-t_b|=\Delta t.
\]

在线性路径下：

\[
x_{t_a}=(1-t_a)x_0+t_a\epsilon_a,
\]

\[
x_{t_b}=(1-t_b)x_0+t_b\epsilon_b.
\]

noise intervention 为：

- 50%：`time-only`，两条 view 共享同一个 noise，\(\epsilon_a=\epsilon_b\)；
- 50%：`time + noise`，两条 view 使用独立 noise。

因此 U0 不是简单的数据重复，它让同一 source 在两个明确不同的 diffusion state 下出现，并同时覆盖时间变化与噪声变化。

### 2.3 条件信息保持一致

两个 view 使用相同类别标签，并共享 classifier-free guidance dropout mask。否则一条 view 可能有类别条件、另一条被置为 unconditional，模型会把 conditioning 变化误认为 trajectory variation。

共享 CFG mask 的作用是保证 pair 中受控变化主要来自：

- timestep；
- noise realization；
- 而不是 class conditioning。

## 3. U0 的训练目标

### 3.1 双视图 denoising

线性 interpolant 的 velocity target 分别为：

\[
v_a=\dot\alpha(t_a)x_0+\dot\sigma(t_a)\epsilon_a,
\]

\[
v_b=\dot\alpha(t_b)x_0+\dot\sigma(t_b)\epsilon_b.
\]

每个 source 的 denoising loss 是两个 view 的平均：

\[
L_{denoise}=rac{1}{2}
\left(
\|f(x_{t_a},t_a)-v_a\|^2+
\|f(x_{t_b},t_b)-v_b\|^2
\right).
\]

这要求网络不能为了 invariance 丢掉当前 trajectory state；它仍必须区分两个 timestep/noise 状态并预测不同 velocity。

### 3.2 共享 clean-image teacher target

原始 clean image 经冻结 DINOv2-B 得到 normalized patch tokens：

\[
z=T_{DINO}(x_0).
\]

同一个 (z) 被复制给两个 view。SiT block 8 的 token 经过 MLP projector：

\[
\hat z_a=P(h_8(x_{t_a},t_a)),
\]

\[
\hat z_b=P(h_8(x_{t_b},t_b)).
\]

代码中的 REPA loss 是 normalized token cosine similarity 的负值，并在两个 view 上平均：

\[
L_{REPA}=-\frac{1}{2}
\left(
\cos(\hat z_a,z)+\cos(\hat z_b,z)
\right).
\]

默认 `factor_proj_target_view_mode=both`，所以不是只监督其中一条轨迹；两条轨迹都直接看到同一个 external semantic target。

### 3.3 总目标

U0 没有额外显式 factorization regularizer，实际目标可写成：

\[
L_{U0}=L_{denoise}+0.5\,s(k)L_{REPA},
\]

其中 (s(k)) 是随训练 step 变化的 schedule。

以下系数全部为零：

- invariant loss；
- persistent loss；
- evolving loss；
- recomposition loss；
- transition loss。

因此 U0 的收益不能归因于 persistent/evolving 分解。

### 3.4 REPA schedule

U0 使用 factor schedule 控制 REPA：

\[
s(k)=s_{warmup}(k)s_{decay}(k).
\]

- 0--10k：线性 warmup，从 0 增长到 1；
- 10k--250k：保持 1；
- 250k--400k：cosine decay 到 0；
- 400k：REPA scale 为 0。

这意味着 U0 并不是在整个训练过程中持续以固定强度复制 DINO，而是：

1. 早期逐步建立语义坐标系；
2. 中期稳定跨轨迹语义；
3. 后期逐渐释放 teacher，让生成目标完成适配。

## 4. 为什么 U0 是 invariant representation learning

对于同一个 source：

\[
P(h_8(x_{t_a}))\approx z,
\]

\[
P(h_8(x_{t_b}))\approx z.
\]

由此间接得到：

\[
P(h_8(x_{t_a}))\approx P(h_8(x_{t_b})).
\]

所以 projector 可读取的表示被要求对以下 nuisance 稳定：

- timestep；
- 部分 pair 中的独立 Gaussian noise；
- 同一 source 的不同 trajectory state。

但需要准确描述这种 invariance：

- invariant 的是 block 8 中可被 REPA projector 读取的部分；
- 不是完整 hidden tensor 被强制相等；
- denoising objective 同时要求 hidden state 保留 state-dependent information；
- 因而 U0 更像“隐式 semantic/state coexistence”，而不是显式二分。

一个更准确的机制描述是：

> U0 用 shared DINO target 指定跨轨迹应保持稳定的语义，用双视图 velocity prediction 指定必须随轨迹变化的动态信息，二者共同塑造同一 hidden representation。

## 5. 为什么 U0 可能有效

### 5.1 teacher 提供稳定语义坐标系

仅要求两条 noisy trajectory feature 彼此相似，模型可能学习低信息量捷径。DINO target 提供了由 clean image 决定的高信息量语义锚点，减少 collapse 或 nuisance shortcut。

### 5.2 paired view 把 REPA 从单点监督变成轨迹约束

标准 REPA 只要求随机抽到的一个 state 对齐 clean target；U0 在同一步中明确要求同一 source 的两个相隔较远 state 都能恢复相同语义。这使监督直接覆盖 trajectory variation。

### 5.3 denoising 防止过度 invariance

如果只有 shared semantic target，两个 view 可以完全相同。但 velocity target 不同，模型仍需保留足够的 timestep/noise information。语义稳定性与生成动态充分性在同一 hidden space 内竞争并达成隐式平衡。

### 5.4 后期 teacher decay 降低任务冲突

REPA 在 250k 后逐渐衰减，可能使模型先获得稳定表示，再在后期专注生成目标。这是 U0 的重要组成部分，但现有结果也说明 schedule 本身不是全部原因：单独 scheduled REPA 并不能达到 U0。

## 6. 实验证据

### 6.1 U0 三次结果

| Job | Seed | FID | sFID | IS | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 65 | 0 | 23.043364 | 6.611714 | 63.543480 | 0.593080 | 0.646400 |
| 72 | 1 | 22.849329 | 6.482513 | 64.937309 | 0.592020 | 0.641600 |
| 73 | 2 | 22.573737 | 6.533122 | 64.813347 | 0.590380 | 0.648300 |
| Mean | — | 22.822143 | 6.542450 | 64.431379 | 0.591827 | 0.645433 |

### 6.2 关键对照

| 方法 | FID | 对 U0 的解释 |
|---|---:|---|
| no-REPA baseline | 35.900961 | 没有 external semantic coordinate |
| A3 paired two-view | 30.107791 | 只有双视图 compute，不足以解释 U0 |
| A5 explicit TFCR | 29.825559 | 显式 factorization 贡献有限 |
| standard A2 REPA seed0 | 28.154142 | 只有单状态 REPA，明显弱于 U0 |
| U0 paired REPA seed0 | 23.043364 | 双轨迹共享 external semantic target |
| U2 selective semantic | 23.270599 | 显式选择性语义没有超过 U0 |
| U8 late weak decomposition | 23.918411 | 在 U0 上加分解反而退化 |
| Job 106 P2 split + REPA | 23.531836 | 固定 semantic/state split 未增强 U0 |
| Job 108 semantic-only REPA | 23.789724 | 把对齐限制到固定 semantic branch 更差 |
| Job 110 identifiable + REPA | 23.980545 | 更强可辨识约束继续退化 |

上述结果支持：

1. paired view 本身有效，但不是主要增益的全部来源；
2. external teacher target 是关键；
3. 同一 target 覆盖两条 trajectory view 比普通单视图 REPA 更强；
4. 当前固定 semantic/state 分解没有提供额外收益。

### 6.3 尚未被证明的结论

FID 改善不能单独证明：

- U0 的 linear probe 一定更好；
- block 8 的 feature 真正更 invariant；
- spatial correspondence 得到提升；
- 跨层 representation 更丰富；
- 模型已经把 teacher 知识内化到不依赖 projector 的 backbone。

这些需要通过 probe、retrieval、CKA、effective rank 和 intervention 直接验证。

## 7. U0 当前实现的局限

### 7.1 单一 student depth

U0 只在 block 8 读取 REPA representation，没有回答 invariance 应该在早、中、晚哪一层出现，也没有利用 DiT 的层级结构。

### 7.2 单一 teacher level

两条轨迹在所有 timestep 都对齐 DINOv2-B 的同一最终 patch representation。低噪声可恢复的局部信息和高噪声可恢复的全局信息没有区分。

### 7.3 uniform trajectory treatment

高噪声状态可能没有足够信息恢复精细 patch semantics，但 U0 仍使用相同 target 和相同权重。这可能迫使模型依赖 class prior，或造成不必要的 teacher/denoising 冲突。

### 7.4 只定义 invariance，不定义 equivariance

U0 指定了 projector 输出应该不变，却没有显式刻画 hidden feature 应如何随 timestep/noise 改变。动态信息完全由 denoising objective 间接维持。

### 7.5 跨层职责未知

没有约束或测量不同 block 是否学到互补信息。多个 block 可能逐渐同质化，也可能已经自然分工，目前没有直接证据。

### 7.6 view generation 与 factorization 基础设施耦合

代码通过 `trajectory_factorization` 路径生成 paired views，即使所有显式 factor loss 都为零，仍会实例化和返回 factorization head。方法上 U0 并不需要这些 head。后续应把“paired trajectory data path”从“factorization objective”中独立出来，使方法定义更简洁，也避免无用参数和分析歧义。

### 7.7 创新叙事仍然有限

“两个 augmentation/view 对齐同一 teacher target”在自监督和蒸馏研究中是常见范式。U0 的特殊性是 view 来自 diffusion trajectory，但若只停留在 paired REPA，创新点仍偏工程组合。

## 8. 可挖掘的创新方向

### 8.1 Trajectory-adaptive hierarchical invariance

核心假设：不同 timestep 的可恢复语义粒度不同，不同 DiT depth 也应承担不同表示职责。

\[
L_{hier}=\sum_l w_l(t)
D(P_l(h_l(x_t)),T_{r(l)}(x_0)).
\]

- early DiT block 对齐 teacher early/intermediate feature；
- middle block 对齐 object/part representation；
- late block 对齐 global semantics；
- (w_l(t)) 根据 SNR 或 recoverability 调整。

创新点是 trajectory-conditioned invariance，而不是简单增加更多 REPA 层。

### 8.2 Invariant–equivariant representation geometry

不要再固定切分 semantic/state channel。使用 pair feature difference 描述动态方向：

\[
\Delta h_l=h_l(x_{t_b})-h_l(x_{t_a}).
\]

同时学习：

\[
P_l(h_l(x_{t_a}))\approx P_l(h_l(x_{t_b})),
\]

\[
Q_l(\Delta h_l)\rightarrow
\phi(\Delta t,\epsilon_b-\epsilon_a).
\]

这把“什么保持不变”和“什么随轨迹规律变化”统一在 representation geometry 中，比固定 bottleneck 更自然。

### 8.3 Recoverability-aware target granularity

按当前 state 能恢复的信息决定监督内容：

| 状态 | 合理 target |
|---|---|
| low noise | patch-level local structure + object semantics |
| middle noise | part/object semantics |
| high noise | pooled/global semantics |

权重可以先由确定性 SNR schedule 给出，再考虑 student/teacher similarity 或 retrieval confidence gate。

### 8.4 跨 block 的互补 invariance

目标应是：

- 同一 block 跨 trajectory 稳定；
- 不同 block 学到互补的 invariant information。

优先通过不同 teacher layer routing 获得自然分工。如果实测 off-diagonal block CKA 仍过高，再加入带 margin 的弱 CKA/cross-covariance 约束，并同时保持 variance/effective rank，避免用 collapse 换取低相似度。

### 8.5 Semantic geometry preservation

U0 是 point-to-point target matching。可以进一步研究不同 trajectory state 是否保持 teacher 的邻域结构：

\[
\operatorname{sim}(P(h_i^t),P(h_j^t))
\approx
\operatorname{sim}(z_i,z_j).
\]

关键不是普通 relation distillation，而是根据 timestep 改变邻域粒度：低噪声保持实例/patch 邻域，高噪声只保持类别或 coarse semantic 邻域。

### 8.6 Teacher knowledge internalization

U0 在后期将 REPA 权重衰减到零，但这只是撤销监督，不代表 student 已经内化 teacher。更严格的方法是依据 representation quality 决定 teacher 是否退场，例如：

- cross-noise retrieval 达到门槛；
- student/teacher neighborhood agreement 达到门槛；
- effective rank 和 class separation 不再依赖 teacher loss。

K3 已说明固定 step-based dropout 仍弱于 U0，因此下一步应研究 quality-gated internalization，而不是继续改变 dropout 时间点。

## 9. 推荐的核心方法叙事

下一方法可以组织为 **Trajectory-Adaptive Hierarchical Paired REPA**：

> Diffusion trajectory 中可恢复的语义随噪声强度变化，因此模型不应在所有 timestep 和所有 block 上学习相同的不变性。方法根据 trajectory recoverability，将不同粒度的 teacher semantics 路由到不同 DiT depth；同时在 feature-difference direction 中保留对 timestep/noise dynamics 的 equivariance。

相对 U0，真正新增的研究问题是：

1. invariance 的粒度是否应随 trajectory state 改变；
2. invariance 是否应在网络深度上形成层级结构；
3. invariant semantic direction 与 equivariant dynamic direction 能否共存；
4. 这种结构是否提升直接 representation metric，而不只改善 FID。

## 10. 最小验证顺序

### Step 0：先分析 U0

- 导出 U0 各 block、各 timestep、各 noise 的 feature；
- 测量 class probe、cross-noise/time retrieval；
- 测量 layer/timestep CKA 和 effective rank；
- 确认 U0 的 invariance 主要出现在哪些 block；
- 检查 block 8 projector 前后是否都获得收益。

### Step 1：只做 hierarchical routing

- 不增加 diversity、state split 或 adaptive gate；
- 三个 student depth 对应三个 teacher depth；
- 判断不同层是否形成不同的 probe/correspondence profile。

### Step 2：加入 trajectory-adaptive target

- low/middle/high noise 使用不同 teacher granularity；
- 先使用固定 SNR schedule；
- 检查高噪声 denoising 与 global retrieval 是否改善。

### Step 3：必要时加入 equivariance

只有当 timestep/velocity probe 相对 U0 下降时，才加入 feature-difference dynamic objective。

### Step 4：必要时加入 block complementarity

只有当 routed model 的跨层 CKA 仍明显偏高时，才加入弱 diversity regularizer。

## 11. 成功标准

下一方法需要同时满足：

1. 至少一个直接语义指标明确超过 U0，例如 class probe 或 cross-noise retrieval；
2. timestep/velocity probe 不明显下降；
3. effective rank 不坍缩；
4. 不同 routing block 的表示职责可通过 probe/CKA 观察；
5. FID 不比 U0 seed0 差超过约 0.3；
6. 改善能够由 trajectory recoverability/hierarchy 解释，而不是新增参数或计算量。

## 12. 当前结论

U0 的核心价值不是简单的双倍数据或显式 factorization，而是：

> 对同一 source 的不同 trajectory state，同时施加共享 clean semantic target 和各自 state-dependent denoising target，从而在同一 hidden space 中隐式形成 semantic invariance 与 dynamic sufficiency。

它已经证明这是一个有效方向，但创新性仍主要停留在 paired trajectory REPA。下一步最有价值的提升不是继续增加固定 semantic/state loss，而是研究 **随 timestep 改变的可恢复语义层级、跨 depth 的 teacher routing，以及 invariant/equivariant representation geometry**。
