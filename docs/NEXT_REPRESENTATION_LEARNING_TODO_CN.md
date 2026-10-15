# 下一阶段 Representation Learning 改进路线与 TODO

更新日期：2026-10-14

## 1. 当前判断

当前最可靠的方法仍是 U0 paired REPA：同一 clean image 产生两个不同 timestep/noise 的轨迹视图，两条视图共享同一个 DINOv2-B target，同时保留各自的 denoising objective。U0 三个 seed 的 FID 为 23.04、22.85、22.57，均明显优于标准单视图 REPA。

Jobs 106--111 尝试把这种隐式分工显式化为 semantic/state bottleneck。六组 400k/50K 结果均没有形成明显增益：

- 106（split + full-hidden REPA）FID 23.53，比 U0 差 0.49；
- 108（semantic-only REPA）FID 23.79，比 106 再差 0.26；
- 107（split, no REPA）FID 29.69，只比 A5 好 0.13；
- 109（semantic alignment, no REPA）FID 29.69，与 107 基本相同；
- 110（identifiability + REPA）FID 23.98，比 108 再差 0.19；
- 111（identifiability, no REPA）FID 29.64，只比 107 好 0.06。

因此下一步不继续围绕固定 semantic/state split 调系数。更合理的假设是：U0 已经通过“共享语义 target + 独立 denoising state”形成隐式分工，改进点应从固定通道切分转向以下问题：

1. invariant representation 应出现在哪些 block；
2. 不同 block 是否应学习不同粒度的 invariant information；
3. REPA 占用多少表示容量才不会挤压 timestep/noise/velocity 信息；
4. 高噪声和低噪声状态是否应使用不同的语义约束；
5. 表征能力是否真的提升，而不只是 FID 改变。

## 2. 总体目标

目标不是继续降低某一个 seed 的 FID，而是得到同时满足以下条件的方法：

- 同一样本跨 timestep/noise 的语义表示更稳定；
- frozen representation 的类别、对象和空间结构更容易被读取；
- 不同 DiT block 保持互补，而不是都复制同一个 teacher target；
- backbone 仍保留生成所需的 timestep、noise 和 velocity 信息；
- 生成质量至少保持 U0 水平，不因过度 invariance 明显退化。

建议把下一方法暂称为 **Routed Multi-Level Paired REPA（RMPR）**。

## 3. 方案 A：分层、多尺度 teacher routing（最高优先级）

### 3.1 动机

当前 U0 使用共享 DINO target 建立跨轨迹语义坐标系，但多个 DiT block 若都趋向同一个高层 target，可能造成：

- 早层过早丢失局部纹理和空间细节；
- 中晚层学习重复特征，block CKA 偏高；
- teacher alignment 改善了生成，却没有充分利用网络深度形成层级表征。

### 3.2 方法

从 DINO 提取多个层级的 patch token：

- early teacher：局部纹理、边缘和位置关系；
- middle teacher：part/object structure；
- late teacher：全局语义和类别信息。

选择三个 DiT depth，例如 4/8/12，并建立 routing：

| DiT depth | Teacher target | 主要职责 |
|---:|---|---|
| 4 | DINO early/intermediate layer | 局部空间结构与 patch correspondence |
| 8 | DINO middle layer | object/part semantics 与跨噪声稳定性 |
| 12 | DINO late layer | 全局语义与生成条件 |

每个 depth 使用独立 projector：

\[
L_{route}=\sum_l \lambda_l D(P_l(h_l(x_t)), z_{r(l)}(x)).
\]

两条 trajectory view 在相同 depth 上共享同一个 clean-image teacher target，但不同 depth 不再共享完全相同的 teacher representation。

### 3.3 TODO

- [ ] 扩展 encoder 输出，使 DINO 可返回明确指定的多个 block token。
- [ ] 将 `z_dims`/projector 配置从单 target 扩展为 `student_depth -> teacher_depth` mapping。
- [ ] 在 SiT forward 中收集指定 student block 的 hidden token。
- [ ] 为每个 routing pair 建立独立 projector 和 loss 权重。
- [ ] 在 checkpoint args 中保存完整 routing，保证 generate/analysis 可重建模型。
- [ ] 日志中分别记录每个 depth 的 alignment loss、cosine similarity 和 feature std。
- [ ] 增加配置合法性检查：depth 数量、teacher 输出数量和 projector shape 必须匹配。
- [ ] 增加模型加载、forward、backward、checkpoint round-trip 测试。

## 4. 方案 B：低秩 semantic readout，而非固定 hidden split

### 4.1 动机

Jobs 106--111 的固定 128 维 semantic/state 分支没有明显收益。问题可能不是 semantic capacity 太小或 loss 太弱，而是人为要求所有表示都经过固定二分。U0 更可能在完整 hidden space 内自动分配语义和动态信息。

### 4.2 方法

保留完整 hidden state，只增加训练期低秩 readout：

\[
s_l=P_l(h_l),\quad \dim(s_l) \ll \dim(h_l).
\]

仅 `s_l` 接受 teacher alignment；原始 `h_l` 和 denoising path 不接受 semantic injection，不进行固定 state split。projector 是可读取语义子空间，而不是改变主干信息流的 bottleneck。

推荐先固定 128 维，不立即做大量维度搜索。只有发现 effective rank 被容量截断时，再比较 64/128/256。

### 4.3 TODO

- [ ] 新增 routed semantic readout，确保不向主干注入 feature。
- [ ] 移除该路线中的 semantic FiLM、recomposition decoder 和固定 state projector。
- [ ] 记录 readout rank、feature std 和 teacher/student covariance spectrum。
- [ ] 确认关闭 readout 时模型初始化和历史 U0 完全一致。
- [ ] 增加梯度测试：projector 与对应 backbone block 有梯度，sampling graph 不依赖 projector。

## 5. 方案 C：跨 block 互补性约束

### 5.1 动机

trajectory invariance 要求同一个 block 对不同 timestep/noise 稳定；但跨 block 不应全部相同。需要同时实现：

- view 维度：invariant；
- depth 维度：diverse/complementary。

### 5.2 候选实现

优先使用弱约束，避免重现强制分解对生成的负面影响：

1. 对不同 block 的 semantic readout 做 centered cross-covariance penalty；
2. 对跨 block CKA 设置上界 margin，而不是无条件最小化；
3. 对每个 block 保持 variance/effective-rank 下界；
4. teacher routing 已不同时，先观察自然多样性，再决定是否增加显式 diversity loss。

建议 loss：

\[
L_{div}=\sum_{i<j}\max(0,\operatorname{CKA}(s_i,s_j)-m)^2.
\]

不要直接最大化 block 差异，否则可能鼓励噪声或破坏语义连续性。

### 5.3 TODO

- [ ] 实现 detached/低成本 minibatch linear CKA 或 cross-covariance 版本。
- [ ] 默认只在 routing block 之间计算，不覆盖所有 block。
- [ ] 增加 variance floor，防止通过塌缩降低 CKA。
- [ ] 记录每对 block 的 CKA、effective rank 和 diversity loss active fraction。
- [ ] 先运行无显式 diversity 的 routed baseline，再决定是否启用该项。

## 6. 方案 D：动态信息充分性保护

### 6.1 动机

更强 invariance 可能损害 timestep/noise/velocity 信息，而这些信息对 diffusion/flow denoising 必不可少。之前的 state bottleneck 太强，可以改为训练期 diagnostic probe。

### 6.2 方法

在 semantic readout 的正交残差或完整 hidden 上增加轻量 probe：

- timestep regression/classification；
- velocity prediction；
- noise realization discrimination，仅作为诊断；
- probe 不注入 sampling path。

优先使用 stop-gradient probe 评估可读性；只有确认 routed REPA 明显破坏动态信息时，才让 state-sufficiency loss 反向传播到 backbone。

### 6.3 TODO

- [ ] 实现 frozen-feature online probes，区分 diagnostic 与 training loss 模式。
- [ ] 日志分别记录 semantic readout 和 residual/full-hidden 的 timestep accuracy。
- [ ] 比较 U0 与 routed method 的 velocity/timestep 可读性。
- [ ] 设置启用条件：只有动态 probe 相对 U0 明显下降时才加入保护 loss。

## 7. 方案 E：按轨迹难度自适应的 alignment

### 7.1 动机

低噪声状态可以恢复 patch-level structure，高噪声状态通常只能可靠恢复全局语义。所有 timestep 使用相同 target、相同权重可能造成不合理监督。

### 7.2 方法候选

- 低噪声：early/middle/late teacher 全部启用，强调 spatial correspondence；
- 中噪声：middle/late target；
- 高噪声：只保留 pooled/global semantic target；
- 权重由 SNR、teacher/student similarity 或在线 retrieval confidence 决定。

第一版使用确定性 SNR schedule，避免一开始引入不稳定的 learned gate：

\[
\lambda_l(t)=\lambda_l^{max} q_l(\operatorname{SNR}(t)).
\]

### 7.3 TODO

- [ ] 定义每个 teacher level 的 timestep/SNR 有效区间。
- [ ] 记录每层实际平均 loss weight，避免 schedule 名义启用但实际接近零。
- [ ] 对比 uniform routing 与 SNR-aware routing。
- [ ] 后续再评估 confidence-gated weighting，不与第一轮同时引入。

## 8. 最小实验矩阵

不优先做多 seed；每个实验回答一个方法问题。

| ID | 方法 | 相对 U0 的唯一主要变化 | 要回答的问题 |
|---|---|---|---|
| R0 | U0 reproduction/analysis | 无 | 建立相同 checkpoint 与直接表征指标基线 |
| R1 | Multi-level routing | 不同 student block 对齐不同 DINO depth | 分层 target 是否提升语义层级与 block specialization |
| R2 | R1 + low-rank readout | teacher 只约束低秩 semantic readout | 是否能保持生成动态并提高 invariant representation |
| R3 | R2 + weak block diversity | CKA margin/弱 cross-covariance | 是否减少跨层同质化而不损伤 FID |
| R4 | R2 + SNR-aware routing | 不同噪声等级使用不同 target/weight | 是否减少高噪声过强监督 |
| R5 | 最佳 R3/R4 + state protection | 仅在 probe 显示退化时加入 | 是否恢复 timestep/velocity 信息 |

不建议第一轮把 R1--R5 全部叠加。严格按顺序推进，前一个实验没有机制证据时不进入更复杂版本。

## 9. 评价协议

### 9.1 生成兼容性

- FID、sFID、IS、Precision、Recall；
- 相同 400k、50K samples、SDE 250 steps、CFG 1.0；
- U0 seed0 作为主要方法控制，J10 作为当前最佳单 run 参考。

### 9.2 直接表征能力

- frozen ImageNet/class linear probe；
- 同源跨 timestep retrieval；
- 同源跨 independent noise retrieval；
- token/patch spatial correspondence；
- source/timestep/noise variance decomposition；
- 每层 effective rank；
- layer CKA 与 timestep CKA；
- semantic readout 与 full hidden 的对比。

### 9.3 动态充分性

- timestep probe；
- velocity probe；
- 高噪声条件下 denoising loss；
- 不同 timestep bucket 的生成/重建误差。

## 10. 决策门槛

一个方案只有同时满足以下条件才进入下一阶段：

1. class probe 或跨噪声 retrieval 至少一项明确优于 U0；
2. 改善不是以 effective-rank collapse 或 block CKA 上升为代价；
3. timestep/velocity probe 不出现明显退化；
4. FID 不比 U0 seed0 差超过约 0.3；
5. mechanism metric 与设计方向一致，例如 routing 后不同层确实匹配不同 teacher level。

停止条件：

- 只有 FID 小幅波动，所有直接表征指标不变；
- probe 改善但 FID 和动态充分性明显退化；
- 多个 block CKA 继续升高，说明 routing 没有形成职责分工；
- 需要不断增加 loss 才能维持训练稳定，方法复杂度超过收益。

## 11. 工程 TODO 清单

### P0：先完成现有证据闭环

- [x] Jobs 109/110 已完成，并已记录六组 400k/50K 指标。
- [ ] 对 106--111 运行 `run_representation_106_111_by_rank.sh`。
- [ ] 对 U0、106、108、111 生成统一 representation report。
- [ ] 确认当前分析 exporter 能正确重建所有对应 checkpoint。
- [ ] 将 probe/retrieval/effective-rank/CKA 写入实验报告。

### P1：实现 routed REPA 基础设施

- [ ] 设计 CLI：student depths、teacher depths、每层维度和权重。
- [ ] 修改 DINO feature extraction，返回多层 token。
- [ ] 修改 SiT feature collection 与 projector registration。
- [ ] 修改 REPA loss，支持 mapping 和分层日志。
- [ ] 修改 checkpoint/generate/analysis 重建逻辑。
- [ ] 添加单元测试与 dry-run 配置验证。

### P2：实现直接表征评估

- [ ] 导出每个 routed block 的 full hidden 与 semantic readout。
- [ ] 报告 per-layer probe、retrieval、variance、rank、CKA。
- [ ] 增加跨方法汇总脚本，自动生成 U0/R1/R2/R3 对比表。
- [ ] 将训练日志诊断与离线 representation report 分开，避免用 minibatch loss 代替能力评估。

### P3：逐项消融

- [ ] R1：routing only。
- [ ] R2：routing + low-rank readout。
- [ ] 根据 R1/R2 的 CKA 决定是否运行 R3。
- [ ] 根据 timestep bucket 结果决定是否运行 R4。
- [ ] 根据动态 probe 是否退化决定是否运行 R5。

## 12. 风险与注意事项

- DINO 层级不天然等价于“低层/中层/高层语义”，必须通过 correspondence/probe 验证。
- CKA 降低不一定代表更好，可能只是噪声增加；必须同时看 probe 和 effective rank。
- 高噪声 global target 可能鼓励类别先验而非实例语义，应使用 instance retrieval 检查。
- projector 过强可能自己拟合 teacher，而 backbone 没有改善；需要比较 projector 前后的 probe。
- 在线 probe 可能成为新的训练捷径，默认只做 diagnostic。
- 不能仅凭 FID 判断 representation learning；FID 是必要的生成兼容性约束，不是充分证据。

## 13. 推荐的近期动作

近期最值得做的不是立即提交更多训练任务，而是：

1. 完成 106--111 和 U0 的统一 representation evaluation；
2. 根据 layer CKA、effective rank 和跨噪声 retrieval 选择 routing depth；
3. 只实现 R1，并确认不同层确实获得不同职责；
4. R1 有机制收益后再实现 low-rank readout R2；
5. diversity、SNR-aware weighting 和 state protection 均作为有证据触发的后续项，而不是一次性叠加。

这一顺序可以避免再次出现“增加多个 loss、FID 基本不变、但无法判断 representation 是否真正改善”的情况。
