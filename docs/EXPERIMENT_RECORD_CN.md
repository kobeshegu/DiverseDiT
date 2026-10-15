# DiverseDiT / TFCR 实验总记录

> 更新日期：2026-10-12  
> 证据范围：当前仓库中的训练脚本、`results/*/args.json`、训练日志、`sampled_images/**/*_metrics.txt` 和 Git 历史。只有存在 50K 样本评测文件的结果才记为“已评测”；仅有脚本不等于已经运行。

## 1. 项目与当前研究问题

仓库的主项目是 **DiverseDiT**：通过长残差连接和跨 block 的 diversity loss 提升 DiT 内部表示多样性。当前分支在主项目之上开展 **TFCR（Trajectory Factorization via Cross-View Recomposition）** 研究，问题是：扩散轨迹中哪些表示应跨时间/噪声保持，哪些应随轨迹演化，以及这种分解是否能改善 ImageNet 生成质量。

当前实验主线已经从“显式 Persistent/Evolving 分解”逐步转向一个更强、也更简单的事实：**同一样本的双轨迹视图 + 外部 DINO REPA 目标（U0 paired REPA）是目前最可靠的方案**。显式分解、teacher-free 自蒸馏和动态 clean transfer 尚未证明能稳定叠加在 U0 之上。

## 2. 统一评测口径

- 模型/数据：SiT-B/2，ImageNet 256×256。
- 训练：通常 400k steps、batch size 256、fp16；特殊的 200k screening 单独说明。
- 采样：400k checkpoint、50K samples、SDE 250 steps、CFG 1.0、VAE `mse`。
- 指标：FID（越低越好）、sFID（越低越好）、IS/Precision/Recall（越高越好）。
- 无 REPA seed-0 基线：FID **35.900961**。
- 跨方法结论应使用相同 seed、采样设置和 external-target 条件；CFG 1.8 或 450k 的历史结果不能直接混入下表。

## 3. 已完成实验及其回答的问题

| 实验族 | 核心目的 | 已有证据 | 当前结论 |
|---|---|---|---|
| A0–A8 / Jobs 1–11 | 验证双视图、显式 invariance、Persistent/Evolving/recomposition、配对比例和噪声配对 | 主要配置已有 400k 评测 | ratio 从 0.5/0.75 提至 1.0 是主要增益来源；A5 FID 29.83，但只比 compute-matched A3 30.11 好 0.28，显式分解贡献较小 |
| Q / Jobs 12–21 | teacher-free 轨迹 orbit invariant subspace 能否替代 TFCR | Q0/Q2/Q3 已评测 | 未验证：Q0 33.68，Q2/Q3 34.48/34.42；复杂子空间目标反而退化 |
| A9–A14 / Jobs 24–29 | orbit consensus 与 adversarial nuisance purification 是否有效 | A9/A12/A13 已评测 | 未验证：A9 31.97，A12 32.32；shuffled-label A13 31.23 反而更好，收益更像普通正则而非正确 nuisance 因果信号 |
| V / Jobs 31–41 | clean consensus、可靠子空间、task-selective weighting（VGSC） | V1–V8 多数已评测 | 最好 V2 30.24，未超过 A3/A5；V5 shuffled source 仍达 30.28，且 Jobs 33/37/39 样本重复，因果归因不成立，需重采样 |
| S / Jobs 42–50 | shared target、自蒸馏、对比/关系目标能否改善 A5 | S1–S9 已评测 | 外部 DINO 的 S1 23.66 明显有效；teacher-free 最好 S3 30.18，未超过 A5 |
| T / Jobs 52–57 | antithetic source/evolution 与 native recomposition | 当前未见完整 50K 指标 | 仍是未验证设计，不能形成结果结论 |
| U / Jobs 58–65, 72–82 | paired REPA、选择性语义分解、系数/噪声/弱分解消融 | U0、U1/U2/U8 已评测 | U0 当前最强；U2 23.27 和 U8 23.92 均未超过 U0 23.04，说明额外显式分解没有带来新的 representation benefit |
| SF / Jobs 66–69 | EMA teacher/self-flow 能否替代外部编码器 | SF1–SF4 已评测 | 最好 SF2 29.89，接近 A5 但远弱于 U0；shuffled teacher 与正确 teacher 很接近，语义对应关系证据弱 |
| I / Jobs 86–89 | 隐式 pair interaction、随机投影和 BYOL 对齐 | 全部已评测 | I2 30.08 仅比 A3 好 0.03，不构成可靠提升；I0/I1 adapter 严重退化，I3 30.32 |
| J / Jobs 93–102 | dynamic clean/orbit/transport/self-guidance 能否替代或增强 U0 | 全部已有 400k 50K 指标 | teacher-free 最好 J4 30.14，且 shuffled 优于正确 target，说明该 objective 没有学到可靠语义对应；J10 22.60 表明 late clean loss 与 U0 兼容；J5 sampling guidance 崩溃至 FID 92.47 |
| K / Jobs 103–105 | 能否减少/移除外部 DINO supervision，并靠 late asymmetric clean transfer 接管 | 2026-10-10 新完成，均有 50K 指标 | 减少或移除 DINO 均有明显损失；K3 的渐进 dropout 优于单视图 K1，但仍弱于全程 paired REPA，clean transfer 尚不能替代外部目标 |
| M1 / Jobs 106–111 | semantic/state split、semantic-only alignment、identifiability constraints | 六组均已有 400k 50K 指标 | 全部结果没有超过对应强基线；显式 bottleneck、semantic-only alignment 与可辨识约束均未转化为明显生成增益，直接表征评估仍待完成 |

说明：编号 40/41、70/71、79、83–85、90–92 等是独立 eval/resume 包装脚本，不应被当成新的训练方法。仓库存在部分只有脚本或训练目录、没有 50K 指标的任务，本记录不据此推断其效果。

## 4. 核心可比结果

### 4.1 用于方法判断的代表性结果

| 方法 | Job | FID | sFID | IS | Precision | Recall | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| 无 REPA baseline | baseline | 35.900961 | 6.688123 | 41.130138 | 0.524620 | 0.644600 | 统一基线 |
| A3 双视图控制 | 30 | 30.107791 | 6.413116 | 50.217392 | 0.554620 | 0.648800 | 绝大部分 TFCR 增益来自密集双视图训练 |
| A5 完整 TFCR | 22 | 29.825559 | 6.331125 | 50.477547 | 0.555660 | 0.642900 | 比 A3 仅改善 0.282232 FID，当前分解作用有限 |
| 标准 A2 REPA | 51 | 28.154142 | 7.220941 | 54.345894 | 0.561460 | 0.647300 | 外部 teacher 提供语义坐标系 |
| U0 paired REPA | 65 | 23.043364 | 6.611714 | 63.543480 | 0.593080 | 0.646400 | 双轨迹共享外部语义 target，当前核心方法基线 |
| J10 U0 + late hard clean | 102 | 22.603269 | 6.643863 | **65.051247** | **0.595140** | 0.646400 | late clean objective 与语义锚定兼容，但尚未证明增强表征结构 |

这组结果用于回答方法问题：双轨迹本身有效，外部语义 target 带来最大增益，而现有显式分解与 self-clean target 没有展示出独立、可解释的 representation learning 能力。FID 只说明方法与生成任务兼容，后续必须用直接表征指标判断能力是否真正增强。

### 4.2 2026-10-10 新增但此前未入记录的 K 系列

| Job | 方法与要验证的假设 | FID | Δ vs baseline | sFID | IS | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 103 | K1：只给第一条视图 DINO REPA，另一视图靠 FM + 100k 后 hard clean transfer；验证单视图 teacher 能否转移到另一视图 | 23.787075 | 12.113886 | 6.684746 | 62.585499 | 0.589440 | 0.643200 |
| 104 | K2：完全移除 DINO，只保留 100k 后 hard clean asymmetric consistency；验证 clean transfer 能否独立替代 teacher | 30.344000 | 5.556961 | 6.527787 | 50.254082 | 0.551360 | **0.654100** |
| 105 | K3：先用 paired REPA 启动，100k–250k 将外部 target dropout 逐步升到 1，同时保留 late hard clean；验证能否“teacher 退场” | 23.413834 | 12.487127 | 6.566337 | 63.366932 | 0.590960 | 0.640700 |

K 系列的直接结论：

1. K2 比 U0 seed-0 差 **7.300636 FID**，说明 late clean consistency 单独只能回到 A3/A5 量级，不能替代 DINO 语义监督。
2. K1 比 U0 seed-0 差 **0.743711 FID**，表明只监督一条视图确实能发生一定转移，但完整 paired supervision 仍更好。
3. K3 比 K1 好 **0.373241 FID**，说明“先 paired teacher 启动、再渐退”优于从始至终只监督一条视图；但它仍比 U0 seed-0 差 **0.370470 FID**，也比 J10 差 **0.810565 FID**。
4. K2 的 Recall 最高，但 FID/IS/Precision 明显落后；这更像质量—覆盖率权衡，不能视为总体改善。

## 5. 已经得到的结论

### 从 representation learning 角度可以明确的结论

- 密集的同样本双轨迹训练有效；factor batch ratio=1.0 明显优于较低比例。
- paired REPA 是当前最有效的结构。它说明模型需要的不只是更多轨迹样本，而是一个跨噪声、跨时间仍稳定的语义坐标系。
- A3 双视图已经获得大部分 TFCR 增益，而 A5 显式 Persistent/Evolving 分解只增加很小收益。这说明当前分解 loss 主要在约束特征数值，还没有迫使两个分支学习功能上不同、不可互相替代的信息。
- teacher-free clean consistency、orbit subspace 和 EMA self-distillation 都停留在 FID 约 30。模型自己的 clean estimate 含有生成误差和低层状态信息，不是足够强的 semantic target。
- shuffled target 多次接近或优于正确 target，说明现有 self-target objective 容易退化为普通平滑/正则化，没有真正利用同一图像的语义对应关系。
- K1/K3 保留了部分 paired REPA 能力，但移除 DINO 的 K2 明显退化。当前方法学问题不是怎样更快丢掉 teacher，而是怎样把 teacher 提供的语义结构真正写入模型内部，并让这种结构与轨迹动态共存。

### 当前方法的根本缺口

- **只有 alignment，没有 semantic organization。** U0 把 hidden state 拉向 DINO，但没有规定哪些通道保存类别/结构、哪些通道服务于 timestep/noise/velocity。
- **只有 loss 命名，没有 branch identifiability。** Persistent 和 Evolving 仍可能互相复制，或者由一个分支携带全部信息。
- **跨轨迹一致与跨 block 多样性尚未统一。** REPA 倾向让若干 block 同质化；DiverseDiT 希望不同 block 专门化。当前方法没有同时控制这两个方向。
- **teacher dropout 不是 representation transfer。** 按训练步数关闭 target 只是在撤掉监督，并不能保证学生已经形成可独立维持的语义表征。
- **主要依据仍是生成指标。** 如果目标是提升 representation learning，就必须直接测量语义可分性、跨噪声检索、有效秩、block specialization 和分支因果作用。

## 6. 下一步方法：语义锚定的多样化轨迹表征

下一步不应继续堆叠 clean consistency 或更复杂的 pair adapter。建议以 U0 为起点，构造 **Semantic-Anchored Diverse Trajectory Representation（语义锚定的多样化轨迹表征）**。核心不是让两条轨迹的全部特征相同，而是让它们共享语义子空间，同时保留互补的状态子空间，并让不同 block 承担不同抽象层级。

### 6.1 表征结构

把每个选定 block 的表示拆成两个有明确功能的子空间：

```text
h_l = W_p p_l + W_e e_l

p_l: source-semantic，跨 timestep/noise 稳定
e_l: trajectory-state，保留 timestep/noise/velocity 所需信息
```

- `p_l` 对齐 DINO 的 token/patch representation，并做同源跨轨迹一致性；不要对齐完整 `h_l`。
- `e_l` 不做 invariance，而是预测 velocity residual、timestep/noise 或 clean residual，保证动态信息没有被语义一致性抹掉。
- 对 `p_l` 与 `e_l` 加 cross-covariance/orthogonality 约束，并限制各自维度，避免两个分支复制相同信息。
- 重构必须使用 `p_other + e_current`，同时加入 zero/swap intervention loss，使任一分支被置零或错误交换时任务性能明确下降。

### 6.2 把 DiverseDiT 的跨层多样性纳入语义对齐

不要让所有对齐 block 模仿同一个 DINO target。采用分层、分预算的 routing：

- 早层保留局部结构和高频空间信息；中层学习跨噪声稳定的 object/part semantics；晚层保留生成所需的全局条件与速度信息。
- 不同 block 对齐 teacher 的不同层或不同尺度，避免多层表示塌缩到同一方向。
- 在 `p_l` 上维持跨轨迹语义一致，在跨 block 维度上使用 diversity/effective-rank objective，鼓励互补而非重复。
- 长残差连接传递未被当前 block 消化的信息；每个 block 只读取受控的 semantic/state budget。

这会把项目原始的“block diversity”与当前分支的“trajectory invariance”统一起来：**同一 block 的语义子空间跨轨迹稳定，不同 block 的表征职责保持多样。**

### 6.3 将 teacher 知识内化，而不是按时间关闭 teacher

K3 的固定 schedule dropout 没有判断学生是否已经学会。更合理的方式是：

1. 用 DINO 建立 clean/low-noise semantic anchor；
2. 学习一个由 DiT 自身产生的 semantic memory/EMA prototype；
3. 只有当 student anchor 与 teacher 在 retrieval、token correspondence 和 effective rank 上达到门限时，才逐样本降低 teacher 权重；
4. teacher 降权后继续约束 student memory 的跨噪声一致性与类别间方差，防止语义塌缩。

因此 dropout 应由 representation quality 控制，而不是只由 step 控制。

## 7. 下一轮实验应回答的问题

### 阶段 M0：先确认缺失的 representation 能力

只分析现有 A3、A5、U0、J10、K2、K3 checkpoint，不先训练新模型：

- layer × timestep/noise CKA；
- source/timestep/noise variance decomposition；
- frozen ImageNet linear probe、跨 timestep/noise source retrieval；
- block 间 CKA、effective rank 和 token spatial correspondence；
- zero/swap Persistent/Evolving 或 aligned subspace 的因果干预。

目的不是重新排 FID 名次，而是定位 U0 相比 A3 究竟增加了哪一种表征能力，以及 K3 丢掉了什么。

### 阶段 M1：最小语义/状态双子空间

在 U0 上只增加三项：semantic bottleneck `p`、state residual `e`、cross-covariance penalty。首先验证：

- `p` 的跨噪声 retrieval 和 linear probe 高于完整 hidden state；
- `e` 的 timestep/velocity probe 高于 `p`；
- zero/swap 任一分支都会按预期破坏对应能力；
- block effective rank 不因 REPA 对齐而下降。

只有这些条件成立，才能声称方法提升了 representation learning，而不是增加了一个辅助 loss。

当前已实现第一版 M1：

- `--factor-semantic-state`：启用低维 semantic/state 双子空间；
- `--factor-semantic-conditioning`：有 DINO 时只把 semantic 分支投影到 teacher token，完整 hidden state 不做 REPA；
- `--factor-semantic-cross-covariance-coeff`：去除两个子空间的 centered cross-covariance；
- `--factor-semantic-variance-coeff`：防止任一分支坍缩；
- `--factor-semantic-intervention-coeff`：训练正确的 `semantic_other + state_current` 重组，并要求其优于 zero-semantic、zero-state 和 wrong-state 重组。

为分别归因方案 2、3、4，DLC 实验采用 3×2 矩阵：

| 方案 | REPA | no-REPA | 单独回答的问题 |
|---|---|---|---|
| 2：semantic/state split | Job 106 `106_p2_split_repa_400k.sh` | Job 107 `107_p2_split_no_repa_400k.sh` | 只增加双子空间与 balanced recomposition 是否有效；REPA 版仍对齐完整 hidden state |
| 3：semantic-only alignment | Job 108 `108_p3_semantic_align_repa_400k.sh` | Job 109 `109_p3_semantic_align_no_repa_400k.sh` | 把 alignment 从完整 hidden state 移到 semantic 分支是否更好；no-REPA 用同源跨轨迹一致性 |
| 4：identifiability constraints | Job 110 `110_p4_identifiable_repa_400k.sh` | Job 111 `111_p4_identifiable_no_repa_400k.sh` | cross-covariance、anti-collapse 与 zero/swap necessity 是否让分工真正可识别 |

128 维 bottleneck 从方案 2 开始就是 semantic/state split 的组成部分，不作为方案 4 的新增变量。六个实验使用相同 bottleneck、paired-view 设置、recomposition 权重和 warmup/decay。横向比较 REPA/no-REPA 衡量外部语义锚点的作用；纵向比较 2→3→4 分别只衡量 semantic-only alignment 与 identifiability 约束的增益。

六个 checkpoint 训练完成后，按 DLC rank 0–5 运行 `scripts/dlc_tfcr_jobs/run_representation_106_111_by_rank.sh`。统一报告 class/timestep probe、跨 timestep/noise retrieval、source/timestep/noise variance decomposition、semantic/state effective rank、layer/timestep CKA；方案 4 还要结合训练日志中的 zero-semantic、zero-state 和 wrong-state gap 判断两个分支是否具有因果必要性。FID/IS 只用于确认生成兼容性，不能单独作为表征能力提升的结论。

#### 2026-10-15 完整结果

评测口径均为 400k checkpoint、50K samples、SDE 250 steps、CFG 1.0。Jobs 106--111 六组实验均已完成。

| Job | 方案 | FID | sFID | IS | Precision | Recall | 对应比较 |
|---:|---|---:|---:|---:|---:|---:|---|
| 106 | P2 split + full-hidden REPA | 23.531836 | 6.457949 | 63.032265 | 0.589820 | 0.645800 | 比 U0 23.043364 差 0.488472 |
| 107 | P2 split, no REPA | 29.694973 | 6.303665 | 50.227348 | 0.554840 | 0.641900 | 比 A3 30.107791 好 0.412818；比 A5 29.825559 好 0.130586 |
| 108 | P3 semantic-only REPA | 23.789724 | 6.465170 | 62.758461 | 0.587420 | 0.644300 | 比 U0 差 0.746360，也比 106 差 0.257888 |
| 109 | P3 same-source semantic alignment, no REPA | 29.688769 | 6.343803 | 50.672741 | 0.556220 | 0.648300 | 比 107 仅好 0.006204，基本等价 |
| 110 | P4 identifiable + semantic-only REPA | 23.980545 | 6.554770 | 62.142963 | 0.589220 | 0.643600 | 比 108 差 0.190821；比 U0 差 0.937181 |
| 111 | P4 identifiable, no REPA | 29.637261 | 6.359669 | 50.226261 | 0.557240 | 0.650300 | 比 107 仅好 0.057712；比 A3 好 0.470530 |

当前可以下的结论：

1. **P2 没有形成新的主要增益来源。** no-REPA 的 0.13–0.41 FID 改善很小；加 REPA 后仍比 U0 差 0.49，说明 128 维 semantic/state readout 没有增强 paired REPA 的核心优势。
2. **P3 没有证明 semantic-only alignment 更有效。** REPA 侧 108 比 full-hidden REPA 的 106 差 0.26 FID；no-REPA 侧 109 与 107 只差 0.006。109 最后 minibatch 的 semantic std 约 13.10、cross-covariance 约 420，说明无 identifiability 约束的 semantic consistency 还存在明显的尺度膨胀和分支纠缠。
3. **P4 的 identifiability loss 改善了训练诊断，却没有改善最终性能。** 110/111 的 cross-covariance 分别降到约 0.0039/0.0017，zero-source/state/wrong-state gap 均为正；但 110 比 108 差 0.19 FID，111 比 107 只好 0.058。这只证明约束按预期改变了 readout，不证明 representation learning 能力提升。
4. **总体上，这一版三个方案没有明显改进。** 它们最多保持 A3/A5 或 U0 附近的生成质量，没有超过 U0/J10，也尚无 probe、retrieval、effective-rank 或 CKA 结果支持表征能力提升。后续不应继续只调这些 loss 的系数，应先完成直接表征评估，再决定是否转向分层 teacher routing 与 block diversity。

### 阶段 M2：分层 teacher routing + block diversity

比较单层统一对齐与“teacher 多层/多尺度 → DiT 不同 block”的 routing。目标是同时提高 semantic probe、跨轨迹稳定性和 block specialization，而非只优化 FID。

### 阶段 M3：自适应 teacher internalization

在 M1/M2 的表征已经可解释后，再做 quality-gated teacher decay。K1–K3 可作为对照，但不再继续尝试纯 step-based dropout 变体。

## 8. 方法成功标准

下一方法应同时满足：

1. **语义能力：** frozen linear probe、跨噪声 source retrieval、spatial correspondence 提升；
2. **动态充分性：** state 子空间仍能预测 timestep/velocity，生成质量不因过度 invariance 退化；
3. **分工可识别：** semantic/state probes 与 zero/swap interventions 呈现相反且符合设计的响应；
4. **跨层多样性：** off-diagonal block CKA 降低或 effective rank 提升，不出现多层同质化；
5. **生成兼容性：** FID/IS 作为最终约束，证明更好的 representation 没有牺牲生成能力。

## 9. 记录维护规则

- 新实验必须记录：Job、Git commit、完整命令/`args.json`、seed、checkpoint、采样配置、指标文件路径和样本 hash。
- 状态统一使用：`planned`、`training`、`trained-no-eval`、`evaluated`、`invalid/duplicate`。
- 结论必须区分“生成指标改善”“表征相关性”与“因果干预证明”，不能只凭 FID 给 representation learning 下结论。
- 主结果表只纳入同一评测口径；特殊 CFG、步数或 checkpoint 放入独立历史表。
- 权威数值总表见 `docs/TFCR_ABLATION_RESULTS.md`；方法设计分别见 `docs/TFCR_PROPOSAL.md`、`docs/PAIRED_SHARED_TARGET_EXPERIMENTS.md` 和 `docs/UNIFIED_INVARIANCE_ANALYSIS.md`。
- 下一阶段 representation learning 的方法候选、实验门槛和工程 TODO 见 `docs/NEXT_REPRESENTATION_LEARNING_TODO_CN.md`。
