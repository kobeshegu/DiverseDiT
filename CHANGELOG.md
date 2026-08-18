# Changelog

## [Unreleased] - Contextual Depth-JEPA

The same-timestep Masked JEPA experiment reached FID 11.53, improving over the
11.75 hierarchical result but still showing a limited gain over pure SiT. The
new `contextual_jepa` objective makes the prediction task less local and
reduces late-stage interference with denoising.

- Replaced independent random token masking with four compact spatial target
  regions covering exactly 50% of patch tokens.
- Added independent predictors for `student block 8 -> EMA block 12` and
  `student block 12 -> EMA block 12`; no predictor parameters exist on the
  teacher path.
- Replaced the narrow `0.5 +/- 0.1` timestep window with stratified sampling
  over `[0.2, 0.8]`, while retaining the same timestep and noise for student
  and teacher.
- Increased auxiliary coverage to every optimizer step and half of each local
  batch.
- Added cosine loss-weight decay from step 250k to 400k. The auxiliary branch
  is skipped after its weight reaches zero, leaving the final 50k steps for
  denoising-only refinement.
- Added timestep range, decay scale, and per-depth-pair cosine diagnostics.

This remains distinct from Self-Flow because it predicts contextual hidden
features within one noisy state and never constructs a cross-timestep target.
REPA remains disabled in the default experiment.

Validation covers Python and shell syntax, exact and spatially compact masks,
stratified timestep bounds, decay endpoints, independent predictor gradients,
and legacy CLI parsing. A real-data one-step SiT-B/2 GPU smoke test completed
with REPA disabled, loss `1.6442`, auxiliary loss `2.0182`, and finite gradient
norm `1.7025`.

## [Unreleased] - Same-Timestep Masked Representation Prediction

The hierarchical contrast experiment reached FID 11.7563, effectively tied
with Patch InfoNCE at 11.7906. This suggests that stronger cross-timestep
contrast has saturated, so the next experiment changes the learning signal
rather than increasing its contrastive weight.

Added the `masked_jepa` objective. The online SiT receives a 40% patch-token
mask, while the EMA SiT sees the same noisy latent at the same timestep. A
shared lightweight predictor learns the unmasked EMA backbone features from
blocks 8 and 12. The loss combines masked-patch cosine prediction with small
relational-similarity and variance terms, and introduces no negatives or
trainable teacher projector.

This is distinct from Self-Flow: it does not predict between diffusion
timesteps, construct a flow trajectory target, or align different noise
states. It improves hidden representations by recovering spatial context
within one noise state. The no-REPA setup remains unchanged for a controlled
comparison with pure SiT and Self-Flow.

The default `scripts/train.sh` experiment now runs this objective with loss
coefficient `0.05`, frequency `2`, batch ratio `0.25`, a 10k-step warmup,
timestep `0.5 +/- 0.1`, and no resume checkpoint. New diagnostics report
masked/unmasked cosine, relational and variance losses, feature standard
deviation, effective rank, and per-depth masked cosine.

Python compilation, shell syntax, whitespace, mask/loss gradient tests, and a
tiny multi-depth SiT integration test pass. A one-step SiT-B/2 GPU smoke test
on the configured dataset completed with REPA disabled, finite loss `1.6442`,
trajectory loss `1.9440`, and gradient norm `1.7064`.

## [Unreleased] — Patch-Level Trajectory Prediction

Replaced the experiment configuration's pooled trajectory-DINO objective with a
patch-level student/teacher prediction objective. The pooled DINO experiment
reached FID 12.77 and saturated (`traj_pos_cos` approximately 0.9997) without
improving over the pure SiT baseline (FID approximately 12), motivating an
objective that preserves spatial correspondence and explicitly resists collapse.

### Cross-Noise Patch InfoNCE

Added `patch_infonce` after the positive-only patch experiment reached FID
12.0095 but its positive/negative cosine gap contracted from 0.21 at 50k–100k
steps to 0.055 at 400k–450k steps. The new objective treats corresponding
high/mid-noise student and low-noise EMA teacher patches as positives and uses
patches from other images as explicit negatives.

The implementation samples a bounded number of patches per image, gathers EMA
keys across distributed workers, excludes same-image keys, and filters teacher
keys that are too similar to the positive. A minimum-negative fallback prevents
the similarity filter from removing the full denominator. InfoNCE is normalized
by the effective negative count so its scale remains stable across batch sizes.

### Noise-Aware Hierarchical Contrast

The Patch InfoNCE experiment improved FID from 12.0095 to 11.7906, but its
positive cosine fell from 0.609 at 10k–50k steps to 0.331 at 400k–450k steps.
The new `hierarchical` objective avoids requiring the high-noise representation
to recover exact low-noise patch locations:

```
mid noise  (t=0.50) -> low noise (t=0.15): patch InfoNCE + positive cosine
high noise (t=0.85) -> low noise (t=0.15): pooled image-level InfoNCE
```

The two losses use separate diagnostics and temperatures. The trajectory head
also has a configurable EMA decay, allowing it to track the online projector
more closely without changing the SiT backbone EMA.

### Added

- `TrajectoryPatchEncoder`, with a per-patch projector, timestep conditioning,
  and an online-only predictor.
- `trajectory_patch_loss`, combining patch-wise cosine prediction with
  image-level variance and covariance regularization.
- `trajectory_patch_infonce_loss`, with distributed negatives, patch sampling,
  false-negative filtering, and normalized loss scaling.
- `trajectory_hierarchical_contrastive_loss`, combining mid-noise patch
  contrast with high-noise global contrast.
- `--traj-objective={patch,patch_infonce,hierarchical}` and the configurable
  `--traj-patch-{sim,std,cov}-coeff` loss weights.
- `--traj-global-nce-coeff`, `--traj-patch-positive-coeff`,
  `--traj-global-temperature`, and `--traj-ema-decay` controls.
- `traj_patch_sim`, `traj_patch_std`, `traj_patch_cov`, `traj_patch_nce`,
  `traj_patch_nce_raw`, and `traj_patch_valid_negatives` diagnostics.
- Separate `traj_mid_patch_*` and `traj_high_global_*` diagnostics.

### Training Behavior

The online SiT processes the high- and mid-noise anchors (`0.85`, `0.50`), while
the EMA SiT processes only the low-noise anchor (`0.15`). Hierarchical training
aligns corresponding patches only for the mid-noise state and uses pooled
image-level representations for the high-noise state. Legacy patch, DINO,
VICReg, and InfoNCE objectives remain available.

The default experiment in `scripts/train.sh` is a no-REPA SiT-B/2 hierarchical
comparison: `enc_type=none`, `proj_coeff=0`, trajectory coefficient `0.05` with
a 10k-step warmup, frequency `2`, and batch ratio `0.25`. It uses patch/global
temperatures of `0.2`, patch/global coefficients of `1.0`/`0.5`, positive
cosine coefficient `0.25`, variance coefficient `0.1`, and trajectory head EMA
decay `0.999`. The script trains for 450k steps, generates 50k samples, and
evaluates IS, FID, sFID, precision, and recall.

### Validation

- Python compilation, shell syntax, and whitespace checks pass.
- Patch alignment and Patch InfoNCE unit tests confirm finite gradients,
  correct other-image negative counts, and lower loss for aligned positives.
- Tiny SiT integration confirms backbone gradients, two student trajectory
  steps, one EMA teacher step, and no REPA projectors.
- A one-step Patch InfoNCE GPU smoke test passed with the real dataset and VAE:
  normalized NCE `1.0042`, raw NCE `3.8655`, and `45.97` effective negatives
  out of a maximum of 48 in the reduced smoke batch.
- A one-step hierarchical GPU smoke test passed with normalized mid-patch NCE
  `0.9989` and high-global NCE `1.0006`, both at their expected random baseline.
- Legacy DINO, VICReg, and InfoNCE forward/backward compatibility was verified.

---

## [Unreleased] — Trajectory-DINO for Flow

Implemented a trajectory-level self-supervised auxiliary branch for SiT/REPA training. The new path treats ordered high/mid/low denoising states as a diffusion trajectory view and trains a lightweight temporal encoder with an EMA teacher. Added Semantic-Emergence Guided Sampling (SEGS), an adaptive sampler that uses EMA teacher feature velocity to bias trajectory timesteps toward semantic transition regions.

> **Backward compatibility**: trajectory training is off by default. Existing training commands keep the original flow loss and REPA projection behavior unless `--traj-loss` is enabled.

---

### New Files

| File | Description |
|------|-------------|
| `models/trajectory.py` | Trajectory timestep sampler, SEGS adaptive sampling state, trajectory interpolation helpers, temporal trajectory encoder, and DINO/VICReg/InfoNCE trajectory losses |

### Modified Files

| File | Summary |
|------|---------|
| `models/sit.py` | Added `return_features`, `feature_depth`, and `feature_depths` support for extracting hidden states from selected SiT blocks |
| `train.py` | Added the optional trajectory SSL branch, EMA trajectory encoder, checkpoint support, trajectory logging, and CLI flags |

---

### Method Summary

The default trajectory branch samples two positive diffusion paths for the same image:

```
View A: h(t_high), h(t_mid), h(t_low)  from student SiT
View B: h(t_high'), h(t_mid'), h(t_low') from EMA teacher SiT
```

Each path uses a high/mid/low schedule with jittered anchors. By default, all timesteps inside one path share the same noise tensor, while the two positive paths use independent noise. A small temporal `TrajectoryEncoder` pools patch tokens per timestep, adds timestep embeddings, and encodes the ordered trajectory into a single trajectory representation.

The standard flow branch is kept unchanged: it still samples timesteps from the original training distribution. The trajectory branch is auxiliary and is computed periodically on a sub-batch to keep compute overhead controlled.

### Semantic-Emergence Guided Sampling

SEGS maintains an EMA score over timestep bins. After each trajectory teacher forward, it pools EMA teacher hidden states and computes adjacent feature velocity:

```
velocity(t_k, t_{k+1}) = 1 - cosine(pool(h_ema(t_k)), pool(h_ema(t_{k+1})))
```

The velocity is accumulated into bins and later mixed with stage-uniform sampling:

```
p(bin) = (1 - mix) * p_stage(bin) + mix * softmax(score(bin) / temperature)
```

This makes the sampler compute-neutral: it reuses teacher features already produced for the trajectory loss and does not add extra forward passes.

---

### Main CLI Flags

```
--traj-loss                              Enable trajectory SSL branch
--traj-loss-coeff FLOAT [0.05]           Trajectory loss coefficient
--traj-warmup-steps INT [10000]          Linear warmup for trajectory loss
--traj-loss-frequency INT [8]            Compute trajectory branch every N steps
--traj-batch-ratio FLOAT [0.25]          Local batch fraction used for trajectory branch
--traj-num-steps INT [3]                 Number of timesteps in each trajectory
--traj-anchors STR [0.85,0.50,0.15]      High/mid/low timestep anchors
--traj-base-jitter FLOAT [0.10]          Per-image shared schedule jitter
--traj-view-jitter FLOAT [0.03]          Per-view schedule jitter
--traj-min-gap FLOAT [0.12]              Minimum gap between ordered timesteps
--traj-sampler {jittered,semantic}       Static jittered anchors or SEGS
--traj-semantic-bins INT [32]            Number of SEGS timestep bins
--traj-semantic-mix FLOAT [0.5]          Mixture weight for semantic probabilities
--traj-semantic-temperature FLOAT [0.2]  Softmax temperature for semantic scores
--traj-semantic-momentum FLOAT [0.95]    EMA momentum for bin scores
--traj-semantic-warmup-steps INT [10000] Warmup before using SEGS
--traj-semantic-min-t FLOAT [0.05]       Lower SEGS timestep bound
--traj-semantic-max-t FLOAT [0.95]       Upper SEGS timestep bound
--traj-depth INT [8]                     1-based SiT block used for trajectory features
--traj-objective {dino,vicreg,infonce}   Trajectory SSL objective
```

Recommended first smoke configuration:

```bash
--traj-loss \
--traj-objective dino \
--traj-loss-coeff 0.05 \
--traj-loss-frequency 8 \
--traj-batch-ratio 0.25 \
--traj-depth 8
```

Recommended SEGS ablation:

```bash
--traj-loss \
--traj-objective dino \
--traj-sampler semantic \
--traj-semantic-warmup-steps 10000 \
--traj-semantic-mix 0.5
```

---

### Validation

- `python -m py_compile train.py models/sit.py models/trajectory.py loss.py` passes.
- A real torch forward smoke test was not run in the current shell because `torch` is not installed in that Python environment.

---

## [Unreleased] — Block Diversity Enhancement

A comprehensive set of **architectural** and **loss-based** methods to maximise representation diversity across transformer blocks, while preserving training stability.

> **Backward compatibility**: all new features are off by default. Existing experiments reproduce without changes.

---

### New Files

| File | Description |
|------|-------------|
| `models/layer_drop.py` | Stochastic Depth (Layer Drop) module with three drop strategies and two drop types |

### Modified Files

| File | Summary |
|------|---------|
| `models/sit_2.py` | SiTBlock heterogeneity + 6 new architectural diversity mechanisms |
| `models/sit.py` | Accept new kwargs for forward compatibility (features ignored in basic model) |
| `loss.py` | 3 new diversity losses + auxiliary head loss |
| `train.py` | CLI arguments, diversity warmup, training loop integration |

---

### Architectural Methods (models/sit_2.py)

#### 1A. Per-Block Residual Scaling (`--residual-scaling`)

Each block gets a **learnable scalar** `residual_scale` that multiplies both attention and MLP residual outputs:

```
x = x + scale * gate_msa * attn(...)
x = x + scale * gate_mlp * mlp(...)
```

- **Init**: linearly decreasing 1.0 (shallow) → 0.5 (deep), breaking the symmetry where all blocks "wake up" from identity at the same rate.
- **Stability**: clamped to `[0.01, 2.0]` so no block can vanish or dominate.
- **Why it works**: different optimisation dynamics per block → different features emerge naturally without any explicit loss.

#### 1B. Block-Group Conditioning Transform (`--block-group-conditioning`)

Replaces the shared conditioning `c = t_embed + y` with per-group affine transforms:

```
c_block = cond_transforms[group_id](c)   # Linear → SiLU per group
```

- Divides blocks into `--num-cond-groups` groups (default 4). E.g. for depth=28: blocks [0-6], [7-13], [14-20], [21-27].
- Each group sees a **rotated** view of the conditioning space, so different groups naturally attend to different aspects of timestep / class.
- Strictly stronger than the simple per-block offset (`--per-block-conditioning`), which can only translate. The two are mutually exclusive; group conditioning takes priority.

#### 2A. Heterogeneous MLP Ratio (`--heterogeneous-mlp`)

Assigns varying MLP expansion ratios across depth:

```
Shallow blocks: ratio ≈ 4.0  (broad, general features)
Deep blocks:    ratio ≈ 3.0  (focused, refined features)
+ periodic ±0.5 variation with period 4
```

- Floor at 2.0. Total parameter count stays roughly the same as uniform ratio=4.0.
- Different MLP capacity → different "expressiveness bottleneck" → each block learns features at a different granularity.

#### 2B. Alternating Attention Heads (`--alternating-heads`)

Alternates head counts across blocks:

```
Even blocks:  16 heads × 72  head_dim  →  fine-grained attention
Odd blocks:    8 heads × 144 head_dim  →  coarse-grained attention
```

- QKV parameter count is **identical**; only the partitioning changes.
- Forces the network to alternate between different attention granularities, creating structural diversity.

#### 2C. Depth-Aware Initialization (`--depth-aware-init`)

After standard xavier + adaLN-zero init, scale attention/MLP weights by a depth factor:

```
depth_scale = 1.0 → 0.7  (linearly decreasing)
```

- adaLN modulation stays zero-init, preserving the initial identity property.
- Deeper blocks start with smaller weights → smaller "step size" when adaLN gates open → different blocks diverge from identity at different speeds.

#### Gradient Isolation (`--gradient-isolation`)

Inserts gradient-scaling barriers between block groups via a custom `autograd.Function`:

- `--gradient-isolation-alpha 0.0`: full stop-gradient (each group optimises independently)
- `--gradient-isolation-alpha 0.5`: half-gradient (soft barrier)
- `--gradient-isolation-layers "9,18"`: barrier placement (default: 1/3 and 2/3 depth)

Forces each group to learn independently useful features rather than relying on gradient signals from later layers.

#### Block Shuffling (`--block-shuffling`)

During training, randomly permutes block execution order within fixed-size groups:

- `--block-shuffling-prob 0.1`: 10% of forward passes use shuffled order
- `--block-shuffling-group-size 4`: only shuffle within groups of 4 blocks

Forces each block to produce useful output regardless of its position in the sequence.

#### Per-Block Conditioning Offset (`--per-block-conditioning`)

Lightweight alternative to group conditioning: adds a learnable `(hidden_size,)` offset per block. Simpler but only translates the conditioning space (no rotation).

#### Block-wise Auxiliary Heads (`--block-aux-heads`)

Attaches lightweight prediction heads (`LayerNorm → Linear → SiLU → Linear`) at intermediate blocks:

- `--block-aux-head-layers "7,14,21"`: which blocks get heads (default: 1/4, 1/2, 3/4 depth)
- `--block-aux-head-coeff 0.1`: loss weight

Each head predicts the denoising target from that block's features, providing **deep supervision** that structurally forces different blocks to decode different aspects of the signal.

---

### Loss-Based Methods (loss.py)

#### Block Contrastive Loss / InfoNCE (`--block-contrastive-loss`)

Treats each block's global-pooled representation as an embedding. All other blocks serve as negatives in an InfoNCE objective:

```
loss = mean_i [ logsumexp( sim(block_i, block_j) / tau ) ]   for j ≠ i
```

- `--block-contrastive-temperature 0.1`: InfoNCE temperature
- `--block-contrastive-loss-coeff 0.01`: loss weight
- Stronger than pairwise cosine similarity because all negatives are contrasted simultaneously.

#### Barlow Twins Cross-Correlation Loss (`--block-barlow-twins-loss`)

For each block pair, computes the cross-correlation matrix `C = z_a^T z_b / N` of batch-normalised representations:

- **Target**: zero matrix (not identity, because we want **de-correlation** between blocks)
- Diagonal: penalises same-dimension correlation across blocks
- Off-diagonal: penalises cross-dimension correlation (weighted by `--block-barlow-lambda 0.005`)
- Sampled over max 10 block pairs for efficiency.

#### VICReg Diversity Loss (`--block-vicreg-loss`)

Three complementary terms:

| Term | What it does | Weight flag |
|------|-------------|-------------|
| **Variance** | Hinge loss ensuring each block's features have std ≥ 1 (prevents collapse) | `--block-vicreg-lambda 25.0` |
| **Invariance** (inverted) | Minimises cosine similarity between block pairs (pushes blocks apart) | `--block-vicreg-mu 25.0` |
| **Covariance** | Decorrelates feature dimensions within each block (efficient capacity use) | `--block-vicreg-nu 1.0` |

Overall coefficient: `--block-vicreg-loss-coeff 0.01`

---

### Stochastic Depth / Layer Drop (models/layer_drop.py)

Enable with `--layer-drop`. Three drop probability strategies:

| Strategy | Schedule | Best for |
|----------|----------|----------|
| `uniform` | All layers have the same drop rate | Baseline |
| `linear` | 0 (shallow) → `drop_rate` (deep) | **Recommended**: forces deep layers to be self-sufficient |
| `cosine` | Cosine curve: gentle start, steep middle, gentle end | Smooth alternative |

Two drop types:

| Type | Behaviour |
|------|-----------|
| `random` | Per-sample independent Bernoulli (stronger regularisation) |
| `batch` | Whole batch shares the same drop pattern (faster, less variance) |

Safety: `min_keep_layers = depth // 2` ensures at least half the layers always execute.

```
--layer-drop --layer-drop-rate 0.15 --layer-drop-strategy linear --layer-drop-type random
```

---

### Diversity Warmup (train.py)

All diversity losses are multiplied by a **linear warmup** coefficient:

```
warmup(step) = min(1.0, step / warmup_steps)
```

- `--diversity-warmup-steps 10000` (default)
- Allows the network to learn basic denoising ability before diversity pressure kicks in.
- Applies to: `block_diversity_loss`, `block_contrastive_loss`, `block_barlow_twins_loss`, `block_vicreg_loss`, `block_aux_loss`.
- Logged as `diversity_warmup` in wandb for monitoring.

---

### Recommended Configurations

**Minimal (safe, high-impact)**:
```bash
--residual-scaling \
--block-group-conditioning --num-cond-groups 4 \
--depth-aware-init \
--diversity-warmup-steps 10000
```

**Full architectural heterogeneity**:
```bash
--residual-scaling \
--block-group-conditioning --num-cond-groups 4 \
--heterogeneous-mlp \
--alternating-heads \
--depth-aware-init \
--block-diversity-loss \
--diversity-warmup-steps 10000
```

**Everything (architecture + loss + regularisation)**:
```bash
--residual-scaling \
--block-group-conditioning --num-cond-groups 4 \
--heterogeneous-mlp \
--alternating-heads \
--depth-aware-init \
--block-vicreg-loss --block-vicreg-loss-coeff 0.01 \
--layer-drop --layer-drop-rate 0.15 --layer-drop-strategy linear \
--block-aux-heads --block-aux-head-layers "7,14,21" \
--diversity-warmup-steps 10000
```

---

### Full CLI Reference (new flags only)

```
# Architectural heterogeneity
--residual-scaling                      Per-block learnable residual scale
--block-group-conditioning              Per-group conditioning affine transform
--num-cond-groups INT [4]               Number of conditioning groups
--heterogeneous-mlp                     Varying MLP ratio across blocks
--alternating-heads                     Alternate attention head counts
--depth-aware-init                      Depth-scaled weight init

# Regularisation
--layer-drop                            Enable stochastic depth
--layer-drop-rate FLOAT [0.1]           Max drop probability
--layer-drop-strategy {uniform,linear,cosine} [linear]
--layer-drop-type {random,batch} [random]
--gradient-isolation                    Gradient barriers between groups
--gradient-isolation-alpha FLOAT [0.0]  Gradient scale (0=full stop)
--gradient-isolation-layers STR         Barrier positions, e.g. "9,18"
--block-shuffling                       Random block order shuffling
--block-shuffling-prob FLOAT [0.1]      Shuffle probability
--block-shuffling-group-size INT [4]    Group size for shuffling

# Conditioning
--per-block-conditioning                Learnable offset per block
--block-group-conditioning              (see above, takes priority)

# Auxiliary heads
--block-aux-heads                       Block-wise prediction heads
--block-aux-head-layers STR             Head positions, e.g. "7,14,21"
--block-aux-head-coeff FLOAT [0.1]      Aux head loss weight

# Diversity losses
--block-contrastive-loss                InfoNCE between blocks
--block-contrastive-loss-coeff FLOAT [0.01]
--block-contrastive-temperature FLOAT [0.1]
--block-barlow-twins-loss               Barlow Twins cross-correlation
--block-barlow-twins-loss-coeff FLOAT [0.01]
--block-barlow-lambda FLOAT [0.005]     Off-diagonal weight
--block-vicreg-loss                     VICReg diversity loss
--block-vicreg-loss-coeff FLOAT [0.01]
--block-vicreg-lambda FLOAT [25.0]      Variance weight
--block-vicreg-mu FLOAT [25.0]          Invariance weight
--block-vicreg-nu FLOAT [1.0]           Covariance weight

# Warmup
--diversity-warmup-steps INT [10000]    Linear warmup for diversity losses
```
