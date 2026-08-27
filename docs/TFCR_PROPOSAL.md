# Learning What Stays and What Changes

## Trajectory Factorization via Cross-View Recomposition (TFCR)

**Working CVPR proposal — implementation branch:** `codex/persistent-evolving-recomposition`

## 1. One-sentence thesis

Existing representation-guided diffusion methods mainly teach a model what must remain stable across corruption views. TFCR additionally organizes what must change by factorizing each trajectory representation into a cross-view **Persistent** component and a view-specific **Evolving** component, defined operationally through cross-view recomposition.

The intended paper claim is not that an invariant/variant split is new. The claim is:

> Diffusion trajectories supply ordered, controllable paired views. Cross-view recomposition turns the information discarded by invariance-only alignment into explicit representation supervision, improving generation and revealing what alignment methods actually learn.

## 2. Motivation and unified view of prior methods

For a clean latent `x0`, rectified-flow training creates

```text
x_t = (1 - t) x0 + t epsilon.
```

This is a naturally parameterized augmentation family: views share a source sample, while `t`, corruption strength, and optionally the noise realization are controlled.

### REPA: externally specified persistence

REPA projects an intermediate noisy DiT feature toward a clean-image representation from a frozen visual encoder. Its direct supervision is therefore `information that should survive corruption`, with DINO-like semantics defining that information. Later layers remain free to model details, but evolving information is only indirectly shaped by the flow objective.

### SRA: internally distilled persistence

SRA replaces the external teacher with a stronger internal target: a deeper, lower-noise, EMA representation guides a shallower, higher-noise representation. The teacher changes, but the learning primitive remains alignment toward a more stable target.

### Self-Flow: heterogeneous-view training and noise-space augmentation

Self-Flow applies heterogeneous timesteps across tokens and attributes gains to information asymmetry. The subsequent *From SRA to Self-Flow* analysis reports that blocking cross-noise attention does not remove the gain, supporting a second mechanism: dual-timestep scheduling is also effective data augmentation along the noise dimension.

### The missing objective

All three strengthen access to shared/stable information. Flow matching guarantees that the final velocity is correct, but does not require intermediate view-specific information to be structured, complementary, non-collapsed, or predictably organized. This yields the central question:

> Can the information that changes across diffusion views become useful representation supervision rather than an implicit residual left to the output loss?

## 3. Hypotheses

- **H1 — persistence:** useful semantic/source information is shared across trajectory views and can be learned without assigning it a hand-designed label such as “structure.”
- **H2 — complementary evolution:** current-view information contains useful, sample-specific state beyond timestep identity; explicitly organizing it improves generation beyond invariance-only training.
- **H3 — recomposition:** a valid decomposition should support `Persistent(other view) + Evolving(current view) -> current deeper feature`.
- **H4 — ordering:** if the optional transition predictor helps, evolving codes form an ordered trajectory rather than arbitrary per-view residuals.
- **H5 — two axes of diversity:** DiverseDiT's cross-block diversity and TFCR's within-trajectory factor diversity are complementary.

## 4. Method

### 4.1 Paired trajectory views

For each source latent, sample two timesteps with a controlled gap:

```text
x_a = alpha(t_a) x0 + sigma(t_a) epsilon_a
x_b = alpha(t_b) x0 + sigma(t_b) epsilon_b
delta = |t_a - t_b| in [delta_min, delta_max].
```

Pairing is controllable:

- `same-epsilon`: `epsilon_a = epsilon_b`, isolating evolution along one trajectory;
- `cross-epsilon`: independent noise, forcing separation of source-shared and realization-specific information;
- `mixed`: samplewise mixture, recommended default `p(cross-epsilon)=0.5`.

Mixed pairing matters because same-noise-only training permits Persistent codes to leak trajectory-specific noise.

### 4.2 Factorization

At source block `m`, the shared SiT backbone produces `h_a^m` and `h_b^m`. Two lightweight tokenwise projectors produce

```text
p_a = P_p(h_a^m),  e_a = P_e(h_a^m)
p_b = P_p(h_b^m),  e_b = P_e(h_b^m).
```

No semantic/texture/structure assignment is hard-coded. The losses define the roles.

### 4.3 Persistent objective: learn what stays

```text
L_p = 0.5 * [1 - cos(p_a, sg(p_b)) + 1 - cos(p_b, sg(p_a))].
```

This is deliberately comparable to alignment methods and forms the invariance-only ablation.

### 4.4 Cross-view recomposition: learn what changes

A recomposer predicts a deeper feature at target block `n >= m`:

```text
h_hat_a = R(p_b, e_a)
h_hat_b = R(p_a, e_b)

L_r = NMSE(h_hat_a, sg(h_a^n)) + NMSE(h_hat_b, sg(h_b^n)).
```

Persistent codes are exchanged; evolving codes are not. Thus `p` must be exchangeable across views, while `e` must supply what is needed for the current state. The target is a stopped-gradient, layer-normalized feature rather than a pixel reconstruction, avoiding a second autoencoder objective.

The recomposer receives no raw timestep by default. This removes the easiest `e_t = MLP(t)` shortcut at the recomposition boundary. A compact factor dimension supplies a second information bottleneck.

### 4.5 Optional ordered transition

```text
e_hat_b = T(e_a, t_b - t_a)
e_hat_a = T(e_b, t_a - t_b)

L_t = d(e_hat_b, sg(e_b)) + d(e_hat_a, sg(e_a)).
```

This is an enhancement, not the minimal method. It is promoted to the main model only if TFCR without it already beats invariance-only and compute-matched two-view controls.

### 4.6 Full objective

```text
L = L_FM + lambda_REPA L_REPA
         + lambda_p L_p
         + lambda_r L_r
         + lambda_t L_t
         + optional collapse regularizers.
```

Recommended first full setting: `lambda_p=0.1`, `lambda_r=0.1`, `lambda_t=0`, mixed noise `0.5`, source block `8`, final block as target, factor width `256`.

Variance and decorrelation terms are implemented but default to zero. They are rescue regularizers, not core contributions: orthogonality does not imply information disentanglement.

## 5. Why this is not merely “two heads”

Generic invariant/equivariant splits predate this work, including SIE. Diffusion timestep-specific attributes have also been studied. The defensible novelty boundary is the combination of:

1. diffusion/flow trajectories as ordered and controllable paired views;
2. a Persistent/Evolving shared-private factorization with no predefined semantics;
3. cross-view exchangeability as the operational definition of persistence;
4. current-view recomposition as positive supervision for evolving information;
5. a generation-focused study that uses the factorization to explain REPA, SRA, and Self-Flow.

Avoid claiming strict equivariance: stochastic corruption is information-destroying and does not generally define a group action. Use **Persistent/Evolving**, not Invariant/Equivariant, in the paper.

## 6. Implementation map

- `models/sit.py`: factor projectors, recomposer, optional signed-delta transition predictor, feature capture at source/target blocks.
- `loss.py`: paired time/noise construction, paired flow loss, persistence/recomposition/transition losses, collapse and swap diagnostics.
- `train.py`: CLI, external-encoder-free mode, weighted objectives, W&B diagnostics.
- `generate.py`: checkpoint-compatible auxiliary-head construction. Auxiliary heads do not alter the denoising output path at inference.
- `scripts/tfcr_ablation.sh`: executable A0–A8 matrix.
- `tests/test_trajectory_factorization.py`: shape, gradient, transition, and legacy-path tests.

Current implementation performs two views in one concatenated forward and therefore approximately doubles backbone training FLOPs per source image. Every headline result must include a view/FLOP-matched baseline.

## 7. Core experiment matrix

| ID | Method | Two views | Persistent loss | Evolving/recomposition | Purpose |
|---|---|---:|---:|---:|---|
| A0 | SiT | no | no | no | original generative baseline |
| A1 | DiverseDiT | no | no | block diversity only | original-repository baseline |
| A2 | REPA | no | external | no | external persistence |
| A3 | Two-view FM | yes | no | no | critical compute/data control |
| A4 | Inv-only | yes | yes | no | “what stays” control |
| A5 | TFCR | yes | yes | yes | minimal proposed method |
| A6 | TFCR + transition | yes | yes | yes + ordered motion | test whether ordering adds value |
| A7 | DiverseDiT + TFCR | yes | yes | yes | cross-block/trajectory complementarity |
| A8 | REPA + TFCR | yes | external + internal | yes | complementarity to external alignment |

External SRA and Self-Flow results must use official recipes/checkpoints or faithful ports. Do not silently compare against differently tuned unofficial runs.

## 8. Experimental plan

### Stage 0 — correctness and shortcut audit (1–2 days)

- Tiny-model unit tests and 1k-step overfit on 1–4k ImageNet samples.
- Confirm finite losses and gradients in both projectors and recomposer.
- Required healthy signals:
  - Persistent similarity increases;
  - Evolving similarity remains lower and varies with `delta-t`;
  - recomposition gap (`wrong evolving error - correct error`) becomes positive;
  - both branch standard deviations stay away from zero.
- Run same-epsilon and cross-epsilon separately. If only same-epsilon works, check noise leakage. If only cross-epsilon works, check whether “trajectory evolution” has degraded into noise-instance reconstruction.

### Stage 1 — mechanism screen (SiT-B/2, ImageNet 256, 100k steps)

- Run A0, A3, A4, A5, A6 with 3 seeds.
- Report both equal-step and equal-training-FLOP comparisons.
- Sweep only after the default result:
  - `lambda_p`: 0.03, 0.1, 0.3;
  - `lambda_r`: 0.03, 0.1, 0.3;
  - factor width: 128, 256, 512;
  - source depth: 4, 8, 10 for B/2;
  - target depth: source, midpoint, final;
  - `delta-t`: [0.05,0.2], [0.15,0.7], [0.5,0.9];
  - cross-noise probability: 0, 0.5, 1.

### Stage 2 — main ImageNet evidence (SiT-L/2 and XL/2)

- Promote only configurations that satisfy Stage-1 go criteria.
- Train A0–A8 at the standard 400k horizon; at least 3 seeds for A0/A3/A4/A5, 1–3 for expensive secondary baselines.
- Evaluate checkpoints at 50k/100k/200k/400k to measure convergence, not only final FID.
- Generate 50k samples with identical solver, steps, VAE, CFG scale, guidance interval, and seed policy.
- Report FID-50K, sFID, IS, precision, recall, model parameters, training FLOPs, throughput, peak memory, and sampling cost.

### Stage 3 — generality

- ImageNet 512 on the best L/XL setting.
- Text-to-image extension on COCO only after the ImageNet claim is stable.
- At least one second backbone or flow parameterization if compute permits.
- Test compatibility with DiverseDiT and REPA; these combinations are strategically stronger than presenting TFCR as a replacement for every alignment technique.

## 9. The mechanism study: what REPA, SRA, and Self-Flow learn

Train or obtain matched checkpoints and freeze them. At the same layer/timestep grid, extract token and pooled features under controlled `(x0, t, epsilon)` changes.

### 9.1 Factor sensitivity grid

Vary exactly one factor at a time:

- same `x0`, same epsilon, different `t`;
- same `x0`, same `t`, different epsilon;
- different `x0`, same class/t/epsilon seed;
- same `x0`, fixed SNR, spatially heterogeneous timesteps for Self-Flow.

Measure CKA, cosine similarity, centered effective rank, covariance spectrum, and nearest-neighbor identity retention. The central plot is similarity versus `delta-t`, separately for Persistent and Evolving codes.

### 9.2 Probes with explicit interpretation

- class/semantic linear probe: information about source identity/semantics;
- timestep regression: direct timestep leakage;
- noise-instance discrimination or epsilon regression: corruption-realization information;
- velocity reconstruction conditioned on Persistent only, Evolving only, and both: complementary generative information;
- clean-latent reconstruction/probe: source content retained at each timestep;
- cross-noise retrieval: source information invariant to epsilon.

Expected interpretation:

- REPA: strongest semantic/class persistence, reduced timestep sensitivity in the aligned subspace;
- SRA: similar persistence without an external semantic basis, biased toward the deeper/cleaner teacher features;
- Self-Flow: broader coverage of noise/timestep views; gains may remain under attention separation, supporting augmentation rather than purely cross-token teaching;
- TFCR: Persistent matches or approaches their stable information while Evolving preserves conditional information needed to distinguish and reconstruct the current state.

These are hypotheses to test, not conclusions to write before the evidence exists.

### 9.3 Causal interventions

- Swap Persistent across views of the same source and verify target reconstruction remains stable.
- Swap Evolving and verify the reconstructed feature follows the donor state.
- Replace Evolving with a timestep-only MLP. If performance/recomposition is unchanged, the method has failed to learn “what changes.”
- Randomize epsilon while holding `t`; a useful Evolving code should respond, especially under cross-noise pairing.
- Attention-separate a Self-Flow-style input and compare representations, not only FID.
- Measure per-layer/per-timestep gradient cosine between FM and auxiliary objectives to identify where alignment helps or conflicts.

## 10. Go / no-go criteria

Proceed to XL/2 only if all are met on B/2:

1. A5 beats A3 (two-view compute control) by a meaningful margin: target at least 0.2 FID or 10% fewer steps to the same FID, outside seed variation.
2. A5 beats A4, establishing value beyond invariance.
3. recomposition gap is positive and grows during training.
4. Persistent and Evolving branches do not collapse; no branch is replaceable by zeros or a timestep-only code without a clear penalty.
5. gains reproduce across at least 3 seeds and do not come only from doubled views/FLOPs.

No-go or pivot conditions:

- A3 explains the full gain: reframe as trajectory augmentation or redesign the objective.
- A4 equals A5: no evidence that explicit evolution helps.
- Evolving is predicted almost perfectly from `t` alone: increase cross-noise pairing, remove direct time paths, tighten bottlenecks, or add sample-conditional counterfactuals.
- Persistent collapses: enable a small variance term, reduce Evolving capacity, or add branch-usage margins.
- auxiliary gradients hurt late training: decay/stop TFCR after the acceleration phase, analogous to stage-wise alignment observations.

## 11. Reviewer-risk checklist

- **“Two heads are old.”** Lead with trajectory recomposition and causal evidence, not the split.
- **“It is just 2x compute.”** Include A3 and exact FLOP/throughput accounting.
- **“Evolving only encodes timestep.”** Include timestep-only replacement and cross-noise tests.
- **“Reconstruction does not imply disentanglement.”** Use swap interventions, probes, and branch-necessity tests; avoid overclaiming information-theoretic independence.
- **“Self-Flow already models variation.”** Directly compare dual-view augmentation and attention-separated controls; distinguish heterogeneous input augmentation from explicit shared/private factorization.
- **“Why call it equivariant?”** Do not. Use Persistent/Evolving.
- **“Only ImageNet/SiT.”** Add a second resolution/backbone and, if resources allow, COCO.
- **“Auxiliary teacher target is unstable.”** Track target variance; compare same-network stop-gradient, EMA target, and source-depth target.

## 12. Six-week execution schedule

| Week | Deliverable |
|---|---|
| 1 | unit/smoke tests, A0/A3/A4/A5 short runs, diagnostic dashboards |
| 2 | B/2 three-seed mechanism matrix, pairing and depth ablations |
| 3 | L/2 or XL/2 main runs, matched REPA/SRA/Self-Flow setup |
| 4 | representation probes, swap interventions, gradient/CKA analysis |
| 5 | 512/generalization and DiverseDiT/REPA compatibility |
| 6 | final 50k evaluations, figures, failure audit, paper draft |

## 13. Paper skeleton

1. **Introduction:** alignment learns what stays; trajectories also contain learnable change.
2. **A unified analysis:** controlled view diagnostics for REPA, SRA, Self-Flow.
3. **TFCR:** paired views, operational factorization, recomposition, optional transition.
4. **Generation results:** compute-matched matrix and scaling.
5. **What is learned:** probes, similarity curves, swap interventions, failure cases.
6. **Limitations:** stochastic transformations are not strict equivariances; decomposition is operational, not guaranteed statistically independent; training uses extra views.

## 14. Primary references

- Yu et al., [Representation Alignment for Generation (REPA)](https://arxiv.org/abs/2410.06940), ICLR 2025.
- Jiang et al., [Representation Alignment for Diffusion Transformers without External Components (SRA)](https://openreview.net/pdf?id=ds5w2xth93), ICLR 2026.
- Chefer et al., [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis (Self-Flow)](https://arxiv.org/abs/2603.06507), 2026.
- Jiang et al., [From SRA to Self-Flow: Data Augmentation or Self-Supervision?](https://arxiv.org/abs/2607.02508), 2026.
- Garrido et al., [Self-supervised learning of Split Invariant-Equivariant representations](https://proceedings.mlr.press/v202/garrido23a.html), ICML 2023.
- Yue et al., [Exploring Diffusion Time-steps for Unsupervised Representation Learning](https://arxiv.org/abs/2401.11430), ICLR 2024.

## 15. Immediate commands

```bash
# Minimal proposed method
DATA_DIR=/path/to/imagenet PRETRAINED_MODEL_PATH=/path/to/vae \
  bash scripts/tfcr_ablation.sh a5_tfcr

# Critical controls
bash scripts/tfcr_ablation.sh a3_two_view
bash scripts/tfcr_ablation.sh a4_inv_only

# Ordered evolution and complementarity
bash scripts/tfcr_ablation.sh a6_tfcr_transition
bash scripts/tfcr_ablation.sh a7_tfcr_diversedit
bash scripts/tfcr_ablation.sh a8_tfcr_repa
```

For sampling a self-supervised TFCR checkpoint, pass the same factorization architecture and disable REPA projectors:

```bash
torchrun --nproc_per_node=8 generate.py \
  --ckpt /path/to/checkpoint.pt \
  --trajectory-factorization \
  --no-projection \
  --factor-dim 256 \
  --factor-projector-dim 1024 \
  --factor-source-depth 8
```
