# Unified analysis of what stays and what changes

This document is the second theoretical and implementation review of TFCR. It
turns the paper claim into falsifiable experiments and deliberately separates
what is already guaranteed by the objective from what must be established by
evidence.

## 1. Review outcome

The overall direction is reasonable, but the strongest defensible claim is not
that REPA, SRA, Self-Flow, and DiverseDiT are all “invariance methods.” A more
accurate unification has three axes:

```text
view construction  +  target selection  +  information routing
```

- REPA uses ordinary noisy views, an external clean-image target, and an
  alignment route at selected blocks.
- SRA changes the target to a lower-noise, deeper EMA feature and therefore
  introduces both timestep and depth asymmetry.
- Self-Flow changes view construction through heterogeneous token timesteps and
  uses contextual feature reconstruction; information asymmetry can train both
  stable content and state-dependent completion.
- DiverseDiT primarily changes routing across the block axis. It encourages
  block specialization and is not, by itself, a cross-view invariant learner.
- TFCR constructs paired trajectory views and explicitly routes pair-symmetric
  and view-residual information into Persistent and Evolving branches.

This framing is broader and safer than claiming that previous methods learn
only invariance. The hypothesis to test is that they allocate representation
capacity differently between source-stable, timestep-varying, noise-varying,
and block-specific components.

## 2. Theoretical formulation

For source latent `x0`, timestep `t`, and noise realization `epsilon`, let a
block representation be

```text
h_l(x0, t, epsilon).
```

The flow-matching objective identifies the output velocity, not a unique hidden
representation. Invertible transformations of hidden states can leave the
output unchanged, so semantic interpretations of internal dimensions are not
identifiable from the generative loss alone. Alignment and routing objectives
reduce this solution class by introducing operational constraints.

For a balanced controlled tensor `[source, timestep, noise, channel]`, the law
of total variance gives an assumption-light decomposition:

```text
Var(h) = Var_source(E_t,e[h])
       + E_source Var_t(E_e[h])
       + E_source,t Var_e(h).
```

We name these terms source-stable, timestep-varying, and noise-varying energy.
They do not presuppose semantic, structural, texture, or frequency meanings.
Those meanings are measured later by probes.

## 3. Balanced TFCR objective

The first implementation used a nonlinear joint recomposer:

```text
R(P_other, E_current) -> h_current.
```

That objective admits a serious degenerate solution: `E_current` can carry the
entire target while the recomposer ignores `P_other`. Cross-view recomposition
alone therefore does not make the decomposition identifiable.

The revised implementation uses additive, observable branch decoders. For a
target pair `(h_a, h_b)`:

```text
c       = (h_a + h_b) / 2
r_a     = h_a - c
r_b     = h_b - c

D_P(p_a), D_P(p_b) -> c
D_E(e_a)           -> r_a
D_E(e_b)           -> r_b

h_hat_a = D_P(p_b) + D_E(e_a)
h_hat_b = D_P(p_a) + D_E(e_b).
```

This directly requires both branches to be useful and retains cross-view
exchangeability. It still does not prove statistical independence or a unique
semantic decomposition: invertible transforms within either latent and
pair-distribution effects remain. The correct claim is **balanced operational
factorization**, verified by branch necessity and controlled probes.

## 4. Falsifiable predictions by method

| Method | Expected source-stable signal | Expected variant signal | Distinct mechanism signature |
|---|---|---|---|
| SiT | grows with training | unstructured time/noise residual | baseline block/timestep CKA |
| REPA | rises near aligned block; teacher CKA rises | time/noise energy displaced to other blocks/subspaces | local drop in cross-block CKA around aligned block |
| SRA | high-noise shallow feature becomes closer to low-noise deep EMA target | depth/timestep asymmetry remains outside aligned subspace | student–teacher CKA and gradient concentration |
| Self-Flow | context/source recovery improves under token corruption | heterogeneous-token state and missing-region residual | improvement survives or changes under attention separation/matched augmentation |
| DiverseDiT | not predetermined | block-specific energy increases | lower off-diagonal block CKA and higher effective rank |
| TFCR | Persistent source fraction and cross-t retrieval increase | Evolving timestep/noise fractions increase | both usage gaps positive; correct swap beats wrong/zero branches |

These are predictions, not conclusions. A result contradicting them is useful:
it either refines the unified framework or falsifies the paper story.

## 5. DiverseDiT-style analysis program

Use ImageNet-256, SiT-B/2, batch 256, and checkpoints at 5k, 50k, 200k, and
450k. For every method export a balanced tensor with 512 fixed source samples,
five timesteps `[.05, .25, .50, .75, .95]`, four fixed noise realizations, and
the same block depths. Use identical latent samples and noise banks across
methods.

### U0 — layer × trajectory CKA atlas

Produce three heatmap families at every checkpoint:

1. block × block CKA, matching the original DiverseDiT analysis;
2. timestep × timestep CKA for each block;
3. method × method CKA at matched block and timestep.

This reveals whether a method creates block specialization, trajectory
invariance, or both.

### U1 — controlled variance decomposition

Report source/timestep/noise fractions for every block and checkpoint. A single
number is insufficient; show curves over depth and training time. Bootstrap
source samples for 95% confidence intervals and repeat the final result across
three training seeds.

### U2 — information probes without semantic assumptions

Fit probes on frozen representations:

- source identity retrieval across timesteps and noises;
- ImageNet class linear probe;
- timestep regression;
- noise/velocity reconstruction;
- clean latent low-frequency and high-frequency reconstruction;
- spatial correspondence or dense-position probe.

Use separate train/test source identities for timestep/noise probes so a probe
cannot memorize examples. Report both raw block features and the residual after
regressing out timestep.

### U3 — causal routing interventions

For TFCR, evaluate feature and velocity error after:

- zero Persistent / zero Evolving;
- swap Persistent between same-source views;
- swap Persistent across different sources but same class;
- swap Evolving across timestep, noise, and source independently;
- replace Evolving with a timestep-only MLP;
- shuffle spatial tokens within either branch.

For REPA/SRA, replace only the aligned subspace or transplant aligned-block
features between controlled views. For Self-Flow, separate heterogeneous-view
augmentation from cross-token attention using the official attention-separation
control. For DiverseDiT, remove individual long skips or homogenize selected
block inputs. Measure immediate velocity change and full 50k-sample FID for the
small set of most diagnostic interventions.

### U4 — objective influence maps

At matched checkpoints, compute per-block gradient cosine and gradient norm for
FM versus REPA/SRA/TFCR/diversity objectives. Plot `block × timestep` maps. This
tests whether improvements arise from compatible optimization, localized
specialization, or late-stage conflict. Auxiliary-loss termination should be
selected from these maps, not only from FID.

### U5 — alignment/routing budget sweep

Following DiverseDiT's single-block versus multi-block alignment analysis:

- move each objective across early/middle/late blocks;
- apply it to one, three, or all selected blocks;
- vary paired-view ratio at matched source images and FLOPs;
- compare same-noise, cross-noise, and 50/50 mixed pairing.

If more aligned blocks reduce block diversity while increasing source fraction,
the framework predicts a U-shaped generation trade-off rather than monotonic
improvement.

### U6 — mechanism-to-performance mediation

Across checkpoints, depths, loss weights, and methods, regress generation
metrics on representation measurements:

```text
FID ~ source_fraction + timestep_fraction + noise_fraction
    + off_diagonal_block_CKA + effective_rank + compute.
```

This is correlation, not causal proof. Pair it with U3 interventions. The
strongest paper evidence is a representation metric that predicts performance
and whose targeted intervention changes performance in the predicted direction.

## 6. Effect-validation hierarchy

1. **Correctness:** finite objectives, gradients in both projectors and both
   decoders, no inference overhead.
2. **Necessity:** both branch-usage gaps are positive; zero/replace interventions
   hurt reconstruction and velocity.
3. **Specificity:** Persistent has more source-stable energy than Evolving;
   Evolving has more timestep/noise energy than Persistent.
4. **Non-shortcut:** timestep-only Evolving and source-only Persistent are
   insufficient.
5. **Generation:** A5 beats matched-view A3 and invariance-only A4 outside seed
   variance under equal-step and equal-FLOP reporting.
6. **Unification:** predicted signatures for REPA/SRA/Self-Flow/DiverseDiT are
   observed using the same controlled tensor and metrics.

Do not scale to XL/2 until levels 1–4 pass on B/2. A generation gain without
specificity supports trajectory augmentation, not factorization.

## 7. Practical execution

The repository now provides:

```bash
python analysis/extract_trajectory_features.py \
  --ckpt /path/to/checkpoint.pt \
  --data-dir /path/to/imagenet_latents \
  --method TFCR \
  --output analysis/features/tfcr.npz

python analysis/trajectory_metrics.py analysis/features/*.npz \
  --output analysis/results/invariant_variant.json
```

The exporter covers local SiT-family checkpoints. SRA and Self-Flow should use
small exporters in their official repositories and write the same NPZ schema;
faithfulness is more important than forcing every architecture into one codebase.
