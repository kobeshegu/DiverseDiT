# Selective Semantic Source Factorization

## Motivation

The updated controls separate target quality from invariance strength.  Standard
REPA reaches FID 28.15, while paired A5 + REPA reaches 23.66.  Teacher-free
full-feature consistency objectives remain around FID 30.  The new method
therefore aligns only a low-dimensional source code to the clean DINO target
and preserves the ordinary hidden stream as the timestep/noise-dependent state.

This implements the narrower hypothesis:

```text
useful semantic source-stable context
    + state-dependent evolving stream
    -> better velocity learning.
```

## Method

At `factor_source_depth` the existing factorization head produces source and
evolving codes from the current hidden tokens.  Only the source code is mapped
to the clean-image DINO token space.  The evolving code receives no cross-view
alignment target.  A weak squared-cosine penalty is available to discourage the
two readouts from using identical directions.

The decoded spatial source tokens are injected through a zero-initialized,
channel-gated FiLM shift before the remaining transformer blocks.  Consequently,
the aligned representation has a direct, spatially resolved path to the sampled
velocity, while the full hidden stream keeps the information needed for the
current timestep and noise realization.

Zero initialization makes the first forward pass identical to the historical
SiT/A5 path.  `factor_semantic_injection_scale=0` disables FiLM without changing
checkpoint parameter shapes and is used both as a training control and as a
same-checkpoint causal sampling intervention.

Default main coefficients:

```text
semantic REPA       0.5
source consistency  0.0
source/evolving     0.005
FiLM scale          1.0
```

Direct source consistency is disabled by default because both paired views
already share the same clean DINO target.  This avoids adding another broad
invariance constraint.

## Experiment Matrix

| Priority | Job | Steps | Purpose |
| --- | --- | ---: | --- |
| P0 | `58_u0_paired_repa_ratio1_200k.sh` | 200k | Equal-FLOP paired REPA control without A5 losses. |
| P1 | `60_u2_selective_semantic_ratio1_200k.sh` | 200k | Main-method screen. |
| P2 | `62_u3_semantic_no_injection_ratio1_200k.sh` | 200k | Is source alignment useful when it cannot affect late blocks? |
| P3 | `64_u5_semantic_shuffled_source_ratio1_200k.sh` | 200k | Does correct source identity causally matter? |
| P4 | `63_u4_semantic_no_a5_ratio1_200k.sh` | 200k | Are historical A5 losses needed after semantic factorization? |
| P5 | `59_u1_scheduled_repa_400k.sh` | 400k | Does S1 improve because REPA is warmup/decay scheduled? |
| Gate | `61_u2_selective_semantic_ratio1_400k.sh` | 400k | Full main run only after the 200k screen passes. |
| Gate | `65_u0_paired_repa_ratio1_400k.sh` | 400k | Same-step paired REPA control for S1 and the full main run. |

All paired jobs use batch ratio 1.0, legacy paired trajectory sampling,
cross-noise probability 0.5, seed 0, and DINOv2-B targets.  U0 applies the same
REPA warmup/decay schedule as S1, so the only intended difference is the A5
factor objective.  The rank launcher follows the priority order:

```bash
bash scripts/dlc_tfcr_jobs/run_semantic_by_rank.sh "$RANK"
```

## Decision Rules

1. Compare Job 58 at 200k with standard REPA at 400k for an approximate
   equal-backbone-forward comparison.
2. Continue the full 400k main run only if Job 60 beats Job 58 and the existing
   S1 200k checkpoint by a meaningful early margin.
3. A task-coupling claim requires Job 60 to beat Job 62 and same-checkpoint
   sampling with `factor_semantic_injection_scale=0`.
4. A source-identity claim requires the correctly matched Job 60 to beat the
   shuffled-target Job 64.
5. If Job 63 matches Job 60, remove the historical A5 losses from the final
   method and keep the simpler semantic source/state-stream formulation.

## Diagnostics

- `factor_semantic_repa_loss`
- `factor_semantic_source_similarity`
- `factor_semantic_source_evolving_cosine_sq`
- `factor_semantic_source_std`
- `factor_semantic_evolving_std`
- `factor_semantic_modulation_rms`

The modulation RMS should grow above zero without exploding.  A low REPA loss
with near-zero modulation and no FID change indicates an auxiliary
representation that the velocity network ignores.
