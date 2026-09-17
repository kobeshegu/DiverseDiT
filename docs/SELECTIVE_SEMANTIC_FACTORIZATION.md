# Selective Semantic Source Factorization

## Motivation

The updated controls separate target quality from invariance strength.  Standard
REPA reaches FID 28.15, paired A5 + REPA reaches 23.66, and the same-step U0
paired REPA control without A5 losses reaches 23.04.  The selective semantic
U2 run reaches 23.27, close to U0 but not better.  Teacher-free full-feature
consistency objectives remain around FID 30.  The new method therefore aligns
only a low-dimensional source code to the clean DINO target and preserves the
ordinary hidden stream as the timestep/noise-dependent state.

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

## Current Results

Job 65 is now the matched external-target control for this line.  It removes
the A5 factor losses but keeps the paired same-step REPA target and reaches the
best current FID.  Job 61 is the full selective semantic factorization run.

| Job | Run | Main distinction | FID | Delta vs baseline | sFID | IS | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / U0 | `u0_paired_repa-SiT-B-2-s0-r1.0-paired-repa-j65-u0-paired-repa-ratio1-400000` | paired REPA, no A5 factor losses | 23.043364 | 12.857597 | 6.611714 | 63.543480 | 0.593080 | 0.646400 |
| 61 / U2 | `u2_selective_semantic-SiT-B-2-s0-r1.0-semantic-j61-u2-selective-semantic-ratio1-400000` | selective semantic source REPA + FiLM | 23.272617 | 12.628344 | 6.521849 | 63.135998 | 0.594460 | 0.646300 |
| 42 / S1 | `s1_a5_shared_repa-SiT-B-2-s0-r1.0-x0.5-j42-s1-a5-shared-repa-ratio1-400k` | A5 + shared REPA | 23.663463 | 12.237498 | 6.494441 | 62.335083 | 0.588440 | 0.647100 |
| 51 / A2 | `a2_repa-SiT-B-2-s0-j51-repa-baseline-400k` | standard REPA | 28.154142 | 7.746819 | 7.220941 | 54.345894 | 0.561460 | 0.647300 |
| 59 / U1 | `u1_scheduled_repa-SiT-B-2-s0-scheduled-j59-u1-scheduled-repa-400000` | scheduled single-view REPA | 28.263778 | 7.637183 | 6.618042 | 52.407143 | 0.568080 | 0.647900 |

U2 improves over S1 by 0.390846 FID and over standard REPA by 4.881525 FID,
but it is 0.229253 FID worse than U0.  It does have slightly better sFID and
precision than U0, so the selective semantic source pathway is not broken, but
the current FID evidence favors the simpler paired REPA control.  U1 is only
0.109636 FID worse than standard REPA and remains 4.991161 FID worse than U2,
so REPA scheduling alone does not explain the paired-run gain.  This makes
23.043364 the main gate for any selective semantic factorization claim.
Beating A5 alone is no longer enough; the method needs to beat or closely match
U0 while adding a meaningful causal or diagnostic advantage.

## Decision Rules

1. Compare Job 58 at 200k with standard REPA at 400k for an approximate
   equal-backbone-forward comparison.
2. For 400k claims, compare the full semantic run against Job 65 first, then
   S1 and standard A2 REPA.
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
