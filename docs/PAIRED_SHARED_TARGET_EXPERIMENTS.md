# Paired Shared-Target Experiments

## Goal

These jobs test paired trajectory supervision as a source-defined shared-target
mechanism, instead of manually choosing invariant dimensions.  All jobs use the
existing A5 persistent/evolving factorization path at ratio 1.0 and train for
400k steps unless overridden.

The intent is to separate three questions:

1. Can a stable target be defined from paired trajectory/noise views?
2. Does the shared target improve generation beyond dense paired supervision?
3. Does preserving a private/evolving code help avoid over-invariance?

## Implemented Objectives

| Group | Objective | Main switch | Rationale |
| --- | --- | --- | --- |
| S1 | External shared REPA target | `--factor-shared-repa` | Bridge/control: paired views share the same DINO clean-image target. |
| S2 | Analytic clean consensus | `--factor-shared-clean` | Uses paired velocity predictions to recover and align confidence-weighted `x0_hat`. |
| S3 | Full-feature self-distill | `--factor-shared-self-distill` | Aligns full hidden features to a reliable stop-grad view consensus. |
| S4 | Trajectory contrastive | `--factor-shared-contrastive` | Same-source paired views are positives; other sources are negatives. |
| S5 | Relation consistency | `--factor-shared-relation` | Matches source-level and local spatial relations rather than absolute vectors. |
| S6 | Private/evolving separation | `--factor-evolving-separation` | Keeps evolving codes view-specific with a cosine-distance margin. |
| S7 | Contrastive + private separation | `--factor-shared-contrastive --factor-evolving-separation` | Combines source identity with explicit shared/private pressure. |
| S8 | Weak S4 | same as S4 | Tests whether contrastive regularization was too strong. |
| S9 | Weak S7 | same as S7 | Tests weaker contrastive/private combination. |

S1 is intentionally close to REPA and should be treated as a control, not the
main novelty claim.  S2--S9 are teacher-free or source-structured objectives
defined by the paired diffusion trajectory itself.

## DLC Jobs

Submit individual one-card jobs:

```bash
bash scripts/dlc_tfcr_jobs/42_s1_a5_shared_repa_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/43_s2_a5_shared_clean_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/44_s3_a5_shared_self_distill_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/45_s4_a5_shared_contrastive_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/46_s5_a5_shared_relation_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/47_s6_a5_private_separation_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/48_s7_a5_contrastive_private_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/49_s8_s4_contrastive_weak_ratio1_400k.sh
bash scripts/dlc_tfcr_jobs/50_s9_s7_contrastive_private_weak_ratio1_400k.sh
```

For a task-array style submission, `run_shared_target_by_rank.sh` maps task
indices 0--6 to S1--S7.

## Priority

If only two jobs are available, run:

1. `49_s8_s4_contrastive_weak_ratio1_400k.sh`
2. `50_s9_s7_contrastive_private_weak_ratio1_400k.sh`

If seven jobs are available, prioritize:

1. S4 contrastive
2. S7 contrastive + private separation
3. S6 private separation
4. S5 relation consistency
5. S2 clean consensus
6. S3 self-distill
7. S1 shared REPA control

## Expected Interpretation

- If S4/S8 improves, source identity contrast across trajectory views is a
  useful shared-target signal.
- If S7/S9 improves over S4/S8, preserving an explicit evolving/private role is
  important.
- If S6 improves alone, the bottleneck is over-invariance rather than the
  shared target.
- If only S1 improves, the method is likely too close to REPA and should be
  positioned as a paired-REPA control.
- If S5 improves, relation-level consistency may be a better paper direction
  because it differs most clearly from absolute-target REPA.

## Verification

Before committing, the following checks were run:

```bash
/root/anaconda3/envs/repa/bin/python -m py_compile train.py loss.py models/sit.py generate.py
bash -n scripts/tfcr_ablation.sh scripts/dlc_train_tfcr.sh scripts/tfcr_common.sh scripts/dlc_tfcr_jobs/common.sh
DRY_RUN=1 bash scripts/dlc_tfcr_jobs/49_s8_s4_contrastive_weak_ratio1_400k.sh
DRY_RUN=1 bash scripts/dlc_tfcr_jobs/50_s9_s7_contrastive_private_weak_ratio1_400k.sh
```

Small CPU forward/backward tests covered shared REPA, self-distill,
contrastive, relation, and evolving separation losses.
