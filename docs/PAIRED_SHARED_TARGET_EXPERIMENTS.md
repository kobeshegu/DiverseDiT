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

## Current Results

All rows below use `SiT-B/2`, 400k training steps, batch size 256, seed 0,
`factor_batch_ratio=1.0`, `cross_noise_prob=0.5`, 50k SDE samples, CFG 1.0,
and VAE `mse`.  Delta is measured against the no-REPA baseline FID 35.900961.
`Delta vs A5` is `A5 FID - experiment FID`, so positive means better than the
current no-external-teacher A5 TFCR reference.

Reference rows:

| Run | Purpose | FID | Delta vs baseline | sFID | IS | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | no REPA baseline | 35.900961 | 0.000000 | 6.688123 | 41.130138 | 0.524620 | 0.644600 |
| A3 | two-view control, ratio 1.0 | 30.107791 | 5.793170 | 6.413116 | 50.217392 | 0.554620 | 0.648800 |
| A5 | TFCR, ratio 1.0 | 29.825559 | 6.075402 | 6.331125 | 50.477547 | 0.555660 | 0.642900 |
| A2 | standard REPA baseline, DINOv2-B | 28.154142 | 7.746819 | 7.220941 | 54.345894 | 0.561460 | 0.647300 |
| U1 / Job 59 | scheduled single-view REPA | 28.263778 | 7.637183 | 6.618042 | 52.407143 | 0.568080 | 0.647900 |
| U0 / Job 65 | paired REPA control, no A5 losses | 23.043364 | 12.857597 | 6.611714 | 63.543480 | 0.593080 | 0.646400 |
| U2 / Job 61 | selective semantic source REPA + FiLM | 23.272617 | 12.628344 | 6.521849 | 63.135998 | 0.594460 | 0.646300 |
| VGSC v4 | clean consensus + task-selective invariance | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551180 | 0.645800 |

Shared-target batch:

| Job | Objective | Main added signal | FID | Delta vs baseline | Delta vs A5 | sFID | IS | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 / S1 | External shared REPA | DINO clean-image target, coeff 0.5 | 23.663463 | 12.237498 | 6.162096 | 6.494441 | 62.335083 | 0.588440 | 0.647100 |
| 43 / S2 | Analytic clean consensus | `x0_hat` consensus + shared clean, coeff 0.05 | 30.546366 | 5.354595 | -0.720807 | 6.379982 | 49.565720 | 0.548860 | 0.647400 |
| 44 / S3 | Full-feature self-distill | reliable hidden-feature consensus, coeff 0.05 | 30.180189 | 5.720772 | -0.354630 | 6.308376 | 49.576607 | 0.550020 | 0.646600 |
| 45 / S4 | Trajectory contrastive | same-source positives, coeff 0.05 | 30.605042 | 5.295919 | -0.779483 | 6.399550 | 49.607513 | 0.552020 | 0.643600 |
| 46 / S5 | Relation consistency | cross-view source/spatial relation matching | 30.676918 | 5.224043 | -0.851359 | 6.319471 | 49.011528 | 0.548100 | 0.642200 |
| 47 / S6 | Private/evolving separation | evolving-code margin, coeff 0.05 | 30.265704 | 5.635257 | -0.440145 | 6.350427 | 49.477959 | 0.550860 | 0.647300 |
| 48 / S7 | Contrastive + private | S4 + S6, coeffs 0.05/0.05 | 30.787519 | 5.113442 | -0.961960 | 6.453773 | 49.189491 | 0.550340 | 0.645800 |
| 49 / S8 | Weak S4 | contrastive coeff 0.025 | 30.582932 | 5.318029 | -0.757373 | 6.418774 | 49.663208 | 0.552260 | 0.643700 |
| 50 / S9 | Weak S7 | contrastive/separation coeffs 0.025/0.025 | 30.407736 | 5.493225 | -0.582177 | 6.351509 | 49.462532 | 0.551040 | 0.643900 |

The S1--S9 sample NPZ files have distinct SHA256 hashes, so this batch does not
repeat the previous VGSC duplicate-sample artifact.

## Original Interpretation Checklist

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

## Result Interpretation

Within the original S1--S9 batch, S1 is the only clearly better result: FID
23.663463, improving over A5 by 6.162096 FID, over the no-REPA baseline by
12.237498 FID, and over the standard A2 REPA baseline by 4.490679 FID.  The
follow-up U0/Job 65 matched control is stronger at FID 23.043364.  It removes
the A5 factor losses and still improves over S1 by 0.620099 FID.  U2/Job 61
selective semantic factorization lands between them at FID 23.272617, improving
over S1 by 0.390846 FID but trailing U0 by 0.229253 FID.  The external-target
gain should therefore be attributed primarily to paired same-step REPA
supervision rather than to the A5 decomposition; the selective semantic source
path is compatible with the target but is not yet additive in FID.

Among teacher-free shared-target variants, S3 is best at FID 30.180189.  It is
competitive with the A3 two-view control, only 0.072398 FID worse than A3, but
still 0.354630 FID worse than A5.  S6 is the second-best teacher-free result at
30.265704, suggesting that preserving view-specific evolving information is
less harmful than forcing a stronger shared target.

The clean-consensus, contrastive, relation, and combined shared/private losses
do not beat the simpler A5 ratio-1.0 reference.  S2, S4, S5, S7, S8, and S9 all
remain between FID 30.407736 and 30.787519, so the extra shared target
constraints are mostly regularizing rather than improving generation.
Weakening the contrastive/private coefficients helps S7 recover from 30.787519
to 30.407736, but still does not beat A5 or A3.

The current evidence therefore splits into two regimes.  With external semantic
targets, standard REPA is already stronger than A5 by 1.671417 FID, scheduled
single-view REPA is similar at FID 28.263778, S1 shows that applying a shared
REPA target inside the paired A5 path is much stronger still, U2 shows that a
selective semantic source path preserves most of that gain, and U0 shows that
the paired REPA control alone is the best version so far.  Without external
targets, the evidence still points to high-ratio paired trajectory training as
the main source of gain; the teacher-free shared-target definitions tested here
have not yet produced a robust improvement over the A3/A5 matched controls.

## Next Direction

The highest-value follow-up is not more S4/S7 coefficient search.  If continuing
this line, focus on the two least-negative teacher-free signals:

1. S3-style self-distillation with a weaker or later-starting coefficient, since
   it is closest to A3 and has the best sFID in this batch.
2. S6-style private separation combined with the base A5 losses only, since it
   may reduce over-invariance without imposing an unreliable shared target.

For paper positioning, keep A2 REPA as the standard external-teacher baseline,
U0 paired REPA as the strongest paired external-teacher control, U2 as a
near-best selective semantic variant that does not beat U0, S1 as the
A5-plus-REPA comparison, and A5 as the main teacher-free result unless seed
repeats show S3 or S6 consistently overtaking A5.

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
