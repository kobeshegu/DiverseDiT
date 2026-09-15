# TFCR Ablation Results

## Setup

- Model: `SiT-B/2`
- Data: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907`
- Pretrained assets: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models`
- Training: 400k steps, batch size 256, seed 0, fp16, no REPA (`enc-type=none`, `proj-coeff=0`) unless explicitly marked as an external-target control
- Sampling: `0400000.pt`, 50k samples, `sde`, 250 steps, CFG 1.0, VAE `mse`, seed 0
- Reference: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz`
- FID graph: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/REPA/classify_image_graph_def.pb`
- Baseline for delta: `sit_b2_no_repa_baseline_seed0`, FID 35.900961
- Baseline metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sit_b2_no_repa_baseline_seed0/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- REPA metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/a2_repa-SiT-B-2-s0-j51-repa-baseline-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`

Common TFCR defaults:

- `factor_dim=256`
- `factor_projector_dim=1024`
- `factor_source_depth=8`
- `factor_min_delta_t=0.15`
- `factor_max_delta_t=0.7`
- `factor_inv_coeff=0.1`
- `factor_persistent_coeff=0.05`
- `factor_evolving_coeff=0.05`
- `factor_recom_coeff=0.1`
- `factor_warmup_steps=10000`
- `factor_decay_start=250000`
- `factor_decay_end=400000`

## Results

`Delta FID` is `baseline FID - experiment FID`; higher is better.
The baseline row is integrated from the existing `sit_b2_no_repa_baseline_seed0` metrics file above, timestamped `2026-08-24 06:06:46`.
The A3 ratio-1.0 run was submitted with `RUN_SUFFIX=j24-two-view-ratio100`, so its output directory contains `j24`; it corresponds to the A3 two-view ratio-1.0 control job.
The A12 adversarial purification run was submitted before the script rename, so its output directory contains `ratio100`; it corresponds to `27_a12_adv_purification_ratio1_400k.sh`.
VGSC Jobs 33, 37, and 39 currently have identical sample NPZ SHA256
`143acdd1994765adecd4d12f1dcd0bce31358305a57f4420b9bed1c897ae869c`.
Their metrics are therefore reported for traceability, but comparisons among
full VGSC, shuffled utility, and weak VGSC require regenerated samples.
External-target and S-series paired shared-target follow-ups are summarized
after the main table; A2/S1 use external DINO targets and S2--S9 are
teacher-free.

| Rank | Job | Experiment | Key config | FID | Delta FID | sFID | IS | Precision | Recall |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | baseline | `sit_b2_no_repa_baseline_seed0` | no REPA baseline | 35.900961 | 0.000000 | 6.688123 | 41.130138 | 0.524620 | 0.644600 |
| 1 | 22 | `a5_tfcr-SiT-B-2-s0-r1.0-x0.5-j22-ratio100` | TFCR, ratio 1.0, cross-noise 0.5 | 29.825559 | 6.075402 | 6.331125 | 50.477547 | 0.555660 | 0.642900 |
| 2 | 30 | `a3_two_view-SiT-B-2-s0-r1.0-x0.5-j24-two-view-ratio100` | paired two-view control, ratio 1.0, factor losses 0 | 30.107791 | 5.793170 | 6.413116 | 50.217392 | 0.554620 | 0.648800 |
| 3 | 35 | `v2_selective_uniform-SiT-B-2-s0-r1.0-x0.5-j35-v2-selective-uniform-ratio1-400k` | V-series clean consensus + selective subspace, uniform weighting, ratio 1.0 | 30.239068 | 5.661893 | 6.462473 | 49.687996 | 0.553380 | 0.645900 |
| 4 | 36 | `v5_vgsc_shuffled_source-SiT-B-2-s0-r1.0-x0.5-j36-v5-vgsc-shuffled-source-ratio1-400k` | V4 with shuffled source targets, ratio 1.0 | 30.281753 | 5.619208 | 6.441387 | 49.814598 | 0.551280 | 0.646000 |
| 5 | 32 | `v1_clean_consensus-SiT-B-2-s0-r1.0-x0.5-j32-v1-clean-consensus-ratio1-400k` | clean consensus only, ratio 1.0 | 30.315289 | 5.585672 | 6.457972 | 49.715931 | 0.551200 | 0.649600 |
| 6 | 33 | `v4_vgsc-SiT-B-2-s0-r1.0-x0.5-j33-v4-vgsc-ratio1-400k` | full VGSC task weighting, ratio 1.0; duplicate sample hash with Jobs 37/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551180 | 0.645800 |
| 7 | 37 | `v6_vgsc_shuffled_utility-SiT-B-2-s0-r1.0-x0.5-j37-v6-vgsc-shuffled-utility-ratio1-400k` | V4 with shuffled utility; duplicate sample hash with Jobs 33/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551160 | 0.645800 |
| 8 | 39 | `v8_vgsc_weak-SiT-B-2-s0-r1.0-x0.5-j39-v8-vgsc-weak-ratio1-400k` | weak full VGSC; duplicate sample hash with Jobs 33/37 | 30.411934 | 5.489027 | 6.479882 | 49.275566 | 0.551160 | 0.645800 |
| 9 | 38 | `v7_selective_task_only-SiT-B-2-s0-r1.0-x0.5-j38-v7-selective-task-only-ratio1-400k` | task-weighted selective subspace only, no clean consensus, ratio 1.0 | 30.453639 | 5.447322 | 6.376558 | 49.380562 | 0.550420 | 0.651400 |
| 10 | 34 | `v3_selective_stability-SiT-B-2-s0-r1.0-x0.5-j34-v3-selective-stability-ratio1-400k` | V-series clean consensus + selective subspace, stability weighting, ratio 1.0 | 30.509307 | 5.391654 | 6.481290 | 49.351646 | 0.550620 | 0.644500 |
| 11 | 23 | `a4_inv_only-SiT-B-2-s0-r1.0-x0.5-j23-inv-only-ratio100` | invariance only, ratio 1.0, coeff 0.1 | 30.659704 | 5.241257 | 6.478675 | 49.242001 | 0.549860 | 0.647200 |
| 12 | 28 | `a13_adv_shuffled-SiT-B-2-s0-r1.0-x0.5-j28-adv-shuffled-ratio1-400k` | A12 with shuffled adversarial labels, ratio 1.0 | 31.232931 | 4.668030 | 6.254956 | 47.430557 | 0.545660 | 0.640600 |
| 13 | 09 | `a5_tfcr-SiT-B-2-s0-r0.75-x0.5-j09-ratio075` | TFCR, ratio 0.75, cross-noise 0.5 | 31.796648 | 4.104313 | 6.375076 | 47.054230 | 0.545080 | 0.645900 |
| 14 | 11 | `a4_inv_only-SiT-B-2-s0-r0.75-x0.5-j11-inv-only-ratio075` | invariance only, ratio 0.75, coeff 0.1 | 31.892969 | 4.007992 | 6.570766 | 47.141048 | 0.546260 | 0.646900 |
| 15 | 24 | `a9_orbit_consensus-SiT-B-2-s0-r1.0-x0.5-j24-orbit-consensus-ratio1-400k` | orthogonal orbit, reliable target, velocity recomposition, ratio 1.0 | 31.974437 | 3.926524 | 6.431790 | 47.218697 | 0.542120 | 0.647900 |
| 16 | 10 | `a3_two_view-SiT-B-2-s0-r0.75-x0.5-j10-two-view-ratio075` | paired two-view control, ratio 0.75, factor losses 0 | 32.298296 | 3.602665 | 6.587726 | 47.287582 | 0.543780 | 0.645300 |
| 17 | 27 | `a12_adv_purification-SiT-B-2-s0-r1.0-x0.5-j27-adv-purification-ratio100` | A9 + timestep/orbit adversarial purification, ratio 1.0 | 32.321836 | 3.579125 | 6.362123 | 45.858646 | 0.540220 | 0.649400 |
| 18 | 05 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.0-j05-same-noise` | TFCR, ratio 0.5, cross-noise 0.0 | 33.251927 | 2.649034 | 6.372617 | 44.785519 | 0.538160 | 0.636300 |
| 19 | 04 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.5-j04-default` | TFCR, ratio 0.5, cross-noise 0.5 | 33.329855 | 2.571106 | 6.412439 | 44.192127 | 0.536740 | 0.638900 |
| 20 | 02 | `a3_two_view-SiT-B-2-s0-r0.5-x0.5-j02-two-view-control` | paired two-view control, ratio 0.5, factor losses 0 | 33.347761 | 2.553200 | 6.550316 | 44.671146 | 0.537120 | 0.642600 |
| 21 | 06 | `a5_tfcr-SiT-B-2-s0-r0.5-x1.0-j06-cross-noise` | TFCR, ratio 0.5, cross-noise 1.0 | 33.398930 | 2.502031 | 6.417396 | 44.522701 | 0.533700 | 0.641100 |
| 22 | 03 | `a4_inv_only-SiT-B-2-s0-r0.5-x0.5-j03-inv-only` | invariance only, ratio 0.5, coeff 0.1 | 33.481124 | 2.419837 | 6.514201 | 44.874249 | 0.535540 | 0.644300 |
| 23 | 08 | `a6_tfcr_transition-SiT-B-2-s0-r0.5-x0.5-j08-transition` | TFCR + transition, ratio 0.5, coeff 0.05 | 33.485353 | 2.415608 | 6.411213 | 44.630997 | 0.534340 | 0.646000 |
| 24 | 12 | `q0_invariant_three_view-SiT-B-2-s0-r0.375-linear-j12-full400k` | three-view invariant control, ratio 0.375, losses 0 | 33.682449 | 2.218512 | 6.628808 | 44.513752 | 0.530000 | 0.652600 |
| 25 | 15 | `q3_orbit_full-SiT-B-2-s0-r0.375-linear-j15-full400k` | teacher-free invariant subspace Q3 full | 34.423308 | 1.477653 | 6.737413 | 43.989368 | 0.524720 | 0.648800 |
| 26 | 14 | `q2_orbit_spread-SiT-B-2-s0-r0.375-linear-j14-full400k` | teacher-free invariant subspace Q2 | 34.478621 | 1.422340 | 6.793144 | 44.097008 | 0.527420 | 0.647500 |
| 27 | 07 | `a5_tfcr-SiT-B-2-s0-r0.25-x0.5-j07-ratio025` | TFCR, ratio 0.25, cross-noise 0.5 | 35.125785 | 0.775176 | 6.557955 | 42.306396 | 0.527560 | 0.645600 |

### External-Target and Paired Shared-Target Follow-up

A2 is the standard REPA baseline.  All S-series jobs use the A5 ratio-1.0 path.
`Delta vs A5` is `A5 FID - experiment FID`, so positive means better than the
current no-external-teacher A5 reference.  The full S-series design and per-job
interpretation are recorded in `docs/PAIRED_SHARED_TARGET_EXPERIMENTS.md`.

| Job | Experiment | Added signal | FID | Delta FID | Delta vs A5 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | `s1_a5_shared_repa-SiT-B-2-s0-r1.0-x0.5-j42-s1-a5-shared-repa-ratio1-400k` | external DINO shared REPA | 23.663463 | 12.237498 | 6.162096 | 6.494441 | 62.335083 | 0.588440 | 0.647100 |
| 51 | `a2_repa-SiT-B-2-s0-j51-repa-baseline-400k` | standard DINOv2-B REPA | 28.154142 | 7.746819 | 1.671417 | 7.220941 | 54.345894 | 0.561460 | 0.647300 |
| 44 | `s3_a5_shared_self_distill-SiT-B-2-s0-r1.0-x0.5-j44-s3-a5-shared-self-distill-ratio1-400k` | reliable self-distill | 30.180189 | 5.720772 | -0.354630 | 6.308376 | 49.576607 | 0.550020 | 0.646600 |
| 47 | `s6_a5_private_separation-SiT-B-2-s0-r1.0-x0.5-j47-s6-a5-private-separation-ratio1-400k` | evolving private separation | 30.265704 | 5.635257 | -0.440145 | 6.350427 | 49.477959 | 0.550860 | 0.647300 |
| 50 | `s7_a5_contrastive_private-SiT-B-2-s0-r1.0-x0.5-j50-s9-s7-contrastive-private-weak-ratio1-400k` | weak contrastive + private | 30.407736 | 5.493225 | -0.582177 | 6.351509 | 49.462532 | 0.551040 | 0.643900 |
| 43 | `s2_a5_shared_clean-SiT-B-2-s0-r1.0-x0.5-j43-s2-a5-shared-clean-ratio1-400k` | analytic clean consensus | 30.546366 | 5.354595 | -0.720807 | 6.379982 | 49.565720 | 0.548860 | 0.647400 |
| 49 | `s4_a5_shared_contrastive-SiT-B-2-s0-r1.0-x0.5-j49-s8-s4-contrastive-weak-ratio1-400k` | weak contrastive | 30.582932 | 5.318029 | -0.757373 | 6.418774 | 49.663208 | 0.552260 | 0.643700 |
| 45 | `s4_a5_shared_contrastive-SiT-B-2-s0-r1.0-x0.5-j45-s4-a5-shared-contrastive-ratio1-400k` | contrastive | 30.605042 | 5.295919 | -0.779483 | 6.399550 | 49.607513 | 0.552020 | 0.643600 |
| 46 | `s5_a5_shared_relation-SiT-B-2-s0-r1.0-x0.5-j46-s5-a5-shared-relation-ratio1-400k` | relation consistency | 30.676918 | 5.224043 | -0.851359 | 6.319471 | 49.011528 | 0.548100 | 0.642200 |
| 48 | `s7_a5_contrastive_private-SiT-B-2-s0-r1.0-x0.5-j48-s7-a5-contrastive-private-ratio1-400k` | contrastive + private | 30.787519 | 5.113442 | -0.961960 | 6.453773 | 49.189491 | 0.550340 | 0.645800 |

## Analysis

The ratio-1.0 follow-up is the first clear win for dense paired trajectory supervision in this branch. A5 ratio 1.0 reaches FID 29.83, improving the no-REPA baseline by 6.08 FID and the previous best A5 ratio-0.75 run by 1.97 FID. The compute-matched A3 two-view control at ratio 1.0 is also strong at FID 30.11, improving the baseline by 5.79 FID. IS and precision move strongly upward for both A5 and A3, while A3 has the best recall among the ratio-1.0 runs.

The external-target results now have a clear ordering.  Standard A2 REPA
reaches FID 28.15, improving the no-REPA baseline by 7.75 FID and beating A5
ratio 1.0 by 1.67 FID.  S1, which adds an external DINO shared REPA target on
top of A5, is by far the strongest result at FID 23.66.  It improves over
standard REPA by 4.49 FID and over A5 by 6.16 FID.  This means external DINO
supervision remains stronger than the current teacher-free TFCR objective, and
the paired/A5 path can exploit that external target much more effectively than
vanilla REPA.

The teacher-free S2--S9 set is a different story.  The best result is S3
self-distillation at FID 30.18, followed by S6 private separation at 30.27 and
S9 weak contrastive + private separation at 30.41.  None of these beats A5
ratio 1.0, and only S3 is very close to the A3 two-view control.

The completed VGSC matrix is competitive with the strongest ratio-1.0 family, but it does not beat the A3/A5 front-runners. V2 uniform selective weighting is the best verified V-series run at FID 30.24, only 0.13 behind A3 ratio 1.0 and 0.41 behind A5 ratio 1.0, while improving over A4 inv-only ratio 1.0 by 0.42 FID. V1 clean consensus alone reaches 30.32, and V7 task-selective-only reaches 30.45, so both clean consensus and selective subspace have positive signal. V3 stability-only reaches 30.51, 0.27 worse than uniform, which suggests source-stability weighting alone is not better than a uniform selective subspace in this seed.

The negative controls are mixed. V5 shuffled-source reaches FID 30.28, nearly matching V2 uniform and beating V3, so the current result does not support a strong causal claim that correct source correspondence is required. Jobs 33, 37, and 39 currently share the exact same sample NPZ hash despite different checkpoint hashes; their metric values should be treated as a duplicate-sample artifact until those samples are regenerated. As recorded, the full VGSC/task-weighted result would not beat the simpler V1/V2 controls, but that conclusion needs a clean rerun for Jobs 33/37/39.

The clearest positive signal is now the factor batch ratio. Increasing the paired/factorized supervision ratio from 0.75 to 1.0 improves full TFCR from FID 31.80 to 29.83, A3 two-view control from 32.30 to 30.11, and inv-only from 31.89 to 30.66. Ratio 0.25 only improves by 0.78 FID, so weak factorized supervision is insufficient; in this seed, using the auxiliary trajectory path on every batch is best.

The ratio-1.0 A3 result changes the attribution. A5 still has the best FID, but it beats the compute-matched A3 control by only 0.28 FID, while A3 itself beats A4 inv-only by 0.55 FID. This means most of the gain comes from the dense paired-view trajectory training setup rather than the explicit invariance loss alone. The full persistent/evolving/recomposition objective still appears useful, but its current measured contribution over the two-view control is modest and should be treated as a smaller second-order gain until repeated across seeds.

The A9/A12/A13 follow-up does not support the v2 orbit-consensus stack as a replacement for the simpler ratio-1.0 TFCR setting. A9 reaches FID 31.97, which is 2.15 FID worse than A5 ratio 1.0, 1.87 worse than the A3 two-view ratio-1.0 compute control, and 1.31 worse than A4 inv-only ratio 1.0. A12 further degrades A9 to 32.32, so true-label adversarial purification is not helping here. The shuffled-label A13 control is better at 31.23, improving A9 by 0.74 and A12 by 1.09, but it is still 1.41 FID worse than A5 ratio 1.0 and 1.13 worse than A3 ratio 1.0. This points away from useful nuisance-label adversarial learning: any benefit in A13 looks more like generic regularization/noise than a causal improvement from the proposed adversarial labels.

The cross-noise probability is not a sensitive knob at ratio 0.5 in this seed. Same-noise, mixed-noise, and full cross-noise are all close. The transition head also does not help here; it is slightly worse than the default TFCR configuration.

The teacher-free invariant subspace follow-up did not validate the Q-series replacement idea. Q0, the three-view control, reaches FID 33.68, while Q2 and Q3 are worse at 34.48 and 34.42. This means the model can learn the invariant readout, but the discarded subspace objective does not improve generation quality here. The evidence therefore favors dense paired trajectory factorization over the separate teacher-free invariant-subspace design.

The shared-target batch gives the same directional lesson.  Once the A2/S1
external DINO controls are excluded, explicit teacher-free shared-target losses
are mostly neutral to negative relative to A5.  Contrastive source identity,
relation matching, and analytic clean consensus do not improve generation in
this seed; weakening contrastive/private losses helps but does not close the
gap.  The less-negative signals are S3 and S6, suggesting that softer
self-distillation or protecting private/evolving information may be worth one
more targeted pass, whereas broad S4/S7 coefficient search is lower value.

## Takeaways

- If external representation targets are allowed, S1 shared REPA is the current best result at FID 23.66.
- Standard A2 REPA reaches FID 28.15, beats A5 by 1.67 FID, and should be included as the external-teacher baseline in all headline comparisons.
- Among no-external-teacher runs, the current best configuration remains full TFCR with `factor_batch_ratio=1.0` and `cross_noise_prob=0.5`.
- The main effect is high-ratio paired trajectory supervision; the ratio sweep is monotonic in the tested A5 and A3 settings.
- The full decomposition is positive but modest at ratio 1.0: A5 beats A3 two-view control by 0.28 FID and A4 inv-only by 0.83 FID.
- The new teacher-free shared-target variants do not beat A5; S3 self-distill is the closest at 30.18 FID and S6 private separation is second at 30.27 FID.
- The V-series controls are strong but not yet better than A3/A5: V2 uniform is the best verified V-series result, V1 clean consensus is close, V7 selective-only is also positive, and V3 stability-only is worse than uniform.
- V5 shuffled-source nearly matches V2, so correct source correspondence is not yet causally supported by the current negative-control results.
- Jobs 33/37/39 must be regenerated before using them to compare full VGSC, shuffled utility, and weak VGSC; their current sample NPZ files are byte-identical.
- Invariance-only is not the main explanation for the improvement in this seed, because A3 two-view control ratio 1.0 beats A4 inv-only ratio 1.0 by 0.55 FID.
- A9 orbit consensus is not competitive in its current setting. A12 true-label adversarial purification worsens it, while A13 shuffled labels improve over A9/A12 but still trail A5/A3/A4 ratio 1.0.
- The Q-series teacher-free invariant-subspace alternative should be deprioritized unless a much weaker or differently placed objective is tested.
- Next validation should return to the simpler ratio-1.0 A5/A3 family or run seed repeats; more A10/A11/A13/A14 adversarial attribution jobs are lower priority unless the A9 design is changed.
