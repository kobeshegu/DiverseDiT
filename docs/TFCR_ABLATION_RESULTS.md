# TFCR Ablation Results

## Setup

- Model: `SiT-B/2`
- Data: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907`
- Pretrained assets: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models`
- Training: 400k steps, batch size 256, fp16, no REPA (`enc-type=none`, `proj-coeff=0`) unless explicitly marked as an external-target control; seed follows each job
- Sampling: `0400000.pt`, 50k samples, `sde`, 250 steps, CFG 1.0, VAE `mse`, matching each job seed
- Reference: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz`
- FID graph: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/REPA/classify_image_graph_def.pb`
- Baseline for delta: `sit_b2_no_repa_baseline_seed0`, FID 35.900961
- Baseline metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sit_b2_no_repa_baseline_seed0/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- REPA metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/a2_repa-SiT-B-2-s0-j51-repa-baseline-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- U1 scheduled REPA metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/u1_scheduled_repa-SiT-B-2-s0-scheduled-j59-u1-scheduled-repa-400000/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- U2 selective semantic metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/u2_selective_semantic-SiT-B-2-s0-r1.0-semantic-j61-u2-selective-semantic-ratio1-400000/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- U0 paired REPA metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/u0_paired_repa-SiT-B-2-s0-r1.0-paired-repa-j65-u0-paired-repa-ratio1-400000/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- SF2 EMA source-align metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sf2_ema_source_align-SiT-B-2-s0-r1.0-x0.5-j67-sf2-ema-source-align-ratio1-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- SF1 EMA full-align metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sf1_ema_full_align-SiT-B-2-s0-r1.0-x0.5-j66-sf1-ema-full-align-ratio1-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- SF3 no-injection metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sf3_no_injection-SiT-B-2-s0-r1.0-x0.5-j68-sf3-no-injection-ratio1-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- SF4 shuffled-teacher metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sf4_shuffle_teacher-SiT-B-2-s0-r1.0-x0.5-j69-sf4-shuffle-teacher-ratio1-400k/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- U0 seed-repeat metrics files: Job 72 seed1 and Job 73 seed2 under `sampled_images/u0_paired_repa-SiT-B-2-s{1,2}-r1.0-paired-repa-*`
- A2 seed-repeat metrics files: Job 74 seed1 and Job 75 seed2 under `sampled_images/a2_repa-SiT-B-2-s{1,2}-*`

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
Job 65 U0 paired REPA sample NPZ SHA256 is
`7815604055d51593eb018646b36a744fded760a76b92cac8fde4eeef2ee08ca7`; its
metrics file SHA256 is
`b3dbc7b2e5b3dc045dd13276f972dcb22d90837f2e7668f0b5d5dca767149ee8`.
Job 61 U2 selective semantic sample NPZ SHA256 is
`e40bcd4aecd4b042fb6724cd58436dbf42c46d467b5fc8902697472d60b6cddf`; its
metrics file SHA256 is
`af75a819adf72d3fba73e9a56ace22f3697326adae99169ac80c6fab3029b728`.
Job 59 U1 scheduled REPA sample NPZ SHA256 is
`4e9245bb3adfe6e679ae65f64eba9a80e11add767771dddff09a576f2a783272`; its
metrics file SHA256 is
`91387b015eff93f60a1de8d95bdd47dbfd91b00ccef860c4f43e6d0ef6326093`.
Job 67 SF2 EMA source-align sample NPZ SHA256 is
`d95edf4e171913a2077fa034dae5127ef65aba672ba629d750c35cbbd4c07e4f`; its
metrics file SHA256 is
`9df223dff0a2e904e4f6471806db14849cac427db8efb8e5d6f276a0dae628d9`.
Job 66 SF1 EMA full-align sample NPZ SHA256 is
`43fc5ef54c79dd7c9ae6e5279af665c9bb1fd88fcb89d65ff2468b304e532c5b`; its
metrics file SHA256 is
`74fbacabb59c32d5c783eee7b463d0d8e7896dba849f0f8474288f3cc3ab77b6`.
Job 68 SF3 no-injection sample NPZ SHA256 is
`1e4c8800ac29ca314601d21820995c4239869528263145ac5ebc3452c437f74e`; its
metrics file SHA256 is
`65816b108c609204b4be8fdda4bd049770bffa9cada8d7b677f49375c4ba283a`.
Job 69 SF4 shuffled-teacher sample NPZ SHA256 is
`03b6f0e520e6823fe31644742fa779bd01c55d96c81d559553c9db6a890dd097`; its
metrics file SHA256 is
`124f63bc862d2f68b2c6029fb3146b4097717c187495f8d6e3ffd4007ba8c57d`.
Job 72 U0 seed1 sample NPZ SHA256 is
`9adedf003ae0d6a7cfbb828796ec7a0b5413099970d2e4600726e29045f7436f`; its
metrics file SHA256 is
`9addd84fdbf8b8e5aec849a72e02563d8d6a7867420032bab387843c59e56340`.
Job 73 U0 seed2 sample NPZ SHA256 is
`adb537772289aafc5c59b4d8998a3d3a87f658a6c9e8bb843b2715d4e63eab9b`; its
metrics file SHA256 is
`9b39331d0dd3efa95bf7c263ac0980f8ebcb0b8a828c63d7635a5302d4370014`.
Job 74 A2 seed1 sample NPZ SHA256 is
`a335f3859f20d77688011b979b7de96bfe63049e3a74aa94c14306e89631ffff`; its
metrics file SHA256 is
`0be9fc536a8799e5b31fa845c65b6020ee4d619a06367f0719fbee43881abce7`.
Job 75 A2 seed2 sample NPZ SHA256 is
`94290e127520116d0a550d434c1c8afeed3a19689677fc34e3caf3afecc855fb`; its
metrics file SHA256 is
`3056777ffe6ba874c02ac2a36100d5db695c162f7307d6b0a867bc3371f5d721`.
External-target, EMA teacher, and S-series paired shared-target follow-ups are
summarized after the main table; A2/S1/U0/U1/U2 use external DINO targets,
SF1--SF4 use an internal EMA teacher, and S2--S9 are teacher-free.

| Rank | Job | Experiment | Key config | FID | Delta FID | sFID | IS | Precision | Recall |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | baseline | `sit_b2_no_repa_baseline_seed0` | no REPA baseline | 35.900961 | 0.000000 | 6.688123 | 41.130138 | 0.524620 | 0.644600 |
| 1 | 22 | `a5_tfcr-SiT-B-2-s0-r1.0-x0.5-j22-ratio100` | TFCR, ratio 1.0, cross-noise 0.5 | 29.825559 | 6.075402 | 6.331125 | 50.477547 | 0.555660 | 0.642900 |
| 2 | 67 | `sf2_ema_source_align-SiT-B-2-s0-r1.0-x0.5-j67-sf2-ema-source-align-ratio1-400k` | EMA teacher source align + FiLM, ratio 1.0 | 29.888876 | 6.012085 | 6.368587 | 49.957775 | 0.552840 | 0.642500 |
| 3 | 30 | `a3_two_view-SiT-B-2-s0-r1.0-x0.5-j24-two-view-ratio100` | paired two-view control, ratio 1.0, factor losses 0 | 30.107791 | 5.793170 | 6.413116 | 50.217392 | 0.554620 | 0.648800 |
| 4 | 68 | `sf3_no_injection-SiT-B-2-s0-r1.0-x0.5-j68-sf3-no-injection-ratio1-400k` | EMA teacher source align, no FiLM injection, ratio 1.0 | 30.162065 | 5.738896 | 6.321264 | 50.075989 | 0.549980 | 0.639800 |
| 5 | 35 | `v2_selective_uniform-SiT-B-2-s0-r1.0-x0.5-j35-v2-selective-uniform-ratio1-400k` | V-series clean consensus + selective subspace, uniform weighting, ratio 1.0 | 30.239068 | 5.661893 | 6.462473 | 49.687996 | 0.553380 | 0.645900 |
| 6 | 36 | `v5_vgsc_shuffled_source-SiT-B-2-s0-r1.0-x0.5-j36-v5-vgsc-shuffled-source-ratio1-400k` | V4 with shuffled source targets, ratio 1.0 | 30.281753 | 5.619208 | 6.441387 | 49.814598 | 0.551280 | 0.646000 |
| 7 | 32 | `v1_clean_consensus-SiT-B-2-s0-r1.0-x0.5-j32-v1-clean-consensus-ratio1-400k` | clean consensus only, ratio 1.0 | 30.315289 | 5.585672 | 6.457972 | 49.715931 | 0.551200 | 0.649600 |
| 8 | 33 | `v4_vgsc-SiT-B-2-s0-r1.0-x0.5-j33-v4-vgsc-ratio1-400k` | full VGSC task weighting, ratio 1.0; duplicate sample hash with Jobs 37/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551180 | 0.645800 |
| 9 | 37 | `v6_vgsc_shuffled_utility-SiT-B-2-s0-r1.0-x0.5-j37-v6-vgsc-shuffled-utility-ratio1-400k` | V4 with shuffled utility; duplicate sample hash with Jobs 33/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551160 | 0.645800 |
| 10 | 39 | `v8_vgsc_weak-SiT-B-2-s0-r1.0-x0.5-j39-v8-vgsc-weak-ratio1-400k` | weak full VGSC; duplicate sample hash with Jobs 33/37 | 30.411934 | 5.489027 | 6.479882 | 49.275566 | 0.551160 | 0.645800 |
| 11 | 38 | `v7_selective_task_only-SiT-B-2-s0-r1.0-x0.5-j38-v7-selective-task-only-ratio1-400k` | task-weighted selective subspace only, no clean consensus, ratio 1.0 | 30.453639 | 5.447322 | 6.376558 | 49.380562 | 0.550420 | 0.651400 |
| 12 | 66 | `sf1_ema_full_align-SiT-B-2-s0-r1.0-x0.5-j66-sf1-ema-full-align-ratio1-400k` | EMA teacher full hidden align, ratio 1.0 | 30.505773 | 5.395188 | 6.360299 | 49.515400 | 0.547820 | 0.648200 |
| 13 | 34 | `v3_selective_stability-SiT-B-2-s0-r1.0-x0.5-j34-v3-selective-stability-ratio1-400k` | V-series clean consensus + selective subspace, stability weighting, ratio 1.0 | 30.509307 | 5.391654 | 6.481290 | 49.351646 | 0.550620 | 0.644500 |
| 14 | 23 | `a4_inv_only-SiT-B-2-s0-r1.0-x0.5-j23-inv-only-ratio100` | invariance only, ratio 1.0, coeff 0.1 | 30.659704 | 5.241257 | 6.478675 | 49.242001 | 0.549860 | 0.647200 |
| 15 | 28 | `a13_adv_shuffled-SiT-B-2-s0-r1.0-x0.5-j28-adv-shuffled-ratio1-400k` | A12 with shuffled adversarial labels, ratio 1.0 | 31.232931 | 4.668030 | 6.254956 | 47.430557 | 0.545660 | 0.640600 |
| 16 | 09 | `a5_tfcr-SiT-B-2-s0-r0.75-x0.5-j09-ratio075` | TFCR, ratio 0.75, cross-noise 0.5 | 31.796648 | 4.104313 | 6.375076 | 47.054230 | 0.545080 | 0.645900 |
| 17 | 11 | `a4_inv_only-SiT-B-2-s0-r0.75-x0.5-j11-inv-only-ratio075` | invariance only, ratio 0.75, coeff 0.1 | 31.892969 | 4.007992 | 6.570766 | 47.141048 | 0.546260 | 0.646900 |
| 18 | 24 | `a9_orbit_consensus-SiT-B-2-s0-r1.0-x0.5-j24-orbit-consensus-ratio1-400k` | orthogonal orbit, reliable target, velocity recomposition, ratio 1.0 | 31.974437 | 3.926524 | 6.431790 | 47.218697 | 0.542120 | 0.647900 |
| 19 | 10 | `a3_two_view-SiT-B-2-s0-r0.75-x0.5-j10-two-view-ratio075` | paired two-view control, ratio 0.75, factor losses 0 | 32.298296 | 3.602665 | 6.587726 | 47.287582 | 0.543780 | 0.645300 |
| 20 | 27 | `a12_adv_purification-SiT-B-2-s0-r1.0-x0.5-j27-adv-purification-ratio100` | A9 + timestep/orbit adversarial purification, ratio 1.0 | 32.321836 | 3.579125 | 6.362123 | 45.858646 | 0.540220 | 0.649400 |
| 21 | 05 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.0-j05-same-noise` | TFCR, ratio 0.5, cross-noise 0.0 | 33.251927 | 2.649034 | 6.372617 | 44.785519 | 0.538160 | 0.636300 |
| 22 | 04 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.5-j04-default` | TFCR, ratio 0.5, cross-noise 0.5 | 33.329855 | 2.571106 | 6.412439 | 44.192127 | 0.536740 | 0.638900 |
| 23 | 02 | `a3_two_view-SiT-B-2-s0-r0.5-x0.5-j02-two-view-control` | paired two-view control, ratio 0.5, factor losses 0 | 33.347761 | 2.553200 | 6.550316 | 44.671146 | 0.537120 | 0.642600 |
| 24 | 06 | `a5_tfcr-SiT-B-2-s0-r0.5-x1.0-j06-cross-noise` | TFCR, ratio 0.5, cross-noise 1.0 | 33.398930 | 2.502031 | 6.417396 | 44.522701 | 0.533700 | 0.641100 |
| 25 | 03 | `a4_inv_only-SiT-B-2-s0-r0.5-x0.5-j03-inv-only` | invariance only, ratio 0.5, coeff 0.1 | 33.481124 | 2.419837 | 6.514201 | 44.874249 | 0.535540 | 0.644300 |
| 26 | 08 | `a6_tfcr_transition-SiT-B-2-s0-r0.5-x0.5-j08-transition` | TFCR + transition, ratio 0.5, coeff 0.05 | 33.485353 | 2.415608 | 6.411213 | 44.630997 | 0.534340 | 0.646000 |
| 27 | 12 | `q0_invariant_three_view-SiT-B-2-s0-r0.375-linear-j12-full400k` | three-view invariant control, ratio 0.375, losses 0 | 33.682449 | 2.218512 | 6.628808 | 44.513752 | 0.530000 | 0.652600 |
| 28 | 15 | `q3_orbit_full-SiT-B-2-s0-r0.375-linear-j15-full400k` | teacher-free invariant subspace Q3 full | 34.423308 | 1.477653 | 6.737413 | 43.989368 | 0.524720 | 0.648800 |
| 29 | 14 | `q2_orbit_spread-SiT-B-2-s0-r0.375-linear-j14-full400k` | teacher-free invariant subspace Q2 | 34.478621 | 1.422340 | 6.793144 | 44.097008 | 0.527420 | 0.647500 |
| 30 | 07 | `a5_tfcr-SiT-B-2-s0-r0.25-x0.5-j07-ratio025` | TFCR, ratio 0.25, cross-noise 0.5 | 35.125785 | 0.775176 | 6.557955 | 42.306396 | 0.527560 | 0.645600 |

### External-Target and Paired Shared-Target Follow-up

A2 is the standard REPA baseline.  All S-series jobs use the A5 ratio-1.0 path.
`Delta vs A5` is `A5 FID - experiment FID`, so positive means better than the
current no-external-teacher A5 reference.  The full S-series design and per-job
interpretation are recorded in `docs/PAIRED_SHARED_TARGET_EXPERIMENTS.md`.

| Job | Experiment | Added signal | FID | Delta FID | Delta vs A5 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 73 | `u0_paired_repa-SiT-B-2-s2-r1.0-paired-repa-j73-u0-paired-repa-seed2-ratio1-coeff0.5-400000` | paired REPA seed2, no A5 factor losses | 22.573737 | 13.327224 | 7.251822 | 6.533122 | 64.813347 | 0.590380 | 0.648300 |
| 72 | `u0_paired_repa-SiT-B-2-s1-r1.0-paired-repa-j72-u0-paired-repa-seed1-ratio1-coeff0.5-400000` | paired REPA seed1, no A5 factor losses | 22.849329 | 13.051632 | 6.976230 | 6.482513 | 64.937309 | 0.592020 | 0.641600 |
| 65 | `u0_paired_repa-SiT-B-2-s0-r1.0-paired-repa-j65-u0-paired-repa-ratio1-400000` | paired REPA seed0, no A5 factor losses | 23.043364 | 12.857597 | 6.782195 | 6.611714 | 63.543480 | 0.593080 | 0.646400 |
| 61 | `u2_selective_semantic-SiT-B-2-s0-r1.0-semantic-j61-u2-selective-semantic-ratio1-400000` | selective semantic source REPA + FiLM | 23.272617 | 12.628344 | 6.552942 | 6.521849 | 63.135998 | 0.594460 | 0.646300 |
| 42 | `s1_a5_shared_repa-SiT-B-2-s0-r1.0-x0.5-j42-s1-a5-shared-repa-ratio1-400k` | external DINO shared REPA | 23.663463 | 12.237498 | 6.162096 | 6.494441 | 62.335083 | 0.588440 | 0.647100 |
| 75 | `a2_repa-SiT-B-2-s2-j75-repa-baseline-seed2-coeff0.5-400000` | standard DINOv2-B REPA seed2 | 27.483492 | 8.417469 | 2.342067 | 7.073272 | 55.317108 | 0.562560 | 0.644000 |
| 74 | `a2_repa-SiT-B-2-s1-j74-repa-baseline-seed1-coeff0.5-400000` | standard DINOv2-B REPA seed1 | 27.670844 | 8.230117 | 2.154715 | 7.192937 | 55.569725 | 0.564180 | 0.647400 |
| 51 | `a2_repa-SiT-B-2-s0-j51-repa-baseline-400k` | standard DINOv2-B REPA | 28.154142 | 7.746819 | 1.671417 | 7.220941 | 54.345894 | 0.561460 | 0.647300 |
| 59 | `u1_scheduled_repa-SiT-B-2-s0-scheduled-j59-u1-scheduled-repa-400000` | scheduled single-view REPA | 28.263778 | 7.637183 | 1.561781 | 6.618042 | 52.407143 | 0.568080 | 0.647900 |
| 44 | `s3_a5_shared_self_distill-SiT-B-2-s0-r1.0-x0.5-j44-s3-a5-shared-self-distill-ratio1-400k` | reliable self-distill | 30.180189 | 5.720772 | -0.354630 | 6.308376 | 49.576607 | 0.550020 | 0.646600 |
| 47 | `s6_a5_private_separation-SiT-B-2-s0-r1.0-x0.5-j47-s6-a5-private-separation-ratio1-400k` | evolving private separation | 30.265704 | 5.635257 | -0.440145 | 6.350427 | 49.477959 | 0.550860 | 0.647300 |
| 50 | `s7_a5_contrastive_private-SiT-B-2-s0-r1.0-x0.5-j50-s9-s7-contrastive-private-weak-ratio1-400k` | weak contrastive + private | 30.407736 | 5.493225 | -0.582177 | 6.351509 | 49.462532 | 0.551040 | 0.643900 |
| 43 | `s2_a5_shared_clean-SiT-B-2-s0-r1.0-x0.5-j43-s2-a5-shared-clean-ratio1-400k` | analytic clean consensus | 30.546366 | 5.354595 | -0.720807 | 6.379982 | 49.565720 | 0.548860 | 0.647400 |
| 49 | `s4_a5_shared_contrastive-SiT-B-2-s0-r1.0-x0.5-j49-s8-s4-contrastive-weak-ratio1-400k` | weak contrastive | 30.582932 | 5.318029 | -0.757373 | 6.418774 | 49.663208 | 0.552260 | 0.643700 |
| 45 | `s4_a5_shared_contrastive-SiT-B-2-s0-r1.0-x0.5-j45-s4-a5-shared-contrastive-ratio1-400k` | contrastive | 30.605042 | 5.295919 | -0.779483 | 6.399550 | 49.607513 | 0.552020 | 0.643600 |
| 46 | `s5_a5_shared_relation-SiT-B-2-s0-r1.0-x0.5-j46-s5-a5-shared-relation-ratio1-400k` | relation consistency | 30.676918 | 5.224043 | -0.851359 | 6.319471 | 49.011528 | 0.548100 | 0.642200 |
| 48 | `s7_a5_contrastive_private-SiT-B-2-s0-r1.0-x0.5-j48-s7-a5-contrastive-private-ratio1-400k` | contrastive + private | 30.787519 | 5.113442 | -0.961960 | 6.453773 | 49.189491 | 0.550340 | 0.645800 |

Three-seed external-target summary:

| Method | Jobs | FID mean +- std | Mean Delta FID | Mean sFID | Mean IS | Mean Precision | Mean Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| U0 paired REPA | 65, 72, 73 | 22.822143 +- 0.235991 | 13.078818 | 6.542450 | 64.431379 | 0.591827 | 0.645433 |
| A2 standard REPA | 51, 74, 75 | 27.769493 +- 0.346037 | 8.131468 | 7.162383 | 55.077576 | 0.562733 | 0.646233 |

Matched by seed, U0 improves over A2 by 5.110778 FID at seed0, 4.821515
FID at seed1, and 4.909755 FID at seed2, for a mean same-seed improvement of
4.947349 +- 0.148251 FID.

### EMA Teacher / Self-Flow Follow-up

These jobs use an internal EMA teacher rather than an external DINO target.
`Delta vs A5` is `A5 FID - experiment FID`, so positive means better than the
current teacher-free A5 reference.

| Job | Experiment | Added signal | FID | Delta FID | Delta vs A5 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 67 / SF2 | `sf2_ema_source_align-SiT-B-2-s0-r1.0-x0.5-j67-sf2-ema-source-align-ratio1-400k` | source branch aligns to EMA teacher + FiLM injection | 29.888876 | 6.012085 | -0.063317 | 6.368587 | 49.957775 | 0.552840 | 0.642500 |
| 68 / SF3 | `sf3_no_injection-SiT-B-2-s0-r1.0-x0.5-j68-sf3-no-injection-ratio1-400k` | source branch aligns to EMA teacher, no FiLM injection | 30.162065 | 5.738896 | -0.336506 | 6.321264 | 50.075989 | 0.549980 | 0.639800 |
| 69 / SF4 | `sf4_shuffle_teacher-SiT-B-2-s0-r1.0-x0.5-j69-sf4-shuffle-teacher-ratio1-400k` | shuffled EMA teacher source control | 30.198026 | 5.702935 | -0.372467 | 6.318176 | 50.163769 | 0.553840 | 0.645600 |
| 66 / SF1 | `sf1_ema_full_align-SiT-B-2-s0-r1.0-x0.5-j66-sf1-ema-full-align-ratio1-400k` | full hidden aligns to EMA teacher | 30.505773 | 5.395188 | -0.680214 | 6.360299 | 49.515400 | 0.547820 | 0.648200 |

## Analysis

The ratio-1.0 follow-up is the first clear win for dense paired trajectory supervision in this branch. A5 ratio 1.0 reaches FID 29.83, improving the no-REPA baseline by 6.08 FID and the previous best A5 ratio-0.75 run by 1.97 FID. The compute-matched A3 two-view control at ratio 1.0 is also strong at FID 30.11, improving the baseline by 5.79 FID. IS and precision move strongly upward for both A5 and A3, while A3 has the best recall among the ratio-1.0 runs.

The external-target results now have a clear multi-seed ordering.  U0 paired
REPA is the strongest current family: seed0/1/2 reach FID 23.04, 22.85, and
22.57, for a three-seed mean of 22.82 +- 0.24.  Standard A2 REPA reaches
28.15, 27.67, and 27.48, for a three-seed mean of 27.77 +- 0.35.  Matched by
seed, U0 improves over A2 by 4.95 +- 0.15 FID, so the paired same-step
trajectory REPA advantage is robust in this seed sweep rather than a seed0
accident.  Job 61 U2 selective semantic factorization reaches FID 23.27 at
seed0, beating S1 by 0.39 FID and standard A2 REPA by 4.88 FID, but trailing
the U0 seed0 control by 0.23 FID.  U2 has slightly better sFID and precision
than U0 seed0, but lower IS and worse FID, so the selective semantic path is
compatible with the paired REPA target but has not shown additive FID value
beyond the simpler U0 control.  Job 59 scheduled REPA is close to standard A2
REPA at FID 28.26, which rules out the warmup/decay schedule alone as the
explanation for the 23-FID paired runs.

The teacher-free S2--S9 set is a different story.  The best result is S3
self-distillation at FID 30.18, followed by S6 private separation at 30.27 and
S9 weak contrastive + private separation at 30.41.  None of these beats A5
ratio 1.0, and only S3 is very close to the A3 two-view control.

The EMA teacher / Self-Flow-style follow-up is also not a new best setting.
SF2 source-branch EMA alignment is the best of the completed EMA runs at FID
29.89.  It is essentially tied with A5 but still 0.06 FID worse, while beating
the A3 two-view control by 0.22 FID.  SF3 without late-block source injection
falls to FID 30.16, which means source injection helps, but only by 0.27 FID.
SF4 shuffled-teacher reaches FID 30.20, only 0.04 worse than SF3 and 0.31 worse
than SF2.  That small separation weakens the source-correspondence causality
claim inside the EMA family: the EMA target behaves more like a mild regularizer
than a strong view-consistent teacher.  SF1 full-hidden EMA alignment reaches
FID 30.51 and is worse than source-only alignment, so broadly aligning the whole
hidden state to an EMA target appears less useful than a source-branch target in
this implementation.  Most importantly, all completed EMA runs remain far behind
the DINO-targeted U0/U2 runs at FID 22.57--23.27, so the current evidence does
not support replacing paired DINO REPA with the internal EMA teacher path as the
main recipe.

The completed VGSC matrix is competitive with the strongest ratio-1.0 family, but it does not beat the A3/A5 front-runners. V2 uniform selective weighting is the best verified V-series run at FID 30.24, only 0.13 behind A3 ratio 1.0 and 0.41 behind A5 ratio 1.0, while improving over A4 inv-only ratio 1.0 by 0.42 FID. V1 clean consensus alone reaches 30.32, and V7 task-selective-only reaches 30.45, so both clean consensus and selective subspace have positive signal. V3 stability-only reaches 30.51, 0.27 worse than uniform, which suggests source-stability weighting alone is not better than a uniform selective subspace in this seed.

The negative controls are mixed. V5 shuffled-source reaches FID 30.28, nearly matching V2 uniform and beating V3, so the current result does not support a strong causal claim that correct source correspondence is required. Jobs 33, 37, and 39 currently share the exact same sample NPZ hash despite different checkpoint hashes; their metric values should be treated as a duplicate-sample artifact until those samples are regenerated. As recorded, the full VGSC/task-weighted result would not beat the simpler V1/V2 controls, but that conclusion needs a clean rerun for Jobs 33/37/39.

The clearest positive signal is now the factor batch ratio. Increasing the paired/factorized supervision ratio from 0.75 to 1.0 improves full TFCR from FID 31.80 to 29.83, A3 two-view control from 32.30 to 30.11, and inv-only from 31.89 to 30.66. Ratio 0.25 only improves by 0.78 FID, so weak factorized supervision is insufficient; in this seed, using the auxiliary trajectory path on every batch is best.

The ratio-1.0 A3 result changes the attribution. A5 still has the best FID, but it beats the compute-matched A3 control by only 0.28 FID, while A3 itself beats A4 inv-only by 0.55 FID. This means most of the gain comes from the dense paired-view trajectory training setup rather than the explicit invariance loss alone. The full persistent/evolving/recomposition objective still appears useful, but its current measured contribution over the two-view control is modest and should be treated as a smaller second-order gain until repeated across seeds.

The A9/A12/A13 follow-up does not support the v2 orbit-consensus stack as a replacement for the simpler ratio-1.0 TFCR setting. A9 reaches FID 31.97, which is 2.15 FID worse than A5 ratio 1.0, 1.87 worse than the A3 two-view ratio-1.0 compute control, and 1.31 worse than A4 inv-only ratio 1.0. A12 further degrades A9 to 32.32, so true-label adversarial purification is not helping here. The shuffled-label A13 control is better at 31.23, improving A9 by 0.74 and A12 by 1.09, but it is still 1.41 FID worse than A5 ratio 1.0 and 1.13 worse than A3 ratio 1.0. This points away from useful nuisance-label adversarial learning: any benefit in A13 looks more like generic regularization/noise than a causal improvement from the proposed adversarial labels.

The cross-noise probability is not a sensitive knob at ratio 0.5 in this seed. Same-noise, mixed-noise, and full cross-noise are all close. The transition head also does not help here; it is slightly worse than the default TFCR configuration.

The teacher-free invariant subspace follow-up did not validate the Q-series replacement idea. Q0, the three-view control, reaches FID 33.68, while Q2 and Q3 are worse at 34.48 and 34.42. This means the model can learn the invariant readout, but the discarded subspace objective does not improve generation quality here. The evidence therefore favors dense paired trajectory factorization over the separate teacher-free invariant-subspace design.

The shared-target batch gives the same directional lesson.  Once the
A2/S1/U0/U1/U2 external DINO controls are excluded, explicit teacher-free
shared-target losses are mostly neutral to negative relative to A5.
Contrastive source identity, relation matching, and analytic clean consensus do
not improve generation in this seed; weakening contrastive/private losses helps
but does not close the gap.  The less-negative signals are S3 and S6,
suggesting that softer self-distillation or protecting private/evolving
information may be worth one more targeted pass, whereas broad S4/S7
coefficient search is lower value.

### Implicit Pair Interaction Follow-up

A3 is best interpreted as controlled trajectory data augmentation: the two
views share a source sample but do not interact except through shared model
weights and summed FM gradients.  To test whether a real two-view interaction
adds value without returning to explicit persistent/evolving decomposition, the
new I-series adds implicit pair interaction and projected pair consistency.
I0/I1 use a zero-initialized pair-consensus residual adapter at the paired
hidden layer.  During paired training, each view receives an adapter update from
the pair-average hidden context; at unpaired sampling time the same adapter
falls back to self context.  I2/I3 instead keep the backbone stream
view-specific and align only a low-dimensional readout of paired features.  This
is intentionally weaker than full hidden-state alignment because paired views
usually have different timesteps/noise, and the FM objective still needs
view-specific velocity information.

| Priority | Job | Script | Purpose |
|---:|---|---|---|
| 1 | 86 / I0 | `scripts/dlc_tfcr_jobs/86_i0_a3_pair_interaction_ratio1_400k.sh` | No-REPA A3 plus implicit hidden pair interaction. This tests whether pair interaction beats pure trajectory augmentation and the current no-REPA A5/SF2 tier. |
| 2 | 87 / I1 | `scripts/dlc_tfcr_jobs/87_i1_u0_pair_interaction_repa_ratio1_400k.sh` | U0 paired REPA plus the same interaction adapter. This tests whether interaction adds to the current best paired REPA recipe. |
| 3 | 88 / I2 | `scripts/dlc_tfcr_jobs/88_i2_a3_random_align_ratio1_400k.sh` | No-REPA A3 plus fixed random-projection pair readout alignment. This tests whether a target-free shared subspace regularizer beats pure A3 without adding learnable teacher heads. |
| 4 | 89 / I3 | `scripts/dlc_tfcr_jobs/89_i3_a3_byol_align_ratio1_400k.sh` | No-REPA A3 plus BYOL-style projected pair readout alignment. This tests whether a learned target-free projector/predictor can discover shared view content while leaving timestep/noise-specific hidden capacity intact. |

### Optional Decomposition Diagnostics

The current decomposition evidence is weak.  U0 paired REPA is the strongest
current family with a three-seed mean FID of 22.82, while A5 decomposition only
adds 0.28 FID over the A3 two-view control at seed0.  Several stronger
teacher-free decomposition variants are neutral or negative.  Therefore U6/U7/U8
should be treated as optional diagnostics, not as the main next direction: they
only ask whether a much weaker or delayed regularizer can add to U0 without
disturbing the paired REPA signal.

| Priority | Job | Script | Purpose |
|---:|---|---|---|
| 1 | 82 / U8 | `scripts/dlc_tfcr_jobs/82_u8_u0_late_weak_decomp_400k.sh` | U0 paired REPA from step 0, then weak decomposition starts at 200k with 50k warmup. This is the safest test for additive value because the external target first stabilizes the representation. |
| 2 | 80 / U6 | `scripts/dlc_tfcr_jobs/80_u6_u0_weak_decomp_400k.sh` | U0 paired REPA plus weak decomposition from the start. This checks whether simply reducing factor-loss strength is enough. |
| 3 | 81 / U7 | `scripts/dlc_tfcr_jobs/81_u7_u0_recom_only_weak_400k.sh` | U0 paired REPA plus weak recomposition only. This isolates whether reconstruction consistency helps without explicit persistent/evolving targets. |

All three jobs use batch size 256, one process/GPU, 400k steps, DINOv2-B REPA
with `proj_coeff=0.5`, `factor_batch_ratio=1.0`, shared CFG dropout, and the
same train -> sample -> npz -> FID path as the existing DLC scripts.  They also
archive the relevant code/scripts into each experiment directory before running.
The expected interpretation is strict: if these jobs do not beat the U0 seed0
baseline by more than seed noise, explicit decomposition should be considered
non-additive in the current recipe.

## Takeaways

- If external representation targets are allowed, U0 paired REPA is the current best family: best single run is Job 73 at FID 22.57, and the three-seed mean is 22.82 +- 0.24.
- U0 paired REPA beats standard A2 REPA on every matched seed by 4.82--5.11 FID, with a mean same-seed gain of 4.95 +- 0.15 FID.
- Job 61 U2 selective semantic factorization is second-best at FID 23.27, beating S1 by 0.39 FID but trailing U0 by 0.23 FID.
- Standard A2 REPA reaches a three-seed mean FID of 27.77 +- 0.35 and should be included as the external-teacher baseline in all headline comparisons.
- The external-target gain is better attributed to paired same-step REPA supervision than to the A5 factor objective, because U0 removes the A5 losses and still beats S1.
- Job 59 scheduled REPA reaches FID 28.26, so REPA scheduling alone does not explain the U0/U2/S1 gains.
- EMA teacher methods do not beat A5 in the completed runs. SF2 source-only EMA alignment is the best EMA result at FID 29.89, only 0.06 behind A5 and 0.22 ahead of A3.
- SF3 no-injection trails SF2 by 0.27 FID, so source FiLM injection has a modest positive effect, but it is not enough to produce a new best run.
- SF4 shuffled-teacher is very close to SF3 and only 0.31 FID behind SF2, so EMA source-target correspondence is not strongly validated by this negative control.
- SF1 full-hidden EMA alignment is worse than source-only alignment, suggesting broad hidden-state EMA matching is too blunt in this setup.
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
- Next validation should prioritize the robust U0 paired REPA story and matched baselines. Weak/late decomposition is only an optional diagnostic; more A10/A11/A13/A14 adversarial attribution jobs are lower priority unless the A9 design is changed.
