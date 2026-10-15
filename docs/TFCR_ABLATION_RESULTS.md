# TFCR Ablation Results

Visual current-evidence summary: `docs/TFCR_EXPERIMENT_SUMMARY.html`.

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
- I-series interaction metrics files: Jobs 86--89 under `sampled_images/i{0,1,2,3}_*`
- J-series dynamic self-guidance metrics files: Jobs 93--102 under `sampled_images/j{1,2,3,4,5}_dynamic_*` and `sampled_images/j{6,7,8,9,10}_u0_dynamic_clean*`
- U8 late weak decomposition metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/u8_u0_late_weak_decomp-SiT-B-2-s0-r1.0-paired-repa-j82-u8-u0-late-weak-decomp-ratio1-400000/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`
- K-series teacher-transfer metrics files: Jobs 103--105 under `sampled_images/k{1,2,3}_*`
- Semantic/state metrics files: Jobs 106--111 under `sampled_images/{p2_split,p3_semantic_align,p4_identifiable}_*`; each run contains its 50K sample NPZ and `*_metrics.txt`.

Latest result visibility audit, 2026-10-15:

- Scanned current `sampled_images` and `results` for new `*_metrics.txt` files and matching run artifacts.
- Jobs 103--105 (K1--K3) completed 400k training and 50K evaluation on 2026-10-10; these results are now included below.
- No J9/J10 or K-series seed-repeat metrics were visible at audit time.
- J10 remains the best dynamic-clean single run, while U0 remains the best seed-validated family. K3 is the best teacher-reduction run, but does not match full paired REPA.
- Semantic/state Jobs 106--111 all completed 400k training and 50K evaluation.

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
Job 82 U8 late weak decomposition sample NPZ SHA256 is
`4fe4292d7703ae2a6ae0a4c0fcc04937d354c7348a1dc3e569f0588848d958d1`; its
metrics file SHA256 is
`7336d4dd689d4182c7d58ab377e28e0a760998975b52fcac5b57897da570398d`.
Job 86 I0 pair-interaction sample NPZ SHA256 is
`bb46b07399c08ae45c1c8607af94112b540ded2014ee4f0fd6389534c29af6b0`; its
metrics file SHA256 is
`4cb7230ac6bdb1109a48896fe512a706a768420985b7d4c972457d2bfc055e63`.
Job 87 I1 pair-interaction REPA sample NPZ SHA256 is
`1946e5d1df487c280c0da9e9df8b606cb437f9ff7c140dcd3d6634f9f4bc5c58`; its
metrics file SHA256 is
`211f1a6bcc9cea15e62f7fae2849d0dbc16b83ba071f875fbcf527f8f11bf23d`.
Job 88 I2 random-align sample NPZ SHA256 is
`9dd165701a81e04021037a4f258cbe25dfbaf400c064ec87370646894b8efcab`; its
metrics file SHA256 is
`e2ff0d80500c10aaec95a2aeb7e64951dc11d34eef2cc0a6c9108045915b0816`.
Job 89 I3 BYOL-align sample NPZ SHA256 is
`d20db9bdc55ac6ca8155a6453a67fd0bba81a5dbdc5553c1b4ba988e5e961584`; its
metrics file SHA256 is
`e0d89f8811735832e1dfe3120c83fc021db9a0295995df1e41d99211aae05ba7`.
Job 93 J1 dynamic clean sample NPZ SHA256 is
`40c5c8983f24700a73fd5700936d1fb8966d9d4b72817c701f8f66391fa08235`; its
metrics file SHA256 is
`51eeac1d8c84bf8756b6c7e1e2e79dcbad80e9c01c1c9741a86e7f52f5d0716e`.
Job 94 J2 dynamic orbit sample NPZ SHA256 is
`6b976ca8556a8394076bab297041a2e966bab0473b55a5c598560512ca4c10f9`; its
metrics file SHA256 is
`e752df4bcc54a4794d36d999280fdb5cffad9fcc4a9e03c1314b19262281d519`.
Job 95 J3 dynamic transport sample NPZ SHA256 is
`d628bdce340c956ec9006d4429303a3f5480ef9e6b92c37b6f5b51fe6655ffd9`; its
metrics file SHA256 is
`f2db491ec514f010993b286d2c7d3786f91fb933286ff402cd38c4ff21034da5`.
Job 96 J4 dynamic shuffled-control sample NPZ SHA256 is
`6d2782002c876f4ac7a2e073e4c49a985d3c70c7708d9d5ddc9b29c2cb705183`; its
metrics file SHA256 is
`adb8ce884e7b012caee7873c99e6c4577c1a9ef4303a43dc31d9863a36110661`.
Job 97 J5 dynamic guidance sample NPZ SHA256 is
`3f1a758a01ad1669ccc741c931d79610d2ed98060ffafa24a4fbfa9453018943`; its
metrics file SHA256 is
`58ad78a625a7b8293609fec254604e28607dac37b37d2b31341850a3a767423c`.
Job 98 J6 U0 dynamic clean sample NPZ SHA256 is
`63a2c34d1d95f59ada63dee64f9db1b0001187184cdc1d4e5836bbd6ca37bfe9`; its
metrics file SHA256 is
`dec7ad74cbe013c70722e3fd9d092f5fb0b5dfad5c833f94b722b2fe718cec5c`.
Job 99 J7 U0 dynamic clean hard-target sample NPZ SHA256 is
`b4ee4b9b9ff2fee71c7ef1548613d06be233dd916b2e8f6601d5851791573a80`; its
metrics file SHA256 is
`004ab550475e6cdbb85242f0d6d6b47aed4074dc2492409c695d36566e5050b4`.
Job 100 J8 U0 dynamic clean confident-target sample NPZ SHA256 is
`d2d3630d47fbb27de5b9f99de8a63ce873ee70ddf2b323a75e81e048c305ef58`; its
metrics file SHA256 is
`7bb3e9a4538f3ce7c76c502dfcfc9a26fda6e45d3d768ca28d0ff7afcbbc43e3`.
Job 101 J9 U0 dynamic clean late-start sample NPZ SHA256 is
`2278417d8265199557b27a49e611e5c03c89da9a1668a447ec36d2998d385e88`; its
metrics file SHA256 is
`0c312507f2a150bc40a3b36b34f6ace32082c414900baf261ba1bcf929ab30fc`.
Job 102 J10 U0 dynamic clean hard/confident late-start sample NPZ SHA256 is
`c91d8dfaff29968726366c5661a5866fd92bd74573ef05cd76c7aaf1a08f3310`; its
metrics file SHA256 is
`596336a5868257fe010ff1e9e8cca62dcfcfddc8c2d5c5e7d56adba65affbc4b`.
External-target, EMA teacher, and S/J-series paired shared-target follow-ups are
summarized after the main table; A2/S1/U0/U1/U2/U8/I1/J6--J10 use external DINO
targets, SF1--SF4 use an internal EMA teacher, and S2--S9/J1--J5 are
teacher-free.

| Rank | Job | Experiment | Key config | FID | Delta FID | sFID | IS | Precision | Recall |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | baseline | `sit_b2_no_repa_baseline_seed0` | no REPA baseline | 35.900961 | 0.000000 | 6.688123 | 41.130138 | 0.524620 | 0.644600 |
| 1 | 22 | `a5_tfcr-SiT-B-2-s0-r1.0-x0.5-j22-ratio100` | TFCR, ratio 1.0, cross-noise 0.5 | 29.825559 | 6.075402 | 6.331125 | 50.477547 | 0.555660 | 0.642900 |
| 2 | 67 | `sf2_ema_source_align-SiT-B-2-s0-r1.0-x0.5-j67-sf2-ema-source-align-ratio1-400k` | EMA teacher source align + FiLM, ratio 1.0 | 29.888876 | 6.012085 | 6.368587 | 49.957775 | 0.552840 | 0.642500 |
| 3 | 88 | `i2_a3_random_align-SiT-B-2-s0-r1.0-x0.5-j88-i2-a3-random-align-ratio1-400000` | A3 plus fixed random-projection pair readout consistency, ratio 1.0 | 30.081182 | 5.819779 | 6.464645 | 50.051022 | 0.551120 | 0.641500 |
| 4 | 30 | `a3_two_view-SiT-B-2-s0-r1.0-x0.5-j24-two-view-ratio100` | paired two-view control, ratio 1.0, factor losses 0 | 30.107791 | 5.793170 | 6.413116 | 50.217392 | 0.554620 | 0.648800 |
| 5 | 68 | `sf3_no_injection-SiT-B-2-s0-r1.0-x0.5-j68-sf3-no-injection-ratio1-400k` | EMA teacher source align, no FiLM injection, ratio 1.0 | 30.162065 | 5.738896 | 6.321264 | 50.075989 | 0.549980 | 0.639800 |
| 6 | 35 | `v2_selective_uniform-SiT-B-2-s0-r1.0-x0.5-j35-v2-selective-uniform-ratio1-400k` | V-series clean consensus + selective subspace, uniform weighting, ratio 1.0 | 30.239068 | 5.661893 | 6.462473 | 49.687996 | 0.553380 | 0.645900 |
| 7 | 36 | `v5_vgsc_shuffled_source-SiT-B-2-s0-r1.0-x0.5-j36-v5-vgsc-shuffled-source-ratio1-400k` | V4 with shuffled source targets, ratio 1.0 | 30.281753 | 5.619208 | 6.441387 | 49.814598 | 0.551280 | 0.646000 |
| 8 | 32 | `v1_clean_consensus-SiT-B-2-s0-r1.0-x0.5-j32-v1-clean-consensus-ratio1-400k` | clean consensus only, ratio 1.0 | 30.315289 | 5.585672 | 6.457972 | 49.715931 | 0.551200 | 0.649600 |
| 9 | 89 | `i3_a3_byol_align-SiT-B-2-s0-r1.0-x0.5-j89-i3-a3-byol-align-ratio1-400000` | A3 plus BYOL-style projected pair readout consistency, ratio 1.0 | 30.322469 | 5.578492 | 6.480236 | 49.703026 | 0.550560 | 0.646400 |
| 10 | 33 | `v4_vgsc-SiT-B-2-s0-r1.0-x0.5-j33-v4-vgsc-ratio1-400k` | full VGSC task weighting, ratio 1.0; duplicate sample hash with Jobs 37/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551180 | 0.645800 |
| 11 | 37 | `v6_vgsc_shuffled_utility-SiT-B-2-s0-r1.0-x0.5-j37-v6-vgsc-shuffled-utility-ratio1-400k` | V4 with shuffled utility; duplicate sample hash with Jobs 33/39 | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551160 | 0.645800 |
| 12 | 39 | `v8_vgsc_weak-SiT-B-2-s0-r1.0-x0.5-j39-v8-vgsc-weak-ratio1-400k` | weak full VGSC; duplicate sample hash with Jobs 33/37 | 30.411934 | 5.489027 | 6.479882 | 49.275566 | 0.551160 | 0.645800 |
| 13 | 38 | `v7_selective_task_only-SiT-B-2-s0-r1.0-x0.5-j38-v7-selective-task-only-ratio1-400k` | task-weighted selective subspace only, no clean consensus, ratio 1.0 | 30.453639 | 5.447322 | 6.376558 | 49.380562 | 0.550420 | 0.651400 |
| 14 | 66 | `sf1_ema_full_align-SiT-B-2-s0-r1.0-x0.5-j66-sf1-ema-full-align-ratio1-400k` | EMA teacher full hidden align, ratio 1.0 | 30.505773 | 5.395188 | 6.360299 | 49.515400 | 0.547820 | 0.648200 |
| 15 | 34 | `v3_selective_stability-SiT-B-2-s0-r1.0-x0.5-j34-v3-selective-stability-ratio1-400k` | V-series clean consensus + selective subspace, stability weighting, ratio 1.0 | 30.509307 | 5.391654 | 6.481290 | 49.351646 | 0.550620 | 0.644500 |
| 16 | 23 | `a4_inv_only-SiT-B-2-s0-r1.0-x0.5-j23-inv-only-ratio100` | invariance only, ratio 1.0, coeff 0.1 | 30.659704 | 5.241257 | 6.478675 | 49.242001 | 0.549860 | 0.647200 |
| 17 | 28 | `a13_adv_shuffled-SiT-B-2-s0-r1.0-x0.5-j28-adv-shuffled-ratio1-400k` | A12 with shuffled adversarial labels, ratio 1.0 | 31.232931 | 4.668030 | 6.254956 | 47.430557 | 0.545660 | 0.640600 |
| 18 | 09 | `a5_tfcr-SiT-B-2-s0-r0.75-x0.5-j09-ratio075` | TFCR, ratio 0.75, cross-noise 0.5 | 31.796648 | 4.104313 | 6.375076 | 47.054230 | 0.545080 | 0.645900 |
| 19 | 11 | `a4_inv_only-SiT-B-2-s0-r0.75-x0.5-j11-inv-only-ratio075` | invariance only, ratio 0.75, coeff 0.1 | 31.892969 | 4.007992 | 6.570766 | 47.141048 | 0.546260 | 0.646900 |
| 20 | 24 | `a9_orbit_consensus-SiT-B-2-s0-r1.0-x0.5-j24-orbit-consensus-ratio1-400k` | orthogonal orbit, reliable target, velocity recomposition, ratio 1.0 | 31.974437 | 3.926524 | 6.431790 | 47.218697 | 0.542120 | 0.647900 |
| 21 | 10 | `a3_two_view-SiT-B-2-s0-r0.75-x0.5-j10-two-view-ratio075` | paired two-view control, ratio 0.75, factor losses 0 | 32.298296 | 3.602665 | 6.587726 | 47.287582 | 0.543780 | 0.645300 |
| 22 | 27 | `a12_adv_purification-SiT-B-2-s0-r1.0-x0.5-j27-adv-purification-ratio100` | A9 + timestep/orbit adversarial purification, ratio 1.0 | 32.321836 | 3.579125 | 6.362123 | 45.858646 | 0.540220 | 0.649400 |
| 23 | 05 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.0-j05-same-noise` | TFCR, ratio 0.5, cross-noise 0.0 | 33.251927 | 2.649034 | 6.372617 | 44.785519 | 0.538160 | 0.636300 |
| 24 | 04 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.5-j04-default` | TFCR, ratio 0.5, cross-noise 0.5 | 33.329855 | 2.571106 | 6.412439 | 44.192127 | 0.536740 | 0.638900 |
| 25 | 02 | `a3_two_view-SiT-B-2-s0-r0.5-x0.5-j02-two-view-control` | paired two-view control, ratio 0.5, factor losses 0 | 33.347761 | 2.553200 | 6.550316 | 44.671146 | 0.537120 | 0.642600 |
| 26 | 06 | `a5_tfcr-SiT-B-2-s0-r0.5-x1.0-j06-cross-noise` | TFCR, ratio 0.5, cross-noise 1.0 | 33.398930 | 2.502031 | 6.417396 | 44.522701 | 0.533700 | 0.641100 |
| 27 | 03 | `a4_inv_only-SiT-B-2-s0-r0.5-x0.5-j03-inv-only` | invariance only, ratio 0.5, coeff 0.1 | 33.481124 | 2.419837 | 6.514201 | 44.874249 | 0.535540 | 0.644300 |
| 28 | 08 | `a6_tfcr_transition-SiT-B-2-s0-r0.5-x0.5-j08-transition` | TFCR + transition, ratio 0.5, coeff 0.05 | 33.485353 | 2.415608 | 6.411213 | 44.630997 | 0.534340 | 0.646000 |
| 29 | 12 | `q0_invariant_three_view-SiT-B-2-s0-r0.375-linear-j12-full400k` | three-view invariant control, ratio 0.375, losses 0 | 33.682449 | 2.218512 | 6.628808 | 44.513752 | 0.530000 | 0.652600 |
| 30 | 15 | `q3_orbit_full-SiT-B-2-s0-r0.375-linear-j15-full400k` | teacher-free invariant subspace Q3 full | 34.423308 | 1.477653 | 6.737413 | 43.989368 | 0.524720 | 0.648800 |
| 31 | 14 | `q2_orbit_spread-SiT-B-2-s0-r0.375-linear-j14-full400k` | teacher-free invariant subspace Q2 | 34.478621 | 1.422340 | 6.793144 | 44.097008 | 0.527420 | 0.647500 |
| 32 | 07 | `a5_tfcr-SiT-B-2-s0-r0.25-x0.5-j07-ratio025` | TFCR, ratio 0.25, cross-noise 0.5 | 35.125785 | 0.775176 | 6.557955 | 42.306396 | 0.527560 | 0.645600 |

### External-Target and Paired Shared-Target Follow-up

A2 is the standard REPA baseline.  All S-series jobs use the A5 ratio-1.0 path.
`Delta vs A5` is `A5 FID - experiment FID`, so positive means better than the
current no-external-teacher A5 reference.  The full S-series design and per-job
interpretation are recorded in `docs/PAIRED_SHARED_TARGET_EXPERIMENTS.md`.

| Job | Experiment | Added signal | FID | Delta FID | Delta vs A5 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 73 | `u0_paired_repa-SiT-B-2-s2-r1.0-paired-repa-j73-u0-paired-repa-seed2-ratio1-coeff0.5-400000` | paired REPA seed2, no A5 factor losses | 22.573737 | 13.327224 | 7.251822 | 6.533122 | 64.813347 | 0.590380 | 0.648300 |
| 102 | `j10_u0_dynamic_clean_hard_late-SiT-B-2-s0-r1.0-x0.5-j102-j10-u0-dynamic-clean-hard-conf-late100k-ratio1-400000` | U0 paired REPA + hard/confident dynamic clean, start 100k | 22.603269 | 13.297692 | 7.222290 | 6.643863 | 65.051247 | 0.595140 | 0.646400 |
| 101 | `j9_u0_dynamic_clean_late-SiT-B-2-s0-r1.0-x0.5-j101-j9-u0-dynamic-clean-late100k-ratio1-400000` | U0 paired REPA + weak soft dynamic clean, start 100k | 22.623855 | 13.277106 | 7.201704 | 6.657529 | 65.019531 | 0.594740 | 0.645700 |
| 99 | `j7_u0_dynamic_clean_hard-SiT-B-2-s0-r1.0-x0.5-j99-j7-u0-dynamic-clean-hard-ratio1-400000` | U0 paired REPA + hard reliability-selected dynamic clean | 22.724716 | 13.176245 | 7.100843 | 6.606293 | 65.030014 | 0.592960 | 0.648400 |
| 100 | `j8_u0_dynamic_clean_confident-SiT-B-2-s0-r1.0-x0.5-j100-j8-u0-dynamic-clean-confident-ratio1-400000` | U0 paired REPA + hard dynamic clean, min confidence 0.65 | 22.749460 | 13.151501 | 7.076099 | 6.588927 | 65.072975 | 0.592580 | 0.648400 |
| 98 | `j6_u0_dynamic_clean-SiT-B-2-s0-r1.0-x0.5-j98-j6-u0-dynamic-clean-ratio1-400000` | paired REPA + weak dynamic clean consensus | 22.804396 | 13.096565 | 7.021163 | 6.636861 | 64.171654 | 0.593480 | 0.639000 |
| 72 | `u0_paired_repa-SiT-B-2-s1-r1.0-paired-repa-j72-u0-paired-repa-seed1-ratio1-coeff0.5-400000` | paired REPA seed1, no A5 factor losses | 22.849329 | 13.051632 | 6.976230 | 6.482513 | 64.937309 | 0.592020 | 0.641600 |
| 65 | `u0_paired_repa-SiT-B-2-s0-r1.0-paired-repa-j65-u0-paired-repa-ratio1-400000` | paired REPA seed0, no A5 factor losses | 23.043364 | 12.857597 | 6.782195 | 6.611714 | 63.543480 | 0.593080 | 0.646400 |
| 61 | `u2_selective_semantic-SiT-B-2-s0-r1.0-semantic-j61-u2-selective-semantic-ratio1-400000` | selective semantic source REPA + FiLM | 23.272617 | 12.628344 | 6.552942 | 6.521849 | 63.135998 | 0.594460 | 0.646300 |
| 42 | `s1_a5_shared_repa-SiT-B-2-s0-r1.0-x0.5-j42-s1-a5-shared-repa-ratio1-400k` | external DINO shared REPA | 23.663463 | 12.237498 | 6.162096 | 6.494441 | 62.335083 | 0.588440 | 0.647100 |
| 82 | `u8_u0_late_weak_decomp-SiT-B-2-s0-r1.0-paired-repa-j82-u8-u0-late-weak-decomp-ratio1-400000` | U0 paired REPA + late weak decomposition | 23.918411 | 11.982550 | 5.907148 | 6.509340 | 61.781036 | 0.589500 | 0.652400 |
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
| 87 | `i1_u0_pair_interaction_repa-SiT-B-2-s0-r1.0-x0.5-j87-i1-u0-pair-interaction-repa-ratio1-coeff0.5-400000` | U0 paired REPA + hidden pair-interaction adapter | 39.463312 | -3.562351 | -9.637753 | 7.972670 | 38.316105 | 0.514240 | 0.634300 |

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
REPA is the strongest validated family: seed0/1/2 reach FID 23.04, 22.85, and
22.57, for a three-seed mean of 22.82 +- 0.24.  Standard A2 REPA reaches
28.15, 27.67, and 27.48, for a three-seed mean of 27.77 +- 0.35.  Matched by
seed, U0 improves over A2 by 4.95 +- 0.15 FID, so the paired same-step
trajectory REPA advantage is robust in this seed sweep rather than a seed0
accident.  Jobs 99--102 add dynamic clean consistency on top of U0 and all beat
the U0 seed0 control and U0 three-seed mean.  The best single dynamic-clean run
is J10 at FID 22.60, improving over U0 seed0 by 0.44 FID and over the U0 mean
by 0.22 FID, but it remains 0.03 worse than the best U0 seed2 run.  Therefore
the dynamic clean add-on is a promising single-seed signal, not yet a confirmed
additive gain beyond seed variance.  Job 61 U2 selective semantic factorization reaches
FID 23.27 at seed0, beating S1 by 0.39 FID and standard A2 REPA by 4.88 FID,
but trailing the U0 seed0 control by 0.23 FID.  U2 has slightly better sFID and
precision than U0 seed0, but lower IS and worse FID, so the selective semantic
path is compatible with the paired REPA target but has not shown additive FID
value beyond the simpler U0 control.  Job 82 U8 adds late weak decomposition on
top of U0 and reaches FID 23.92, which is still strong relative to A2 but 0.88
FID worse than the U0 seed0 control.  Job 87 I1 adds a hidden pair-interaction
adapter on top of U0 and collapses to FID 39.46, so that adapter is not
compatible with the paired REPA recipe.  Job 59 scheduled REPA is close to
standard A2 REPA at FID 28.26, which rules out the warmup/decay schedule alone
as the explanation for the 23-FID paired runs.

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

Completed results:

| Job | Script | Added interaction | FID | Delta FID | Delta vs A3 | Delta vs A5/U0 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 / I0 | `scripts/dlc_tfcr_jobs/86_i0_a3_pair_interaction_ratio1_400k.sh` | No-REPA A3 plus hidden pair-consensus residual adapter | 57.933706 | -22.032745 | -27.825915 | -28.108147 vs A5 | 7.969834 | 23.943817 | 0.416280 | 0.603000 |
| 87 / I1 | `scripts/dlc_tfcr_jobs/87_i1_u0_pair_interaction_repa_ratio1_400k.sh` | U0 paired REPA plus hidden pair-consensus residual adapter | 39.463312 | -3.562351 | n/a | -16.419948 vs U0 | 7.972670 | 38.316105 | 0.514240 | 0.634300 |
| 88 / I2 | `scripts/dlc_tfcr_jobs/88_i2_a3_random_align_ratio1_400k.sh` | No-REPA A3 plus fixed random-projection pair readout consistency | 30.081182 | 5.819779 | +0.026609 | -0.255623 vs A5 | 6.464645 | 50.051022 | 0.551120 | 0.641500 |
| 89 / I3 | `scripts/dlc_tfcr_jobs/89_i3_a3_byol_align_ratio1_400k.sh` | No-REPA A3 plus BYOL-style projected pair readout consistency | 30.322469 | 5.578492 | -0.214678 | -0.496910 vs A5 | 6.480236 | 49.703026 | 0.550560 | 0.646400 |

The result is decisive for the adapter design and weaker for the projection
design.  Directly injecting pair-average hidden context is unsafe: I0 collapses
relative to no-REPA A3, and I1 also destroys the current best U0 paired REPA
recipe.  The likely issue is that hidden-state mixing across different
timesteps/noise levels contaminates velocity-specific information instead of
only sharing semantic context.  The projected consistency versions avoid that
failure mode.  I2 is numerically the best A3-style run at FID 30.08 and beats
A3 by 0.03 FID, but it is still 0.26 FID worse than A5, so the current evidence
is a near-tie with A3 rather than a reliable new gain.  I3 is stable but trails
A3 by 0.21 FID, suggesting the BYOL-style learned projector/predictor adds
unhelpful freedom in this setup.

### Dynamic Self-Guidance Follow-up

The J-series tests whether paired trajectories can build their own guidance
target from the model's predicted diffusion dynamics instead of using external
DINO targets or explicit persistent/evolving decomposition.  J1 aligns the two
recovered clean estimates to a confidence-weighted clean consensus, J2 adds
orbit-aware clean/noise/velocity consistency, J3 uses one view's recovered clean
state to synthesize the paired destination state, J4 shuffles the clean target
as a source-correspondence control, J5 trains and samples with a self-predicted
clean guidance token, J6 adds a weak dynamic clean loss on top of U0 paired
REPA, and J7--J10 refine that U0 add-on with hard target selection,
confidence filtering, and delayed activation.

| Job | Script | Added dynamic signal | FID | Delta FID | Delta vs A3/A5/U0 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 93 / J1 | `scripts/dlc_tfcr_jobs/93_j1_dynamic_clean_ratio1_400k.sh` | No-REPA A3 plus confidence-weighted dynamic clean consensus | 30.466300 | 5.434661 | -0.358509 vs A3, -0.640741 vs A5 | 6.460874 | 49.849434 | 0.544760 | 0.644900 |
| 94 / J2 | `scripts/dlc_tfcr_jobs/94_j2_dynamic_orbit_ratio1_400k.sh` | No-REPA A3 plus orbit-aware clean/noise/velocity consistency | 35.038628 | 0.862333 | -4.930837 vs A3, -5.213069 vs A5 | 7.067249 | 43.020679 | 0.524420 | 0.644100 |
| 95 / J3 | `scripts/dlc_tfcr_jobs/95_j3_dynamic_transport_ratio1_400k.sh` | No-REPA A3 plus dynamic cross-state transport supervision | 30.656919 | 5.244042 | -0.549128 vs A3, -0.831360 vs A5 | 6.474069 | 49.683128 | 0.547240 | 0.645700 |
| 96 / J4 | `scripts/dlc_tfcr_jobs/96_j4_dynamic_shuffle_ratio1_400k.sh` | J1 with shuffled dynamic clean targets | 30.142931 | 5.758030 | -0.035140 vs A3, -0.317372 vs A5 | 6.407465 | 50.279217 | 0.553480 | 0.643300 |
| 97 / J5 | `scripts/dlc_tfcr_jobs/97_j5_dynamic_guidance_ratio1_400k.sh` | Train and sample with self-predicted clean guidance token | 92.470211 | -56.569250 | -62.362420 vs A3, -62.644652 vs A5 | 45.977780 | 14.248680 | 0.256460 | 0.536000 |
| 98 / J6 | `scripts/dlc_tfcr_jobs/98_j6_u0_dynamic_clean_ratio1_400k.sh` | U0 paired REPA plus weak dynamic clean consensus | 22.804396 | 13.096565 | +0.238968 vs U0 seed0, +0.017747 vs U0 mean | 6.636861 | 64.171654 | 0.593480 | 0.639000 |
| 99 / J7 | `scripts/dlc_tfcr_jobs/99_j7_u0_dynamic_clean_hard_ratio1_400k.sh` | J6 with hard reliability-selected dynamic clean target | 22.724716 | 13.176245 | +0.318648 vs U0 seed0, +0.097427 vs U0 mean | 6.606293 | 65.030014 | 0.592960 | 0.648400 |
| 100 / J8 | `scripts/dlc_tfcr_jobs/100_j8_u0_dynamic_clean_confident_ratio1_400k.sh` | J7 plus min-confidence 0.65 filtering | 22.749460 | 13.151501 | +0.293904 vs U0 seed0, +0.072683 vs U0 mean | 6.588927 | 65.072975 | 0.592580 | 0.648400 |
| 101 / J9 | `scripts/dlc_tfcr_jobs/101_j9_u0_dynamic_clean_late_ratio1_400k.sh` | J6 soft dynamic clean target, auxiliary loss starts at 100k | 22.623855 | 13.277106 | +0.419509 vs U0 seed0, +0.198288 vs U0 mean | 6.657529 | 65.019531 | 0.594740 | 0.645700 |
| 102 / J10 | `scripts/dlc_tfcr_jobs/102_j10_u0_dynamic_clean_hard_late_ratio1_400k.sh` | Hard/confident dynamic clean target, auxiliary loss starts at 100k | 22.603269 | 13.297692 | +0.440095 vs U0 seed0, +0.218874 vs U0 mean | 6.643863 | 65.051247 | 0.595140 | 0.646400 |

The teacher-free dynamic objectives do not produce a convincing improvement
over the A3/A5 no-REPA front-runners.  J4, the shuffled-target negative
control, is the best no-REPA J-series run at FID 30.14 and nearly matches A3,
while the correct clean-consensus J1 is worse at 30.47.  That inversion is
negative evidence for a causal source-correspondence effect in the current
self-consensus target.  J3 transport is stable but worse than A3/A5, and J2
almost falls back to the no-REPA baseline quality, so orbit-aware dynamic
clean/noise/velocity consistency appears over-constrained.  J5 is the clearest
failure: feeding a self-predicted clean token back into every sampling step
causes a severe quality collapse, likely because first-pass clean-estimate
errors are amplified by the second pass.

The U0 dynamic-clean add-on is now consistently positive at seed0.  J6 reaches
FID 22.80, J7/J8 improve hard/confident target selection to 22.72/22.75, and
J9/J10 improve further to 22.62/22.60 when the auxiliary dynamic-clean loss is
delayed until 100k steps.  This pattern suggests that early self-target noise is
the main bottleneck, while hard target selection and confidence filtering are
secondary.  J10 is the best dynamic-clean run and is only 0.03 FID worse than
the best U0 seed2 run, with the highest IS and precision among the U0-family
runs.  The conservative conclusion is that dynamic clean consistency is
compatible with paired REPA and may add a small gain, but seed repeats are
required before claiming a statistically reliable improvement over U0.

### External-Target Transfer and Dropout Follow-up

Jobs 103--105 test whether the external DINO target can supervise fewer views
or leave training after bootstrapping. All three use ratio 1.0, seed 0, and a
hard clean-space asymmetric loss of 0.025 starting at 100k. K1 gives REPA only
to the first paired view, K2 removes REPA entirely, and K3 starts with paired
REPA then raises target dropout from zero to one between 100k and 250k.

| Job | Script | External-target schedule | FID | Delta FID | Delta vs U0 seed0 | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 103 / K1 | `scripts/dlc_tfcr_jobs/103_k1_one_view_repa_transfer_400k.sh` | REPA on first paired view only; late hard clean transfer | 23.787075 | 12.113886 | -0.743711 | 6.684746 | 62.585499 | 0.589440 | 0.643200 |
| 104 / K2 | `scripts/dlc_tfcr_jobs/104_k2_clean_asym_late_400k.sh` | No REPA; late hard clean consistency only | 30.344000 | 5.556961 | -7.300636 | 6.527787 | 50.254082 | 0.551360 | 0.654100 |
| 105 / K3 | `scripts/dlc_tfcr_jobs/105_k3_repa_dropout_clean_400k.sh` | Paired REPA bootstrap; target fully dropped by 250k | 23.413834 | 12.487127 | -0.370470 | 6.566337 | 63.366932 | 0.590960 | 0.640700 |

K2 falls back to the A3/A5 quality range, so late clean consistency does not
replace DINO semantic supervision. K1 retains much of the paired-REPA gain but
trails U0 by 0.74 FID. K3 improves K1 by 0.37 FID, showing that paired-teacher
bootstrapping followed by dropout is better than supervising only one view for
the entire run, but it still trails full U0 by 0.37 and J10 by 0.81 FID. These
are single-seed results; K3 seed repeats and dropout-schedule controls are
required before making a teacher-transfer claim.

### Optional Decomposition Diagnostics

The current decomposition evidence is weak.  U0 paired REPA is the strongest
current family with a three-seed mean FID of 22.82, while A5 decomposition only
adds 0.28 FID over the A3 two-view control at seed0.  Several stronger
teacher-free decomposition variants are neutral or negative.  U8 now directly
tests whether a much weaker or delayed regularizer can add to U0 without
disturbing the paired REPA signal.

| Job | Script | Purpose | Status | FID | Delta vs U0 seed0 |
|---:|---|---|---|---:|---:|
| 82 / U8 | `scripts/dlc_tfcr_jobs/82_u8_u0_late_weak_decomp_400k.sh` | U0 paired REPA from step 0, then weak decomposition starts at 200k with 50k warmup | completed | 23.918411 | -0.875047 |
| 80 / U6 | `scripts/dlc_tfcr_jobs/80_u6_u0_weak_decomp_400k.sh` | U0 paired REPA plus weak decomposition from the start | not prioritized after U8 | n/a | n/a |
| 81 / U7 | `scripts/dlc_tfcr_jobs/81_u7_u0_recom_only_weak_400k.sh` | U0 paired REPA plus weak recomposition only | not prioritized after U8 | n/a | n/a |

U8 is still much better than standard A2 REPA, but it is 0.88 FID worse than
the U0 seed0 control and worse than all three U0 seed repeats.  This makes the
current explicit decomposition path non-additive once the representation is
already stabilized by paired REPA.  U6/U7 are lower priority unless there is a
new decomposition design that avoids constraining timestep/noise-specific
velocity information.

### Semantic/State Representation Follow-up

Jobs 106--111 isolate three changes: P2 adds a fixed 128-dimensional semantic/state split, P3 moves alignment to the semantic branch, and P4 adds cross-covariance, anti-collapse, and zero/swap necessity constraints. REPA and no-REPA variants are paired; all six runs completed 400k training and 50K evaluation.

| Job | Configuration | Status | FID | Delta vs no-REPA baseline | sFID | IS | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 106 | P2 semantic/state split + full-hidden REPA | evaluated | 23.531836 | 12.369125 | 6.457949 | 63.032265 | 0.589820 | 0.645800 |
| 107 | P2 semantic/state split, no REPA | evaluated | 29.694973 | 6.205988 | 6.303665 | 50.227348 | 0.554840 | 0.641900 |
| 108 | P3 semantic-only DINO alignment | evaluated | 23.789724 | 12.111237 | 6.465170 | 62.758461 | 0.587420 | 0.644300 |
| 109 | P3 same-source semantic alignment, no REPA | evaluated | 29.688769 | 6.212192 | 6.343803 | 50.672741 | 0.556220 | 0.648300 |
| 110 | P4 identifiable semantic/state + REPA | evaluated | 23.980545 | 11.920416 | 6.554770 | 62.142963 | 0.589220 | 0.643600 |
| 111 | P4 identifiable semantic/state, no REPA | evaluated | 29.637261 | 6.263700 | 6.359669 | 50.226261 | 0.557240 | 0.650300 |

The full matrix does not establish an improvement. Job 106 trails U0 seed0 by 0.488472 FID, and semantic-only alignment in Job 108 is another 0.257888 worse than Job 106. Adding P4 identifiability makes Job 110 another 0.190821 worse than Job 108 and 0.937181 worse than U0. On the teacher-free side, Job 109 differs from Job 107 by only 0.006204 FID, while Job 111 improves Job 107 by just 0.057712. P4 does make the final-minibatch cross-covariance small and all zero/swap gaps positive, so the mechanism changes its readouts as intended, but this does not translate into generation gains. These are generation-compatibility results only. No Jobs 106--111 controlled representation reports were present at audit time, so semantic probes, retrieval, effective rank, and CKA remain required before making a representation-learning claim.

## Takeaways

- Jobs 106--111 do not show a meaningful gain from the semantic/state split, semantic-only alignment, or identifiability constraints. Job 106 is 0.49 FID worse than U0; Job 108 is worse than 106; Job 110 is worse than 108; and the teacher-free changes are nearly neutral. Representation probes are still missing, so no representation-learning improvement is claimed.
- If external representation targets are allowed, U0 paired REPA is the current best validated family: best validated single run is Job 73 at FID 22.57, and the three-seed mean is 22.82 +- 0.24.
- U0 paired REPA beats standard A2 REPA on every matched seed by 4.82--5.11 FID, with a mean same-seed gain of 4.95 +- 0.15 FID.
- Jobs 99--102 show a stronger U0 dynamic-clean signal than J6.  J10 reaches FID 22.60, improving U0 seed0 by 0.44 FID and the U0 three-seed mean by 0.22 FID, but it is still 0.03 worse than U0 seed2, so it is promising but not yet a confirmed additive gain.
- Jobs 103--105 show that reducing DINO supervision is possible but not free. K3 teacher dropout is best at FID 23.41, 0.37 worse than full U0 seed0; K1 one-view supervision is 23.79, and K2 without DINO falls to 30.34. Late clean transfer therefore does not replace the external target under the current settings.
- Within the J6 follow-up, late start matters more than hard/confidence filtering: J9/J10 at 22.62/22.60 beat J7/J8 at 22.72/22.75, suggesting early self-target noise is the main issue.
- Job 61 U2 selective semantic factorization is second-best at FID 23.27, beating S1 by 0.39 FID but trailing U0 by 0.23 FID.
- Job 82 U8 late weak decomposition reaches FID 23.92, which is still strong but 0.88 FID worse than U0 seed0, so weak/late explicit decomposition is not additive to paired REPA.
- Standard A2 REPA reaches a three-seed mean FID of 27.77 +- 0.35 and should be included as the external-teacher baseline in all headline comparisons.
- The external-target gain is better attributed to paired same-step REPA supervision than to the A5 factor objective, because U0 removes the A5 losses and still beats S1.
- Job 59 scheduled REPA reaches FID 28.26, so REPA scheduling alone does not explain the U0/U2/S1 gains.
- EMA teacher methods do not beat A5 in the completed runs. SF2 source-only EMA alignment is the best EMA result at FID 29.89, only 0.06 behind A5 and 0.22 ahead of A3.
- SF3 no-injection trails SF2 by 0.27 FID, so source FiLM injection has a modest positive effect, but it is not enough to produce a new best run.
- SF4 shuffled-teacher is very close to SF3 and only 0.31 FID behind SF2, so EMA source-target correspondence is not strongly validated by this negative control.
- SF1 full-hidden EMA alignment is worse than source-only alignment, suggesting broad hidden-state EMA matching is too blunt in this setup.
- Among no-external-teacher runs, the current best configuration remains full TFCR with `factor_batch_ratio=1.0` and `cross_noise_prob=0.5`.
- I2 fixed random-projection pair consistency is the best no-REPA interaction diagnostic at FID 30.08, only 0.03 better than A3 but 0.26 worse than A5; this is a stable near-tie, not a convincing improvement.
- I3 BYOL-style pair consistency is stable at FID 30.32 but trails A3 by 0.21 FID, while I0/I1 hidden pair-context adapters collapse badly.
- The no-REPA teacher-free dynamic self-guidance set does not beat A3/A5.  J4 shuffled dynamic clean is the best no-REPA J-series run at FID 30.14, while correct-target J1 is worse at 30.47, so the no-REPA dynamic clean target does not show source-correspondence causality.
- J2 dynamic orbit consistency is over-constrained at FID 35.04, J3 transport is stable but worse than A3/A5 at 30.66, and J5 self-guided sampling collapses to FID 92.47.
- The main effect is high-ratio paired trajectory supervision; the ratio sweep is monotonic in the tested A5 and A3 settings.
- The full decomposition is positive but modest at ratio 1.0: A5 beats A3 two-view control by 0.28 FID and A4 inv-only by 0.83 FID.
- The new teacher-free shared-target variants do not beat A5; S3 self-distill is the closest at 30.18 FID and S6 private separation is second at 30.27 FID.
- The V-series controls are strong but not yet better than A3/A5: V2 uniform is the best verified V-series result, V1 clean consensus is close, V7 selective-only is also positive, and V3 stability-only is worse than uniform.
- V5 shuffled-source nearly matches V2, so correct source correspondence is not yet causally supported by the current negative-control results.
- Jobs 33/37/39 must be regenerated before using them to compare full VGSC, shuffled utility, and weak VGSC; their current sample NPZ files are byte-identical.
- Invariance-only is not the main explanation for the improvement in this seed, because A3 two-view control ratio 1.0 beats A4 inv-only ratio 1.0 by 0.55 FID.
- A9 orbit consensus is not competitive in its current setting. A12 true-label adversarial purification worsens it, while A13 shuffled labels improve over A9/A12 but still trail A5/A3/A4 ratio 1.0.
- The Q-series teacher-free invariant-subspace alternative should be deprioritized unless a much weaker or differently placed objective is tested.
- Next validation should prioritize seed repeats for J10/J9 against U0, plus mechanism probes for paired representation guidance.  More weak/late decomposition, no-REPA dynamic self-guidance, or A10/A11/A13/A14 adversarial attribution jobs are lower priority unless the design is changed.
