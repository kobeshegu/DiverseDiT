# TFCR Ablation Results

## Setup

- Model: `SiT-B/2`
- Data: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907`
- Pretrained assets: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models`
- Training: 400k steps, batch size 256, seed 0, fp16, no REPA (`enc-type=none`, `proj-coeff=0`)
- Sampling: `0400000.pt`, 50k samples, `sde`, 250 steps, CFG 1.0, VAE `mse`, seed 0
- Reference: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz`
- FID graph: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/REPA/classify_image_graph_def.pb`
- Baseline for delta: `sit_b2_no_repa_baseline_seed0`, FID 35.900961
- Baseline metrics file: `/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/sampled_images/sit_b2_no_repa_baseline_seed0/SiT-B-2-0400000-size-256-vae-mse-cfg-1.0-seed-0-sde_metrics.txt`

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

| Rank | Job | Experiment | Key config | FID | Delta FID | sFID | IS | Precision | Recall |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | baseline | `sit_b2_no_repa_baseline_seed0` | no REPA baseline | 35.900961 | 0.000000 | 6.688123 | 41.130138 | 0.524620 | 0.644600 |
| 1 | 22 | `a5_tfcr-SiT-B-2-s0-r1.0-x0.5-j22-ratio100` | TFCR, ratio 1.0, cross-noise 0.5 | 29.825559 | 6.075402 | 6.331125 | 50.477547 | 0.555660 | 0.642900 |
| 2 | 23 | `a4_inv_only-SiT-B-2-s0-r1.0-x0.5-j23-inv-only-ratio100` | invariance only, ratio 1.0, coeff 0.1 | 30.659704 | 5.241257 | 6.478675 | 49.242001 | 0.549860 | 0.647200 |
| 3 | 09 | `a5_tfcr-SiT-B-2-s0-r0.75-x0.5-j09-ratio075` | TFCR, ratio 0.75, cross-noise 0.5 | 31.796648 | 4.104313 | 6.375076 | 47.054230 | 0.545080 | 0.645900 |
| 4 | 11 | `a4_inv_only-SiT-B-2-s0-r0.75-x0.5-j11-inv-only-ratio075` | invariance only, ratio 0.75, coeff 0.1 | 31.892969 | 4.007992 | 6.570766 | 47.141048 | 0.546260 | 0.646900 |
| 5 | 10 | `a3_two_view-SiT-B-2-s0-r0.75-x0.5-j10-two-view-ratio075` | paired two-view control, ratio 0.75, factor losses 0 | 32.298296 | 3.602665 | 6.587726 | 47.287582 | 0.543780 | 0.645300 |
| 6 | 05 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.0-j05-same-noise` | TFCR, ratio 0.5, cross-noise 0.0 | 33.251927 | 2.649034 | 6.372617 | 44.785519 | 0.538160 | 0.636300 |
| 7 | 04 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.5-j04-default` | TFCR, ratio 0.5, cross-noise 0.5 | 33.329855 | 2.571106 | 6.412439 | 44.192127 | 0.536740 | 0.638900 |
| 8 | 02 | `a3_two_view-SiT-B-2-s0-r0.5-x0.5-j02-two-view-control` | paired two-view control, ratio 0.5, factor losses 0 | 33.347761 | 2.553200 | 6.550316 | 44.671146 | 0.537120 | 0.642600 |
| 9 | 06 | `a5_tfcr-SiT-B-2-s0-r0.5-x1.0-j06-cross-noise` | TFCR, ratio 0.5, cross-noise 1.0 | 33.398930 | 2.502031 | 6.417396 | 44.522701 | 0.533700 | 0.641100 |
| 10 | 03 | `a4_inv_only-SiT-B-2-s0-r0.5-x0.5-j03-inv-only` | invariance only, ratio 0.5, coeff 0.1 | 33.481124 | 2.419837 | 6.514201 | 44.874249 | 0.535540 | 0.644300 |
| 11 | 08 | `a6_tfcr_transition-SiT-B-2-s0-r0.5-x0.5-j08-transition` | TFCR + transition, ratio 0.5, coeff 0.05 | 33.485353 | 2.415608 | 6.411213 | 44.630997 | 0.534340 | 0.646000 |
| 12 | 12 | `q0_invariant_three_view-SiT-B-2-s0-r0.375-linear-j12-full400k` | three-view invariant control, ratio 0.375, losses 0 | 33.682449 | 2.218512 | 6.628808 | 44.513752 | 0.530000 | 0.652600 |
| 13 | 15 | `q3_orbit_full-SiT-B-2-s0-r0.375-linear-j15-full400k` | teacher-free invariant subspace Q3 full | 34.423308 | 1.477653 | 6.737413 | 43.989368 | 0.524720 | 0.648800 |
| 14 | 14 | `q2_orbit_spread-SiT-B-2-s0-r0.375-linear-j14-full400k` | teacher-free invariant subspace Q2 | 34.478621 | 1.422340 | 6.793144 | 44.097008 | 0.527420 | 0.647500 |
| 15 | 07 | `a5_tfcr-SiT-B-2-s0-r0.25-x0.5-j07-ratio025` | TFCR, ratio 0.25, cross-noise 0.5 | 35.125785 | 0.775176 | 6.557955 | 42.306396 | 0.527560 | 0.645600 |

## Analysis

The ratio-1.0 follow-up is the first clear win for the full TFCR objective in this branch. A5 ratio 1.0 reaches FID 29.83, improving the no-REPA baseline by 6.08 FID and the previous best A5 ratio-0.75 run by 1.97 FID. IS and precision also move strongly upward, while recall is slightly lower than the inv-only setting.

The clearest positive signal is now the factor batch ratio. Increasing the paired/factorized supervision ratio from 0.75 to 1.0 improves full TFCR from FID 31.80 to 29.83, and improves inv-only from 31.89 to 30.66. Ratio 0.25 only improves by 0.78 FID, so weak factorized supervision is insufficient; in this seed, using the auxiliary trajectory signal on every batch is best.

The ratio-1.0 comparison changes the interpretation of the decomposition terms. At ratio 0.75, full TFCR only beat inv-only by 0.10 FID, which made the persistent/evolving/recomposition terms look marginal. At ratio 1.0, full TFCR beats inv-only by 0.83 FID. This suggests that the decomposition terms need dense trajectory supervision to become useful; under weaker supervision they are mostly drowned out by the paired/invariance effect.

The cross-noise probability is not a sensitive knob at ratio 0.5 in this seed. Same-noise, mixed-noise, and full cross-noise are all close. The transition head also does not help here; it is slightly worse than the default TFCR configuration.

The teacher-free invariant subspace follow-up did not validate the Q-series replacement idea. Q0, the three-view control, reaches FID 33.68, while Q2 and Q3 are worse at 34.48 and 34.42. This means the model can learn the invariant readout, but the discarded subspace objective does not improve generation quality here. The evidence therefore favors dense paired trajectory factorization over the separate teacher-free invariant-subspace design.

## Takeaways

- The current best configuration is full TFCR with `factor_batch_ratio=1.0` and `cross_noise_prob=0.5`.
- The main effect is high-ratio paired trajectory supervision; the ratio sweep is monotonic in the tested A5 settings: 0.25 -> 0.5 -> 0.75 -> 1.0.
- The full decomposition is now meaningful at ratio 1.0: A5 beats A4 inv-only by 0.83 FID.
- The Q-series teacher-free invariant-subspace alternative should be deprioritized unless a much weaker or differently placed objective is tested.
- Next validation should repeat seeds for baseline, A5 ratio 1.0, A4 ratio 1.0, and ideally A3 two-view ratio 1.0 to separate paired-view compute from the factor losses.
