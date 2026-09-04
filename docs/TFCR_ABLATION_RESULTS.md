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
| 1 | 09 | `a5_tfcr-SiT-B-2-s0-r0.75-x0.5-j09-ratio075` | TFCR, ratio 0.75, cross-noise 0.5 | 31.796648 | 4.104313 | 6.375076 | 47.054230 | 0.545080 | 0.645900 |
| 2 | 11 | `a4_inv_only-SiT-B-2-s0-r0.75-x0.5-j11-inv-only-ratio075` | invariance only, ratio 0.75, coeff 0.1 | 31.892969 | 4.007992 | 6.570766 | 47.141048 | 0.546260 | 0.646900 |
| 3 | 10 | `a3_two_view-SiT-B-2-s0-r0.75-x0.5-j10-two-view-ratio075` | paired two-view control, ratio 0.75, factor losses 0 | 32.298296 | 3.602665 | 6.587726 | 47.287582 | 0.543780 | 0.645300 |
| 4 | 05 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.0-j05-same-noise` | TFCR, ratio 0.5, cross-noise 0.0 | 33.251927 | 2.649034 | 6.372617 | 44.785519 | 0.538160 | 0.636300 |
| 5 | 04 | `a5_tfcr-SiT-B-2-s0-r0.5-x0.5-j04-default` | TFCR, ratio 0.5, cross-noise 0.5 | 33.329855 | 2.571106 | 6.412439 | 44.192127 | 0.536740 | 0.638900 |
| 6 | 02 | `a3_two_view-SiT-B-2-s0-r0.5-x0.5-j02-two-view-control` | paired two-view control, ratio 0.5, factor losses 0 | 33.347761 | 2.553200 | 6.550316 | 44.671146 | 0.537120 | 0.642600 |
| 7 | 06 | `a5_tfcr-SiT-B-2-s0-r0.5-x1.0-j06-cross-noise` | TFCR, ratio 0.5, cross-noise 1.0 | 33.398930 | 2.502031 | 6.417396 | 44.522701 | 0.533700 | 0.641100 |
| 8 | 03 | `a4_inv_only-SiT-B-2-s0-r0.5-x0.5-j03-inv-only` | invariance only, ratio 0.5, coeff 0.1 | 33.481124 | 2.419837 | 6.514201 | 44.874249 | 0.535540 | 0.644300 |
| 9 | 08 | `a6_tfcr_transition-SiT-B-2-s0-r0.5-x0.5-j08-transition` | TFCR + transition, ratio 0.5, coeff 0.05 | 33.485353 | 2.415608 | 6.411213 | 44.630997 | 0.534340 | 0.646000 |
| 10 | 07 | `a5_tfcr-SiT-B-2-s0-r0.25-x0.5-j07-ratio025` | TFCR, ratio 0.25, cross-noise 0.5 | 35.125785 | 0.775176 | 6.557955 | 42.306396 | 0.527560 | 0.645600 |

## Analysis

The method is useful in this run, but the current evidence is not yet decisive. Most TFCR variants improve over the no-REPA baseline, and the default TFCR setting improves FID by 2.57. However, the default TFCR result is almost tied with the two-view control, full cross-noise, and invariance-only variants. This makes it hard to attribute the gain specifically to persistent/evolving/recomposition disentanglement rather than the paired trajectory training signal itself.

The clearest positive signal is the factor batch ratio. Ratio 0.75 is substantially better than the other settings: full TFCR reaches FID 31.80, inv-only reaches 31.89, and two-view control reaches 32.30. Ratio 0.25 only improves by 0.78 FID, so too little factorized supervision appears weak.

The new ratio-0.75 controls sharpen the interpretation. Full TFCR is still the best run, but it only improves over inv-only ratio 0.75 by 0.10 FID and over two-view ratio 0.75 by 0.50 FID. This suggests that the strong ratio-0.75 result is mostly explained by stronger paired/invariance supervision, while the extra persistent/evolving/recomposition terms have at most a small marginal gain in this setting.

The cross-noise probability is not a sensitive knob at ratio 0.5 in this seed. Same-noise, mixed-noise, and full cross-noise are all close. The transition head also does not help here; it is slightly worse than the default TFCR configuration.

## Takeaways

- Keep TFCR as a plausible direction: it improves the no-REPA SiT-B/2 baseline in this setting.
- Do not overclaim the decomposition yet: the two-view control and inv-only ablations are too close to default TFCR.
- Prioritize ratio 0.75 for the next round.
- Repeat seeds for baseline, full TFCR ratio 0.75, inv-only ratio 0.75, and two-view ratio 0.75 before claiming significance.
- If budget allows, sweep around the best ratio, for example 0.625, 0.75, and 0.875.
