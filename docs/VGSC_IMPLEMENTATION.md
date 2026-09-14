# VGSC: A3-based task-selective invariance

## Purpose

VGSC keeps the empirically strongest A3 two-view training path and changes
only the auxiliary constraint.  It does not require persistent/evolving
branches, an EMA teacher, or an external representation model.  The hypothesis
is narrower than full invariance:

> source-stable directions that are useful to the velocity objective should be
> consistent across A3 views; the remaining feature space should stay free to
> represent timestep/noise variation.

All new options default to disabled.  Historical A3/A4/A5/A9/A12 commands and
checkpoints therefore retain their original parameter set and numerical path.

## Method

### 1. Analytic clean-state consensus

For the interpolant

`x_t = alpha_t x_0 + sigma_t epsilon`,
`v_t = d_alpha_t x_0 + d_sigma_t epsilon`,

the main SiT velocity prediction yields

`x0_hat = (d_sigma_t x_t - sigma_t v_hat) /
          (alpha_t d_sigma_t - sigma_t d_alpha_t)`.

This expression is implemented for both linear and cosine paths.  Two A3 views
produce two clean estimates.  Their training-source errors define detached
softmin confidences; both estimates are aligned to their detached weighted
consensus.  No second velocity decoder is introduced.

### 2. Reliable invariant subspace

A rank-`K` linear projector reads one chosen SiT block.  For every projected
direction, the gate measures

`stability_k = between_source_variance_k /
               (between_source_variance_k + within_pair_variance_k)`.

Only the projected row space receives cross-view alignment gradients.  The
orthogonal complement is not aligned.  A basis-orthogonality loss avoids
duplicate directions and an image-variance floor prevents collapse.

### 3. Task-selective weighting

The main flow-matching loss is differentiated once with respect to the selected
block activation.  Its squared projection on each subspace direction defines
velocity utility.  The main configuration uses detached weights proportional
to `stability x utility`.  The early gradient query uses `create_graph=False`,
so the normal training backward does not contain second-order derivatives.

## Key options

- `--factor-clean-consensus`: enable analytic clean-state consensus.
- `--factor-selective-invariance`: enable the low-rank projector.
- `--factor-selective-weighting {uniform,stability,task}`: select the gate.
- `--factor-selective-dim`: subspace rank; default 128 in the V-series.
- `--factor-selective-source-depth`: readout block; default 8 in the V-series.
- `--factor-selective-shuffle-targets`: wrong-source negative control.
- `--factor-selective-shuffle-utility`: wrong task-assignment control.

The new loss coefficients share the existing TFCR warmup/decay schedule.
Classifier-free dropout must be shared across paired views when either new
objective is enabled.

## DLC priority

Run the following independent one-card jobs in order.  The first four are the
minimum decision set when capacity allows nine runs.

| Priority | Job | Question |
| --- | --- | --- |
| P0a | `31_v0_a3_shared_ratio1_400k.sh` | Does shared CFG alone change A3? |
| P0b | `32_v1_clean_consensus_ratio1_400k.sh` | Is the clean target itself useful? |
| P0c | `38_v7_selective_task_only_ratio1_400k.sh` | Is task-selective subspace useful without clean consensus? |
| P0d | `33_v4_vgsc_ratio1_400k.sh` | Does full task-selective invariance beat the matched controls? |
| P0e | `39_v8_vgsc_weak_ratio1_400k.sh` | Is the default full VGSC regularization too strong? |
| P1a | `34_v3_selective_stability_ratio1_400k.sh` | Is task utility better than stability alone? |
| P1b | `35_v2_selective_uniform_ratio1_400k.sh` | Is reliability weighting better than a uniform subspace? |
| P2a | `36_v5_vgsc_shuffled_source_ratio1_400k.sh` | Is correct source correspondence causal? |
| P2b | `37_v6_vgsc_shuffled_utility_ratio1_400k.sh` | Is correct task assignment causal? |

`run_selective_by_rank.sh` maps task indices 0--8 to this exact order.  If fewer
jobs are available, submit indices 0--3 first.  Continue to P1/P2 only if the
clean-only, selective-only, or full run shows a meaningful early advantage over
V0.

## Current Results

The completed Job 32--39 matrix uses `factor_batch_ratio=1.0`,
`cross_noise_prob=0.5`, 400k training steps, 50k SDE samples, CFG 1.0,
VAE `mse`, and seed 0.  Delta is measured against the no-REPA baseline FID
35.900961.

| Job | Experiment | Purpose | FID | Delta | sFID | IS | Precision | Recall | Sample status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 35 | `v2_selective_uniform-SiT-B-2-s0-r1.0-x0.5-j35-v2-selective-uniform-ratio1-400k` | clean consensus + uniform selective subspace | 30.239068 | 5.661893 | 6.462473 | 49.687996 | 0.553380 | 0.645900 | unique NPZ |
| 36 | `v5_vgsc_shuffled_source-SiT-B-2-s0-r1.0-x0.5-j36-v5-vgsc-shuffled-source-ratio1-400k` | shuffled source correspondence | 30.281753 | 5.619208 | 6.441387 | 49.814598 | 0.551280 | 0.646000 | unique NPZ |
| 32 | `v1_clean_consensus-SiT-B-2-s0-r1.0-x0.5-j32-v1-clean-consensus-ratio1-400k` | clean consensus only | 30.315289 | 5.585672 | 6.457972 | 49.715931 | 0.551200 | 0.649600 | unique NPZ |
| 33 | `v4_vgsc-SiT-B-2-s0-r1.0-x0.5-j33-v4-vgsc-ratio1-400k` | full VGSC, task weighting | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551180 | 0.645800 | duplicate with 37/39 |
| 37 | `v6_vgsc_shuffled_utility-SiT-B-2-s0-r1.0-x0.5-j37-v6-vgsc-shuffled-utility-ratio1-400k` | shuffled utility assignment | 30.411935 | 5.489026 | 6.479884 | 49.275566 | 0.551160 | 0.645800 | duplicate with 33/39 |
| 39 | `v8_vgsc_weak-SiT-B-2-s0-r1.0-x0.5-j39-v8-vgsc-weak-ratio1-400k` | weak full VGSC | 30.411934 | 5.489027 | 6.479882 | 49.275566 | 0.551160 | 0.645800 | duplicate with 33/37 |
| 38 | `v7_selective_task_only-SiT-B-2-s0-r1.0-x0.5-j38-v7-selective-task-only-ratio1-400k` | task-selective subspace only | 30.453639 | 5.447322 | 6.376558 | 49.380562 | 0.550420 | 0.651400 | unique NPZ |
| 34 | `v3_selective_stability-SiT-B-2-s0-r1.0-x0.5-j34-v3-selective-stability-ratio1-400k` | clean consensus + stability selective subspace | 30.509307 | 5.391654 | 6.481290 | 49.351646 | 0.550620 | 0.644500 | unique NPZ |

The best verified V-series result is V2 uniform selective weighting at FID
30.239068.  It is 0.13 FID behind A3 ratio 1.0 and 0.41 behind A5 ratio 1.0,
but improves over A4 inv-only ratio 1.0 by 0.42 FID.  V1 clean consensus alone
is close at 30.315289, and V7 task-selective-only is also positive at
30.453639.  This supports the clean-consensus and selective-subspace directions
as A3-compatible regularizers, but the gains are not yet better than the
simpler A3/A5 ratio-1.0 front-runners.

The weighting story is weaker.  V3 stability-only is 0.27 FID worse than V2
uniform, so source-stability weighting alone is not better than a uniform
subspace.  V5 shuffled-source nearly matches V2, which means the current
negative control does not support a strong causal claim for correct source
correspondence.  Jobs 33, 37, and 39 need sample regeneration before comparing
full task weighting, shuffled utility, or weak VGSC: their checkpoint hashes are
different, but their sample NPZ SHA256 is identical
`143acdd1994765adecd4d12f1dcd0bce31358305a57f4420b9bed1c897ae869c`.

## Interpretation checklist

A favorable result requires more than a lower training loss:

1. V1 should improve over V0 while reducing `factor_clean_source_error`.
2. V4 should improve over V1 and over the model-capacity-matched V2/V3 runs;
   wall-clock overhead from the task-gradient query must be reported separately.
3. V4 should raise `factor_selective_source_ratio` without collapsing
   `factor_selective_image_std` or effective dimensions.
4. Shuffling source targets or utility should erase/reverse the V4 benefit.
5. Controlled feature export should show improved source predictability and
   reduced noise/timestep predictability in `invariant`, while `variant` keeps
   nuisance information.

The exporter automatically recognizes VGSC checkpoints and writes its learned
row-space coordinates as `invariant`, the orthogonal residual as `variant`, and
`invariant_kind=task_selective_subspace` for downstream analysis.
