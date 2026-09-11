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

Run the following independent one-card jobs in order.  The first three are the
minimum decision set.

| Priority | Job | Question |
| --- | --- | --- |
| P0a | `31_v0_a3_shared_ratio1_400k.sh` | Does shared CFG alone change A3? |
| P0b | `32_v1_clean_consensus_ratio1_400k.sh` | Is the clean target itself useful? |
| P0c | `33_v4_vgsc_ratio1_400k.sh` | Does full task-selective invariance beat the matched controls? |
| P1a | `34_v3_selective_stability_ratio1_400k.sh` | Is task utility better than stability alone? |
| P1b | `35_v2_selective_uniform_ratio1_400k.sh` | Is reliability weighting better than a uniform subspace? |
| P2a | `36_v5_vgsc_shuffled_source_ratio1_400k.sh` | Is correct source correspondence causal? |
| P2b | `37_v6_vgsc_shuffled_utility_ratio1_400k.sh` | Is correct task assignment causal? |

`run_selective_by_rank.sh` maps task indices 0--6 to this exact order.  Submit
indices 0--2 first.  Continue to P1/P2 only if the clean-only or full run shows
a meaningful early advantage over V0.

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
