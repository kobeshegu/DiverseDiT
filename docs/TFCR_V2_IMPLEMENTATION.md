# TFCR v2: Orbit-Consensus Target and Task-Sufficient Recomposition

TFCR v2 adds three opt-in components to the existing two-view factorization
path. All new switches default to off (or to `legacy`), so the historical
A3/A4/A5 commands and checkpoints retain their original architecture and loss.

## 1. Controlled orbit construction

`--factor-orbit-mode` supports:

- `legacy`: both views have different timesteps; noise changes with
  `--factor-pair-cross-noise-prob`.
- `orthogonal`: each pair is either time-only (same noise, different timestep)
  or noise-only (same timestep, different noise).
- `time-only` and `noise-only`: intervention ablations.

`--factor-orbit-noise-only-prob` selects the noise-only fraction in the
orthogonal mixture. `--factor-share-cfg-dropout` makes both views reuse one
classifier-free label-drop decision.

## 2. Reliability-selected invariant target

`--factor-reliable-target` scores every target channel using a detached ratio
of between-source variance to between-source plus within-orbit variance. The
within-orbit term is the worst observed value across time-only, noise-only,
and joint intervention groups, so a direction must be stable to every sampled
nuisance rather than only stable on average. The
top `--factor-reliability-keep-ratio` channels form the persistent target;
the remainder is assigned to the evolving residual so additive recomposition
still reconstructs the complete deep feature.

`--factor-reliability-floor` optionally excludes weak directions before the
top-channel selection. Distributed
training gathers the detached image embeddings before estimating the score,
so every rank applies the same gate.

## 3. Task-sufficient recomposition

`--factor-velocity-recomposition` adds two training-only additive decoders. The
persistent decoder predicts the pair-common velocity, while the evolving
decoder predicts the current-view residual conditioned on its timestep. Their
sum uses the other view's persistent component and the current view's evolving
component to reconstruct the current flow-matching velocity. Direct component
losses prevent either branch from being ignored; the balanced objective uses
the single `--factor-velocity-recom-coeff` weight. The decoders are retained in
checkpoints for strict loading but are never called during sampling.

## Main experiment

```bash
bash scripts/dlc_tfcr_jobs/24_a9_orbit_consensus_ratio100.sh
```

The job uses ratio 1.0, a 50/50 orthogonal orbit, shared CFG dropout, a 75%
reliability-selected target, and velocity-recomposition coefficient 0.05.
Environment variables can override all settings except the job's defining
ratio and orbit mode.

## 4. Adversarial orbit purification (TFCR v3, opt-in)

`--factor-adversarial` adds four training-only classifiers to the TFCR head:

- a timestep-bin discriminator on spatially pooled persistent codes;
- a time-only/noise-only orbit discriminator on persistent pairs;
- matched timestep and orbit probes on evolving codes.

Persistent discriminator inputs pass through a gradient reversal layer (GRL).
The classifiers therefore minimize their cross-entropy normally while the
persistent projector receives its sign-reversed gradient. Evolving probes use
ordinary gradients, encouraging nuisance information to move to the evolving
factor rather than disappear from the representation. Pair classifiers consume
`[abs(z_a - z_b), z_a * z_b]` and never receive timestep or orbit metadata as
input.

The option requires `--factor-orbit-mode orthogonal`, because its binary orbit
label is only causally interpretable when exactly one nuisance changes. The
training entrypoint also requires `--factor-share-cfg-dropout` whenever CFG
dropout is enabled, preventing an unlabelled conditioning intervention, and
requires a strictly interior noise-only probability so both orbit classes are
observed. The
maximum reverse-gradient strength is
`--factor-adversarial-grl-scale`; it stays zero until
`--factor-adversarial-start-steps` and then ramps for
`--factor-adversarial-warmup-steps`. Setting the maximum to zero while setting
both evolving-probe coefficients to zero trains a true critic-only diagnostic:
persistent discriminator parameters update, but no discriminator gradient
reaches the representation.

Timestep and orbit paths have independent coefficients:

```text
--factor-adv-persistent-time-coeff
--factor-adv-persistent-orbit-coeff
--factor-probe-evolving-time-coeff
--factor-probe-evolving-orbit-coeff
```

`--factor-adversarial-shuffle-labels` provides a compute- and architecture-
matched random-label control. All adversarial flags default off, preserving
older commands, initialization, and checkpoint structure.

### Experiment matrix

```bash
# Single-nuisance attribution
bash scripts/dlc_tfcr_jobs/25_a10_adv_time_ratio100.sh
bash scripts/dlc_tfcr_jobs/26_a11_adv_orbit_ratio100.sh

# Main treatment
bash scripts/dlc_tfcr_jobs/27_a12_adv_purification_ratio100.sh

# Controls
bash scripts/dlc_tfcr_jobs/28_a13_adv_shuffled_ratio100.sh
bash scripts/dlc_tfcr_jobs/29_a14_critic_only_ratio100.sh
```

The primary diagnostics are persistent/evolving timestep and orbit accuracy,
their evolving-minus-persistent separation gaps, and the logged majority-class
baselines (the paired-time marginal is not guaranteed to be uniform). Low
persistent accuracy alone is not success: source variation, direct persistent
velocity prediction, recomposition, and FID must remain healthy to rule out
representation collapse.
