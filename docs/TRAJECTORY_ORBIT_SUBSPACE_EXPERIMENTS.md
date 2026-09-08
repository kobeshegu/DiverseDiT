# Teacher-Free Trajectory-Orbit Invariant Subspace

## 0. Current status

The Q-series teacher-free invariant-subspace experiment did not beat its
matched three-view control in the completed seed-0 400k runs. Q0 reaches FID
33.68, while Q2 and Q3 reach 34.48 and 34.42. The result suggests that the
discarded invariant readout can be optimized, but this separate subspace
objective does not translate into better generation quality in the current
configuration.

The strongest result is instead the dense paired trajectory-factorization
setting: A5 TFCR with `factor_batch_ratio=1.0` reaches FID 29.83, while A4
inv-only at ratio 1.0 reaches FID 30.66. Current evidence therefore favors
full-batch paired trajectory supervision, and the Q-series should be treated as
a negative follow-up unless it is redesigned with weaker coefficients, a
different source block, or direct coupling to the denoising representation.

## 1. Why this replaces explicit Persistent/Evolving factorization

The completed ratio-0.75 controls show that paired trajectory training explains
most of TFCR's gain: FID improves from 35.90 to 32.30 with paired views alone,
to 31.89 with invariant alignment, and to 31.80 with the full factorization.
The explicit Persistent/Evolving/recomposition machinery therefore adds only
about 0.10 FID beyond invariant-only at ratio 0.75. At ratio 1.0, however,
full TFCR improves to FID 29.83 and beats inv-only by 0.83 FID, so the
decomposition terms appear useful when the paired trajectory signal is dense.

The new mode learns only a low-dimensional readout of an intermediate DiT
feature.  Its complement remains unconstrained and can retain timestep, noise,
and velocity information needed by the denoiser.  It is teacher-free: the
training EMA remains checkpoint/evaluation machinery and is never used as an
alignment target.

## 2. Construction

For each selected source latent `x0`, construct three online views together:

- **anchor:** `(x_t, epsilon_1)`;
- **time intervention:** `(x_t', epsilon_1)`, changing only timestep;
- **noise intervention:** `(x_t, epsilon_2)`, changing only noise.

All three share the source latent, class condition, and CFG-drop decision. Their
common information is the source/condition pair; the method therefore does not
depend on different minibatch examples to combine time and noise invariance.

At block `l`, a bias-free linear map with row-normalized basis `W` reads a
`d_inv`-dimensional subspace `z = W h_l`. It does not feed the denoising head.
An orthogonality penalty prevents redundant basis vectors. An MLP readout is
available only as an explicit shortcut-capacity ablation. The main objective is

```text
L = L_FM
  + lambda_t   L_time-orbit
  + lambda_eps L_noise-orbit
  + lambda_img L_image-variance
  + lambda_sp  L_spatial-variance
  + lambda_cov L_covariance
  + lambda_basis L_basis-orthogonality
  + lambda_rel L_local-relation.
```

`L_time-orbit` and `L_noise-orbit` are cosine distances to the anchor. The mean
of all three online views forms the anti-collapse consensus; no view is a
stopped-gradient or EMA teacher. Because a teacher-free input contains almost
no source evidence near pure noise, training views are restricted to `t <= 0.8`
and both alignment losses are weighted by their clean-source reliability
`alpha(t)^2 / (alpha(t)^2 + sigma(t)^2)`. Image variance prevents all sources
from sharing one code,
spatial variance prevents all patches within an image from sharing one code,
and covariance discourages redundant invariant channels.  Local-relation loss
preserves horizontal and vertical patch-neighbour cosine relations without a
dense token Gram matrix.

This is distinct from SRA-style trajectory distillation: there is no
shallow/high-noise to deep/cleaner target, no asymmetric teacher, and no
full-feature matching.  The causal variables are independently intervened on,
and only a declared subspace is regularized.

## 3. Default setting

| Item | Value |
|---|---:|
| backbone | SiT-B/2, no external encoder |
| seed | 0 |
| three-view source ratio | 0.375 |
| effective backbone forwards | 1 + 2 × 0.375 = 1.75 per source |
| time gap | Uniform(0.05, 0.20) |
| maximum supervised timestep | 0.8 |
| source-reliability power | 1.0 |
| source block | 4 |
| invariant projector | row-normalized linear, dimension 256 |
| time/noise coefficients | 0.10 / 0.10 |
| image/spatial variance coefficients | 0.02 / 0.02 |
| covariance/basis/relation coefficients | 0.001 / 0.01 / 0.05 |
| variance targets | image 1.0, spatial 0.5 |
| warmup | 10k steps |

## 4. Full 400k comparison

Submit the `*_400k.sh` job scripts one by one when the scheduler does not
support array tasks.  Each task trains to 400k steps and then runs the shared
train -> sample -> npz -> FID pipeline.  `run_invariant_full_by_rank.sh` is only
a convenience launcher for array-style submission.

| Job | Setting | Question answered |
|---:|---|---|
| 12 / Q0 | three views, all auxiliary weights zero | compute- and sampler-matched control |
| 13 / Q1 | time + noise consistency | does subspace invariance add value beyond pairing? |
| 14 / Q2 | Q1 + image/spatial spread + covariance + basis | is non-collapsed source information required? |
| 15 / Q3 | Q2 + local relation | does spatially structured invariance help generation? |
| 16 | Q1, time loss only | contribution of timestep invariance |
| 17 | Q1, noise loss only | contribution of noise invariance |
| 18 | Q3, source block 8 | early/mid versus later-layer placement |
| 19 | Q3, invariant dimension 128 | whether the default subspace is too wide |
| 20 | Q3, nonlinear MLP readout | can an expressive discarded head absorb the objective? |

Interpret the ladder causally: Q1-Q0 is the net invariant-alignment effect,
Q2-Q1 is the anti-collapse/source-retention effect, and Q3-Q2 is the local
structure effect.  Job 16 versus 17 identifies which nuisance invariance is
actually useful; it must not be inferred from the mixed run alone.

## 5. Selection and confirmation

Select one configuration only after all runs use the same 400k checkpoint,
sampler, sample count, reference statistics, and seed.  A practical gate is:

1. the selected method beats Q0 by at least 0.3 FID at 400k;
2. sFID and recall do not materially regress;
3. `invariant_source_ratio` rises without image/spatial standard deviations
   collapsing toward zero;
4. the result is supported by the Q0/Q1/Q2/Q3 ladder rather than one isolated
   tuning run.

Then repeat Q0 and Q3 at seeds 1 and 2 with
`run_invariant_confirm_by_rank.sh` (or replace Q3 in that runner if another
configuration wins).  Script `21_q3_orbit_full_main_400k.sh` is an optional Q3
variant with a 10k warmup and late auxiliary-scale decay from 300k to 400k,
ending at 0.1, so late denoising specialization is not dominated by the
regularizer.

## 6. Diagnostics needed for the paper claim

Besides FID, sFID, IS, precision, and recall, retain these logged quantities:

- time/noise similarity and local-relation gap, reported separately;
- within-orbit energy, between-source energy, and their source ratio;
- image and spatial standard deviation plus covariance off-diagonal energy;
- basis orthogonality and the mean source-reliability weight;
- held-out-source timestep probes and within-source noise-trajectory retrieval;
- invariant-subspace capture of source, timestep, and noise energy;
- same-class instance retrieval, which rules out a class-condition shortcut;
- the same probes across blocks and timestep bins after training.

For the eventual REPA/SRA/Self-Flow comparison, freeze each trained backbone
and run matched linear probes for class/source identity, timestep, and noise
identity at every block and timestep bin.  The invariant claim is supported by
higher source predictability and cross-t/cross-noise retrieval with lower
timestep/noise predictability in the selected subspace, while the backbone
complement should retain state information.  This is the analysis that can
distinguish what each method makes invariant from a generic FID improvement.

## 7. Failure-directed follow-ups

- If Q1 does not beat Q0 and similarity saturates immediately, reduce both
  consistency coefficients to 0.05 or shorten the warmup; the gradient is too
  strong or too easy for the projector.
- If similarity improves but image/spatial standard deviation collapses, raise
  the corresponding variance coefficient to 0.05 before changing the model.
- If Q2 is healthy internally but FID is worse, move the source block earlier
  (3) or reduce `d_inv` to 128/64; the constrained subspace is probably too
  entangled with denoising state.
- If only one intervention helps, lower the losing coefficient to 0.05 before
  removing its view; retaining three views keeps the main compute comparison
  and per-source consensus well defined.
- If Q3-Q2 is negative, remove local relation from the main method; do not tune
  its coefficient until the simpler Q2 mechanism is established.
