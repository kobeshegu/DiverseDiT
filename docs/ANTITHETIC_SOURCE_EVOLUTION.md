# Antithetic Source--Evolution Factorization

## Purpose

This is the first implementation of the post-S-series main method.  It replaces
feature-consensus guesses with a flow-native, algebraically identifiable target.
Historical A3/A5/V/S commands and checkpoints are unchanged because every new
model and loss option defaults to disabled.

For an interpolant

```text
x_t = alpha(t) x0 + sigma(t) epsilon
v_t = alpha'(t) x0 + sigma'(t) epsilon,
```

the antithetic orbit uses a shared timestep and `epsilon` / `-epsilon`:

```text
x_t+ = alpha(t) x0 + sigma(t) epsilon
x_t- = alpha(t) x0 - sigma(t) epsilon.
```

The pair mean isolates the clean source and the pair difference isolates noise.
Unlike independent two-view consensus, neither target contains prediction error
or residual noise from a finite-view average.

## Main parameterization

The factorization head maps the final transformer feature to persistent and
evolving codes, so every backbone block remains on the sampled velocity path.
Two small decoders predict `x0_hat` and `epsilon_hat`; the velocity used
by both training and sampling is

```text
v_hat = alpha'(t) x0_hat + sigma'(t) epsilon_hat.
```

The standard SiT final head remains in the checkpoint but is not used as the
sampled velocity in the native configuration.  Its optional auxiliary loss is
zero by default; the zero-weight graph keeps every trainable parameter visible
to distributed training without changing the backbone objective.

The exact component losses use deterministic signal/noise energy weights:

```text
w_source = alpha(t)^2 / (alpha(t)^2 + sigma(t)^2)
w_noise  = sigma(t)^2 / (alpha(t)^2 + sigma(t)^2).
```

The evolving decoder also receives an antithetic constraint
`epsilon_hat+ + epsilon_hat- = 0`.  The source-shuffle control changes only the
`x0_hat` supervision; the flow-matching target and every sampled input remain
correct.

## New options

- `--factor-orbit-mode antithetic`
- `--factor-native-parameterization`
- `--factor-native-source-coeff`
- `--factor-native-noise-coeff`
- `--factor-native-antithetic-coeff`
- `--factor-native-base-coeff`
- `--factor-native-shuffle-source`

Native parameterization requires trajectory factorization, antithetic orbit,
factor loss frequency 1, and shared classifier-free dropout across paired views.
The provided native jobs keep the exact component objectives active after their
warmup (`factor_min_loss_scale=1.0`) rather than decaying them to zero.

## DLC screen

| Priority | Job | Steps | Question |
| --- | --- | ---: | --- |
| P0 | `52_t1_antithetic_pair_ratio1_200k.sh` | 200k | Does the identifiable orbit improve over ordinary A3 without a new output head? |
| P1 | `57_t0_native_fm_only_ratio1_200k.sh` | 200k | Architecture control: does recomposed FM alone work? |
| P2 | `55_t4_native_recomposition_ratio1_400k.sh` | 400k | Full main method: do exact source/noise factors improve generation? |
| P3 | `53_t2_native_source_ratio1_200k.sh` | 200k | Contribution of direct clean-source supervision. |
| P4 | `54_t3_native_noise_ratio1_200k.sh` | 200k | Contribution of noise and antisymmetry supervision. |
| P5 | `56_t5_native_shuffled_source_ratio1_200k.sh` | 200k | Is correct source supervision causally necessary? |

T0 and T2--T5 all sample through the native recomposition head.  T0 removes all
explicit component objectives, while T2 and T3 keep only one side.  The main
flow-matching loss always trains the recomposed output.  T1 retains the
historical standard velocity head and isolates augmentation from architecture.

Recommended decision order:

1. Compare T1 with A3 at the same checkpoint horizon.
2. Compare T0 with T1 to isolate the native output architecture.
3. Compare T4 with T0 and A5.  If its early FID trajectory is not competitive,
   stop before running all ablations to 400k.
4. Use T2/T3 to attribute a positive T4 result.
5. A valid source-stable claim requires T5 to degrade relative to T4.

## Logged diagnostics

- `factor_native_source_loss`, `factor_native_source_error`
- `factor_native_noise_loss`, `factor_native_noise_error`
- `factor_native_antithetic_loss`
- `factor_native_source_pair_gap`
- `factor_native_noise_antisymmetry_error`
- `factor_native_base_loss`, `factor_native_base_error`
- `factor_native_source_weight`, `factor_native_noise_weight`

The primary generation output is still reported through the ordinary
`denoising_loss`; it now corresponds to the recomposed native velocity.
