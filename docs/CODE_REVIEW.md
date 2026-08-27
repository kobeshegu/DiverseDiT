# TFCR branch code review

Scope: the TFCR implementation plus the original training/sampling paths it directly touches. Review dimensions: method correctness, compute fairness, DDP/gradient accumulation, mixed precision, checkpoint compatibility, inference overhead, and experiment reproducibility.

## Fixed findings

| Severity | Finding | Resolution |
|---|---|---|
| P0 | TFCR auxiliary projectors were executed on every sampling step, contradicting the zero-inference-overhead claim. | `SiT.forward` now defaults to `return_factorization=False`; training opts in explicitly. Sampling follows only the original denoising path. |
| P0 | Checkpoint saving used `model.module`, which fails in single-process Accelerate runs. | Save through `accelerator.unwrap_model(model)`. |
| P0 | `grad_norm` could be undefined during gradient accumulation, and step-level checkpoint/log/sample blocks could repeat on non-sync microsteps. | Initialize the value and skip step-level work until `accelerator.sync_gradients`. |
| P0 | `train_t2i.py` unpacked `SILoss` as a tuple although the loss returns a dictionary; its fixed sample batch also unpacked three dataset items while the loop expects four. | Added tuple-model-output compatibility inside `SILoss`; updated T2I loss access and fixed sample-batch unpacking. |
| P0 | A nonlinear joint recomposer could ignore Persistent and route the full target through Evolving. | Replaced it with additive branch decoders. Persistent predicts the pair-symmetric common target, Evolving predicts the signed view residual, and swapped components must jointly reconstruct the target. |
| P1 | The signed-delta transition loss was also applied to independent-noise pairs, although they are not two points on one trajectory. | Transition supervision is now masked to same-noise pairs; cross-noise pairs still train invariance/recomposition. |
| P1 | TFCR originally applied two views to every source, imposing a fixed ~2x backbone cost. | The main batch receives one FM view; a configurable subset receives a second view. Default ratio 0.5 gives roughly 1.5x cost. |
| P1 | A paired-view gain could not be separated cleanly from the auxiliary objective. | Added `--factor-paired-view-only` and A3 with identical sampler, batch ratio, frequency, warmup, and decay. |
| P1 | A3/A4 run names encoded the requested noise-pairing ratio without passing it to training. | All paired ablations now forward `CROSS_NOISE_PROB`, so same/mixed/cross-noise sweeps match their recorded names. |
| P1 | Auxiliary losses had no warmup/termination schedule, risking early optimization shock and late representation constraint. | Added linear warmup and cosine decay, following the feature branch's training practice. |
| P1 | REPA projection loss weighted paired sources twice relative to unpaired sources. | Paired per-view projection errors are averaged back to one per-source contribution. |
| P1 | The public block-diversity coefficient argument was ignored in favor of a hard-coded dynamic rule. | Training now uses `--block-diversity-loss-coeff` directly. |
| P1 | Sampling with a DiverseDiT checkpoint retained every block activation even though the diversity loss is training-only. | Block features are now collected only in training mode, removing the avoidable sampling-memory cost. |
| P1 | `--report-to none` was passed to Accelerate as if it were a tracker and still reached tracker setup. | Reporting now maps to `log_with=None`, and tracker initialization is skipped when disabled. |
| P1 | Process-local EMA copies were created before DDP synchronized the online model. | On a fresh run, EMA is synchronized again after `accelerator.prepare`. |
| P2 | No-REPA sampling required a separate boolean and did not accept the feature branch's `--projector-embed-dims none` convention. | Both `none` and `--no-projection` are accepted. |
| P2 | Loading full checkpoints may fail under newer PyTorch `weights_only` defaults. | Full training checkpoints explicitly use `weights_only=False`. |
| P2 | Raw images were transferred to GPU even when no external representation encoder was active. | Transfer and preprocessing are now conditional on REPA usage. |
| P2 | The T2I path always prefetched a qualitative batch and sampled during training, even in headless runs. | Added `--skip-training-samples`, guarded prefetch/sampling, and validated global-batch divisibility. |

## Method-level review

### Correct properties

- View ordering is explicit: `[paired-a, paired-b, unpaired]`; labels and REPA targets are reordered identically.
- FM loss gives each source one contribution: two paired-view errors are averaged before the batch mean.
- Persistent codes are swapped while Evolving codes remain current-view-specific.
- Separate decoders expose each branch contribution; common/residual targets directly train both branches.
- Deeper targets are stopped-gradient and normalized in FP32.
- The additive branch decoders have no direct timestep input.
- Signed delta-t is used in both transition directions.
- same-noise, cross-noise, and mixed pairing are supported.
- Auxiliary heads are checkpointed/EMA-updated but absent from the inference computation graph.

### Diagnostics already logged

- Persistent and Evolving cross-view similarity.
- Correct-vs-wrong Evolving recomposition gap.
- Zero-Persistent and zero-Evolving usage gaps.
- Common/residual target energy fractions.
- Per-branch feature standard deviation.
- Decorrelation and optional variance losses.
- Mean delta-t, actual cross-noise fraction, and actual paired batch fraction.
- Warmup/decay scale and whether the factorization branch was active.

## Remaining limitations and required validation

1. **Balanced recomposition is operational, not statistically identifiable.** The simplest Evolving-only path is now directly constrained, but invertible transforms within each latent and pair-distribution effects remain. Treat semantic interpretations as hypotheses until controlled variance decomposition, zero/replacement interventions, capacity sweeps, and probes pass.
2. **GPU runtime tests remain mandatory.** The local desktop runtime does not contain PyTorch/timm, so only syntax, HTML, and diff checks can run here. Execute `pytest tests/test_trajectory_factorization.py` in the training environment before launching expensive jobs.
3. **TFCR is currently integrated into class-conditional SiT, not MMDiT.** The existing T2I path is repaired for its baseline loss interface, but extending Persistent/Evolving factorization to MMDiT should wait until the ImageNet hypothesis passes the Go/No-Go screen.
4. **Core representation measurements are automated; interventions are not.** `analysis/` now exports controlled local checkpoints and computes source/time/noise variance, CKA, retrieval, and a class probe. Gradient maps, feature transplants, counterfactual swaps, and official SRA/Self-Flow exporters remain to be implemented in their native codebases.
5. **SRA and Self-Flow are comparison dependencies, not local implementations.** Main tables should use official code/checkpoints or a separately reviewed faithful port.
6. **Legacy block diversity code warrants an independent audit.** TFCR does not depend on it; use A7 only after reproducing the original DiverseDiT baseline with the corrected public coefficient.
7. **Compute accounting must be measured, not inferred only from view count.** Record actual images/sec, peak memory, and profiler FLOPs for A0/A3/A4/A5.
8. **The balanced head intentionally changes experimental-checkpoint format.** New checkpoints carry `tfcr_objective=balanced_additive_v1`. Any checkpoint trained at the earlier joint-recomposer commit must be evaluated with that historical code; silently mapping its weights would change the learned objective.

## Pre-flight checklist

```bash
pytest -q tests/test_trajectory_factorization.py tests/test_trajectory_metrics.py
python -m py_compile models/sit.py loss.py train.py generate.py train_t2i.py analysis/*.py
bash -n scripts/tfcr_ablation.sh
bash -n scripts/train_tfcr.sh
```

Then run a 100-step smoke job with `SiT-B/2`, local batch >= 2, `factor_batch_ratio=0.5`, and `report_to=none`. Confirm all logged values are finite, both projectors/decoders receive gradients, and both usage gaps become positive.
