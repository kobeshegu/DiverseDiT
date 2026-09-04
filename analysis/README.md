# Unified invariant/variant analysis

The analysis uses a controlled factorial tensor rather than assigning semantics
to either branch in advance:

```text
[source image, timestep, noise realization, block, channel]
```

For each block, `trajectory_metrics.py` applies the law of total variance:

```text
total = source-stable + timestep-varying + noise-varying
```

It additionally reports cross-timestep and cross-noise source retrieval, a
held-out-source timestep probe, within-source noise-trajectory retrieval, a
class linear probe, same-class instance retrieval, timestep CKA, and block CKA.
The same-class metric prevents a shared class embedding from being mistaken for
source-specific invariance. TFCR archives report the same
decomposition for Persistent and Evolving latents. This makes SiT, REPA, SRA, Self-Flow,
DiverseDiT, TFCR, and the teacher-free invariant-subspace method comparable in
a common coordinate system.

## Export local SiT-family checkpoints

```bash
python analysis/extract_trajectory_features.py \
  --ckpt results/tfcr/checkpoints/0450000.pt \
  --data-dir /path/to/imagenet_latents \
  --method TFCR \
  --num-samples 512 --num-noises 4 \
  --output analysis/features/tfcr.npz
```

The exporter reuses each sampled noise realization across timesteps, so the
timestep and noise axes are controlled independently. By default it samples 64
classes with repeated sources per class, avoiding a meaningless one-example-per-
class linear probe. It supports SiT, REPA, DiverseDiT, TFCR, and trajectory-
invariance checkpoints constructed by this repository.
For a linear invariant checkpoint, the exporter uses orthonormal coordinates of
the learned row space and computes `variant = h - P_invariant h` from the exact
post-skip feature consumed during training. For the nonlinear MLP ablation it
exports only the readout, because an MLP has no well-defined linear complement.
For linear checkpoints, `subspace_energy_capture` reports how much total,
source, timestep, and noise variance lies in that subspace. In
`subspace_contrast`, positive source-retrieval differences and negative
timestep/noise-leakage differences are the desired directions.

## Analyze and compare

```bash
python analysis/trajectory_metrics.py analysis/features/*.npz \
  --output analysis/results/invariant_variant.json
```

For SRA and Self-Flow, add a small exporter in their official repositories that
writes the same NPZ schema. Do not force their checkpoints into this repository
if doing so changes the original architecture or training semantics.

Required keys are `features [S,T,E,L,D]`, `timesteps [T]`, `depths [L]`, and
`method`. Optional keys are `labels [S]`, `persistent [S,T,E,D]`,
`evolving [S,T,E,D]`, `invariant [S,T,E,D_inv]`, and
`variant [S,T,E,D_hidden]`. For the linear projector, `variant` is the
orthogonal complement of its learned row space at the configured source block.
