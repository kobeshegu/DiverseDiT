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

It additionally reports cross-timestep source retrieval, a class linear probe,
timestep CKA, and block CKA. TFCR archives also report the same decomposition
for Persistent and Evolving latents. This makes SiT, REPA, SRA, Self-Flow,
DiverseDiT, and TFCR comparable in a common coordinate system.

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
class linear probe. It supports SiT, REPA, DiverseDiT, and TFCR checkpoints
constructed by this repository.

## Analyze and compare

```bash
python analysis/trajectory_metrics.py analysis/features/*.npz \
  --output analysis/results/invariant_variant.json
```

For SRA and Self-Flow, add a small exporter in their official repositories that
writes the same NPZ schema. Do not force their checkpoints into this repository
if doing so changes the original architecture or training semantics.

Required keys are `features [S,T,E,L,D]`, `timesteps [T]`, `depths [L]`, and
`method`. Optional keys are `labels [S]`, `persistent [S,T,E,D]`, and
`evolving [S,T,E,D]`.
