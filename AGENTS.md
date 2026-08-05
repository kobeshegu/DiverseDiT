# Repository Guidelines

## Project Structure & Module Organization

Core entry points live at the repository root: `train.py` and `generate.py` handle
class-conditional ImageNet experiments, while `train_t2i.py` and
`generate_t2i.py` cover text-to-image workflows. Model definitions are in
`models/`; shared data, loss, sampling, evaluation, and utility code is in
`dataset.py`, `loss.py`, `samplers*.py`, `evaluator.py`, and `utils.py`.
`preprocessing/` contains dataset conversion tools, and `dinov2/` vendors the
DINOv2 components used by representation encoders. Keep documentation images in
`assets/` and reusable experiment commands in `scripts/`. Large datasets,
checkpoints, generated samples, and experiment outputs must remain untracked.

## Setup, Training, and Evaluation

Use Python in an isolated environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run `python preprocessing/dataset_tools.py convert --help` and `encode --help`
before preparing ImageNet data. Start distributed training with
`accelerate launch train.py ...`; `scripts/exp.sh` and the README provide the
full argument set. Generate samples with `torchrun --nproc_per_node=8
generate.py ...`, package them using `python npz_convert.py ...`, and compute
metrics with `python evaluator.py REFERENCE.npz SAMPLES.npz`. Paths for data,
VAE weights, encoder checkpoints, and outputs should be command-line arguments,
not hard-coded machine-specific locations.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for
functions and variables, `PascalCase` for classes, and descriptive lowercase
module names. Keep CLI flags in kebab case (for example,
`--block-diversity-loss`) and mirror them with `snake_case` argument fields.
Group imports by standard library, third-party packages, and local modules.
There is no configured formatter or linter; keep changes focused and consistent
with adjacent code.

## Testing Guidelines

No automated test suite or coverage threshold is currently configured. At
minimum, syntax-check modified Python files with:

```bash
python -m compileall train.py generate.py models preprocessing
```

For training or sampler changes, run a small GPU smoke test with reduced batch
size, workers, steps, and sample count. Verify output shapes, checkpoint
loading, and any new CLI option in both training and generation where relevant.

## Commits & Pull Requests

Recent history uses short, imperative summaries such as `Add trajectory DINO
training branch`. Keep each commit scoped to one behavioral change. Pull
requests should explain the motivation, list exact validation commands, note
dataset/checkpoint assumptions, and link related issues. Include representative
metrics or sample images for changes that affect generation quality, and update
the README when commands or flags change.
