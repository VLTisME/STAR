# STAR: Action-Aware Two-Stage Retrieval for Fine-Grained Person Anomaly Retrieval

STAR is a reproducible research codebase for fine-grained text-to-image retrieval where
the decisive evidence is an action, interaction, or event state rather than the global
scene. It combines action-focused caption enhancement, a PE-Core retrieval stage with
Cross-Batch Memory (XBM), an X-VLM image-text matching reranker, and one-to-one
Gale-Shapley assignment.

> Status: private research release. Paper results are reported only from the train-derived,
> video-disjoint protocol described in [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Highlights

- **Action-focused supervision:** anomaly captions are enhanced with their source event
  description while normal captions remain unchanged.
- **High-recall retrieval:** PE-Core-bigG-14-448 is adapted with visual LoRA and a
  FIFO Cross-Batch Memory.
- **Fine-grained verification:** X-VLM reranks the fixed retrieval candidate set with
  ITC, ITM, and Smooth-AP objectives.
- **Consistent answers:** Gale-Shapley resolves duplicate independent Top-1 predictions.
- **Reproducible experiments:** deterministic split manifests, configuration snapshots,
  JSONL, TensorBoard, W&B, checkpoint checksums, and an experiment ledger.

## Repository Layout

```text
STAR/
  paper/          ECCVW manuscript source
  configs/paper/  frozen experiment recipes
  src/star/       training package
  scripts/        data, training, evaluation, and figure utilities
  docs/           protocol and server instructions
  experiments/    run ledger and generated tables
  tests/          math and data-contract tests
```

## Clean Evaluation Protocol

PAB training videos are deterministically partitioned into `train`, `dev`, and `report`
components. The split keeps all hard-pair endpoints in one partition. `dev` is the only
split permitted to select a hyperparameter, checkpoint, fusion weight, or postprocessor.
`report` is evaluation-only and supplies the paper tables.

```text
train  80%   training subsets: 10K ⊂ 30K ⊂ 50K
dev    10%   configuration and checkpoint selection
report 10%   final paper metrics
```

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the full protocol.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/make_video_disjoint_splits.py \
  --annotation-dir /path/to/train_processed \
  --output-dir manifests/splits
```

The real X-VLM reranker uses a separate pinned environment because the original model
depends on an older Transformers/TIMM stack. See [docs/server_runbook.md](docs/server_runbook.md).

## Tracking

Set credentials outside the repository:

```bash
export WANDB_PROJECT=star-aicity-paper
export WANDB_ENTITY=your-wandb-entity
export WANDB_MODE=online
```

Each run writes JSONL metrics, TensorBoard events, `last` and `best_dev` checkpoints,
configuration snapshots, hashes, and a W&B URL. Checkpoint files are retained on the
experiment server and referenced in `experiments/ledger.csv` by SHA-256.

## Paper and Citation

The working manuscript outline is in [PAPER.md](PAPER.md). Citation metadata will be added
after paper acceptance and the public release license audit.

## License and Data

This repository does not include PAB images, annotations, pretrained weights, cached predictions,
or any held-out competition evaluation artifacts. See [LICENSE.md](LICENSE.md) and
[NOTICE.md](NOTICE.md).
