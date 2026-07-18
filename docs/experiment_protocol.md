# STAR Clean Experiment Protocol

## Objective

This protocol produces paper evidence without using held-out competition data, leaderboard
scores, or historical tuning artifacts for model selection. It applies to every run tracked in
`experiments/ledger.csv`.

## Split Construction

Run `scripts/make_video_disjoint_splits.py` on `train_processed`. The script constructs connected
components over `video_id` values: if an annotation points to `hard_i_id` from another video, both
videos are joined. Components, not individual rows, are assigned deterministically to:

```text
train:  80 percent
dev:    10 percent
report: 10 percent
```

The result records the seed, input fingerprint, component counts, row counts, image counts, and
hard-edge leakage checks. A training pair is invalid if either endpoint is outside `train`.

## Selection Rules

- Select learning rates, loss weights, Top-K, fusion settings, assignment policy, early stopping,
  and checkpoint using `dev` only.
- Run `report` only after a recipe is frozen. Do not choose a new setting after viewing report.
- Record every evaluated recipe, including failed runs, in the experiment ledger.
- Use a fixed seed of `2026` for the main scale table. Repeat only the final chosen 50K recipe
  with seed `2027` when budget permits.

## Required Experiment Matrix

1. 30K clean reference: enhanced captions, PE XBM, X-VLM ITM plus Smooth-AP, Gale-Shapley.
2. 30K caption ablation: original versus enhanced captions.
3. 30K PE ablation: XBM disabled versus enabled.
4. 30K reranker ablations: Smooth-AP disabled/enabled and `lambda_ITM` 1.0/2.0.
5. 30K postprocessing: none, Greedy SCA, Gale-Shapley, and locked Gale-Shapley on identical scores.
6. Frozen full-pipeline data scale: 10K, 30K, 50K.

All paper tables use `report` metrics: mAP, R@1, R@5, R@10. The optional UCF-Crime distractor
study is a separate fixed-gallery stress test; it is not required for the core paper.

## Run Artifacts

Each run directory must contain:

```text
config.resolved.yaml
data_fingerprint.json
train_metrics.jsonl
tensorboard/
last checkpoint
best_dev checkpoint
checkpoint SHA-256 manifest
wandb_url.txt
```

W&B and TensorBoard are monitoring systems; JSONL plus config and checksum files remain the
source of record for paper tables.
