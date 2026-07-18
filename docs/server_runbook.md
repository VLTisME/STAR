# A100 Paper Runbook

This runbook is the only supported path for clean paper experiments. It never consumes held-out
competition files, historical tuning artifacts, leaderboard scores, or cached predictions.

## Machine and environment

Rent one stable machine for the complete cycle:

```text
1 x NVIDIA A100 80 GB
16+ CPU cores
128 GB RAM preferred (96 GB minimum)
250 GB fast local NVMe minimum
Ubuntu 22.04 or 24.04
```

Use separate virtual environments because X-VLM and PE have different dependency constraints.
Keep the A100 workspace on local NVMe, not a network mount.

```bash
git clone git@github.com:VLTisME/STAR.git /workspace/STAR
cd /workspace/STAR

# Store this only in the shell/session manager, never in Git or a notebook.
export WANDB_API_KEY='...'
export WANDB_ENTITY='YOUR_PRIVATE_ENTITY'
export WANDB_PROJECT='star-aicity-paper'
```

Create `pe-venv` with a recent CUDA PyTorch/timm stack and `xvlm-venv` with the pinned
requirements in `requirements-xvlm.txt`. Install W&B, TensorBoard, `nvidia-ml-py`, and the
corresponding requirements into both environments. Keep the external X-VLM source checkout
outside this repository and set `model.xvlm_repo` in each config.

## Immutable data preparation

The data directory is private and ignored. Start from processed annotations and full-resolution
WebP originals. Do not modify originals. Resize selected assets to a separate 448px export.

```bash
python scripts/make_video_disjoint_splits.py \
  --annotation-dir data/annotation/train_processed \
  --output-dir manifests/paper/splits \
  --seed 2026

python scripts/sample_clean_subsets.py \
  --annotation-dir data/annotation/train_processed \
  --video-splits manifests/paper/splits/video_splits.json \
  --output-dir manifests/paper/subsets \
  --image-root data/train_webp \
  --sizes 10k,30k,50k \
  --seed 2026
```

For each selected subset, use the private resize/export helper to create a non-destructive 448px
asset root. Run Qwen caption enhancement only on the train subset: anomaly fields receive natural
action-aware text; normal fields remain byte-for-byte equal to their original caption. Run YOLO11s
on the selected train assets plus dev/report gallery assets and inspect a 500-image overlay sample.

Build a manifest only after all required assets exist:

```bash
python scripts/build_training_manifest.py \
  --subset manifests/paper/subsets/train_30k_hard.jsonl \
  --annotation-dir data/annotation/train_processed \
  --video-splits manifests/paper/splits/video_splits.json \
  --bbox-json artifacts/bbox_all_required.json \
  --output manifests/paper/train_30k_hard.parquet
```

The manifest includes train/dev/report. Training scripts read only `train` and `dev`; `report`
is used later by `scripts/evaluate.py` after the recipe is frozen.

## Smoke run

Before the 30K reference, create a 500--1,000 row training-only smoke subset and run:

1. manifest validation;
2. PE text embedding cache;
3. PE forward/backward with auto-batch calibration;
4. X-VLM `--overfit-one-batch`;
5. one epoch checkpoint/resume test;
6. W&B, TensorBoard, JSONL, and NVML telemetry check.

No scale or ablation run begins until all six checks pass.

## PE run

Use PE config values from `configs/paper/pe_30k.yaml`; first calibrate on the actual manifest.
The selected physical batch must retain at least 6 GiB headroom and target 68--74 GiB peak VRAM.

```bash
python scripts/precompute_pe_text.py \
  --manifest manifests/paper/train_30k_hard.parquet \
  --output artifacts/pe_30k_text.pt \
  --model hf-hub:timm/PE-Core-bigG-14-448 \
  --device cuda

python scripts/train_pe.py \
  --manifest manifests/paper/train_30k_hard.parquet \
  --image-root . \
  --text-cache artifacts/pe_30k_text.pt \
  --out-dir outputs/paper/30k/pe \
  --auto-batch --epochs 4 --xbm-size 4096 \
  --selection-split dev --log-wandb \
  --wandb-group 30k-main --wandb-run-name pe-reference-seed2026 \
  --wandb-tags paper,30k,pe,reference
```

The PE script writes `last_pe.pth` at every completed epoch and chooses `best_dev_pe.pth` by
dev R@10, breaking ties with dev R@1. Exact resume uses only `last_pe.pth`:

```bash
python scripts/train_pe.py ... --resume outputs/paper/30k/pe/last_pe.pth
```

Use `--init-from best_dev_pe.pth` for a changed recipe or a new fine-tuning phase; it intentionally
resets optimizer, scheduler, XBM, and step count.

## X-VLM run

Calibrate the physical batch before the full reference. `pair` means every physical batch consists
of `batch_size / 2` explicit hard pairs and no random fillers.

```bash
python scripts/calibrate_xvlm_batch.py \
  --config configs/paper/xvlm_30k.yaml \
  --output outputs/paper/30k/xvlm/batch_calibration.json

python scripts/train.py --config configs/paper/xvlm_30k.yaml
```

The trainer writes `last.pth` at completed epochs and selects `best_dev.pth` using dev mAP with a
dev R@10 safety floor. Resume exactly only from `last.pth` with the same recipe:

```bash
python scripts/train.py --config configs/paper/xvlm_30k.yaml \
  --resume outputs/paper/30k/xvlm/last.pth
```

For a changed loss weight, new epoch budget, or new augmentation recipe, start a new run with
`--init-from path/to/best_dev.pth`; this is a warm start, not an exact resume.

## Frozen reporting and ledger

After choosing a recipe on `dev`, write its choice to `experiments/ledger.csv`, lock the config,
and evaluate once on report:

```bash
python scripts/evaluate.py \
  --config configs/paper/xvlm_30k.yaml \
  --ckpt outputs/paper/30k/xvlm/best_dev.pth \
  --split report \
  --output outputs/paper/30k/xvlm/report_metrics.json
```

Each completed run directory must keep its resolved config, split/manifest fingerprints, JSONL,
TensorBoard events, W&B URL, `last` and `best_dev` checkpoints, and SHA-256 checksums. Do not upload
multi-GB checkpoints to W&B; log their checksums instead.
