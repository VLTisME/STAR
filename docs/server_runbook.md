# A100 80 GB Clean Paper Runbook

This is the only supported setup for STAR paper experiments. It uses the train-derived,
video-disjoint train / dev / report protocol only. It must never receive official-test files,
historical answer files, cached rankings, leaderboard measurements, or past tuning artifacts.

## 1. Compatibility contract

Use three separate Python 3.10 environments. Do not mix their requirements.

| Environment | Purpose | Reason |
|---|---|---|
| star-pe | PE-Core, YOLO11s/RT-DETR bboxes, data tools | Modern Torch, TIMM, Transformers |
| star-caption | Qwen2.5-7B caption enhancement through vLLM | vLLM owns a modern CUDA/Torch stack |
| star-xvlm | X-VLM ITC/ITM/Smooth-AP training | Needs legacy Transformers 4.12.5 and TIMM 0.4.9 |

Do not use Python 3.12, MIM, MMCV, MMDetection, or MMPose. STAR uses YOLO11s for person boxes;
there is no OpenMMLab or ViTPose dependency in this paper pipeline.

Rent a stable Ubuntu 22.04 A100 machine:

~~~text
1 x NVIDIA A100 80 GB
16+ CPU cores
128 GB RAM preferred; 96 GB minimum
250 GB fast local NVMe minimum
Ubuntu 22.04
~~~

Before installation:

~~~bash
nvidia-smi
lscpu | sed -n '1,25p'
free -h
df -h /workspace
python --version
test -x /usr/bin/python3.10 && /usr/bin/python3.10 --version
~~~

The host CUDA toolkit does not need to equal the PyTorch wheel version. CUDA 12.1 PE wheels and
CUDA 11.8 X-VLM wheels work with a sufficiently new A100 driver. Do not create model environments
from the vendor Python 3.12 executable. If `/usr/bin/python3.10` is absent but `uv` is installed,
use its managed interpreter instead:

~~~bash
uv python install 3.10
uv venv --python 3.10 --seed /workspace/venvs/star-pe
# Repeat the same command with star-caption or star-xvlm as the final path.
~~~

If neither Python 3.10 nor `uv` is available, choose another Ubuntu 22.04 image rather than
attempting to repair the vendor Python 3.12 environment.

Set the interpreter selector once in every new shell before creating an environment:

~~~bash
if test -x /usr/bin/python3.10; then
  export STAR_PYTHON=/usr/bin/python3.10
elif command -v uv >/dev/null; then
  uv python install 3.10
  export STAR_PYTHON="$(uv python find 3.10)"
else
  echo "Python 3.10 and uv are both unavailable" >&2
  exit 1
fi
"$STAR_PYTHON" --version

make_star_venv() {
  local target="$1"
  if command -v uv >/dev/null; then
    uv venv --python "$STAR_PYTHON" --seed "$target"
  else
    "$STAR_PYTHON" -m venv --upgrade-deps "$target"
  fi
}
~~~

Install only system dependencies needed for the experiment:

~~~bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs python3.10 python3.10-venv python3-pip \
  build-essential libglib2.0-0 libgl1 libjpeg-dev ffmpeg zstd tmux rsync
git lfs install
~~~

## 2. Workspace, caches, and private credentials

~~~bash
mkdir -p /workspace/{venvs,checkpoints,hf-cache,wandb-cache,paper-data,third_party,uploads}
git clone git@github.com:VLTisME/STAR.git /workspace/STAR
cd /workspace/STAR

export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export WANDB_DIR=/workspace/wandb-cache
export WANDB_PROJECT=star-aicity-paper
export WANDB_ENTITY='YOUR_PRIVATE_ENTITY'
export WANDB_MODE=online
export WANDB_API_KEY='...'
export BOOTSTRAP='pip==24.3.1 setuptools==75.8.0 wheel==0.45.1'
~~~

Keep W&B and Hugging Face credentials only in the shell, a secret manager, or a protected shell
profile. Do not commit tokens, server paths, or personal information.

The --upgrade-deps flag below prevents the "venv has no pip" problem. If that still occurs, run
python -m ensurepip --upgrade inside the new venv. setuptools 75.8.0 is intentional: it keeps
legacy pkg_resources available for old X-VLM-era imports.

## 3. Modern PE and bbox environment

~~~bash
make_star_venv /workspace/venvs/star-pe
source /workspace/venvs/star-pe/bin/activate

python -m pip install --upgrade $BOOTSTRAP
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements-pe-a100.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_environment.py --stack pe
python scripts/verify_environment.py --stack bbox
python -m pip freeze | sort > /workspace/STAR/artifacts_pe_environment.txt
deactivate
~~~

Expected: Python 3.10, NumPy 1.26.4, CUDA available, an A100, and successful imports for TIMM,
OpenCLIP, Transformers, PyArrow, W&B, Ultralytics, and TorchVision.

The modern requirements deliberately pin NumPy to 1.26.4. Never upgrade it to NumPy 2 in these
environments; that caused the previous "_ARRAY_API not found" and "numpy.dtype size changed" ABI
errors.

## 4. Isolated Qwen caption environment

This is a one-time data-preparation environment. Let vLLM install the Torch version it supports;
do not manually pre-install a different Torch here.

~~~bash
make_star_venv /workspace/venvs/star-caption
source /workspace/venvs/star-caption/bin/activate

python -m pip install --upgrade $BOOTSTRAP
python -m pip install -r requirements-caption-a100.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_environment.py --stack caption
python -m pip freeze | sort > /workspace/STAR/artifacts_caption_environment.txt
deactivate
~~~

The first Qwen load downloads the model to HF_HOME and can take several minutes. Run a smoke test
before writing any subset manifest. RUN_ROOT is defined after extracting the local package in
Section 6.

~~~bash
source /workspace/venvs/star-caption/bin/activate

python scripts/enhance_caption.py \
  --subset "$RUN_ROOT/data/subsets_v2/train_30k_hard.jsonl" \
  --annotation-dir "$RUN_ROOT/data/annotation/train_processed" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend vllm --batch-size 32 --gpu-memory-utilization 0.85 \
  --smoke 50 --overwrite

# Inspect the examples. Only then write; the script keeps a .bak copy.
python scripts/enhance_caption.py \
  --subset "$RUN_ROOT/data/subsets_v2/train_30k_hard.jsonl" \
  --annotation-dir "$RUN_ROOT/data/annotation/train_processed" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend vllm --batch-size 32 --gpu-memory-utilization 0.85 \
  --write --overwrite --clear-unselected
deactivate
~~~

Normal captions remain equal to their original caption. Only anomaly-side enhanced fields are
generated. If vLLM cannot create its engine, check nvidia-smi for stale processes, then retry the
smoke run with --gpu-memory-utilization 0.75. Do not introduce another Torch version first.

## 5. Legacy X-VLM environment

X-VLM needs an isolated old stack. The sequence below avoids the old tokenizers build failure,
pkg_resources failure, NumPy 2 ABI mismatch, scipy.interp2d removal, and missing models module.

~~~bash
make_star_venv /workspace/venvs/star-xvlm
source /workspace/venvs/star-xvlm/bin/activate

python -m pip install --upgrade $BOOTSTRAP
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.2 torchvision==0.16.2
python -m pip install -r requirements-xvlm.txt

# Do not let pip resolve the obsolete tokenizers<0.11 declaration.
python -m pip install --no-deps transformers==4.12.5 timm==0.4.9
python -m pip install --no-deps -e .

python scripts/setup_xvlm_legacy.py \
  --xvlm-dir /workspace/third_party/X-VLM --clone-if-missing
python scripts/verify_environment.py \
  --stack xvlm --xvlm-dir /workspace/third_party/X-VLM
python -m pip freeze | sort > /workspace/STAR/artifacts_xvlm_environment.txt
deactivate
~~~

The setup script checks out X-VLM revision cb4fff15bcc30bba710a7717318e6b963e6935a4 and applies
idempotent patches:

1. relax only the obsolete tokenizers import-time requirement;
2. make unused CIDEr captioning imports optional;
3. replace removed SciPy interp2d relative-position interpolation with Torch bicubic;
4. make trusted checkpoint loading explicit for future Torch versions.

Put the trusted X-VLM checkpoint outside Git and save its checksum:

~~~bash
mkdir -p /workspace/checkpoints

# Option A: download the public X-VLM 16M initialization checkpoint directly.
# Option B: upload the same file from a verified private copy to this exact path instead.
python -m gdown --fuzzy \
  'https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view' \
  --output /workspace/checkpoints/xvlm_16m_base.th

test -s /workspace/checkpoints/xvlm_16m_base.th
test "$(stat -c%s /workspace/checkpoints/xvlm_16m_base.th)" -gt 800000000
sha256sum /workspace/checkpoints/xvlm_16m_base.th \
  | tee /workspace/checkpoints/xvlm_16m_base.th.sha256

python - <<'PY'
from pathlib import Path
import torch

p = Path('/workspace/checkpoints/xvlm_16m_base.th')
state = torch.load(p, map_location='cpu')
print('checkpoint bytes:', p.stat().st_size)
print('top-level type:', type(state).__name__)
print('checkpoint load: OK')
PY
~~~

Do this checkpoint gate immediately after the X-VLM environment verification and before any
training smoke test. The expected file is approximately 825 MiB. It is an initialization
checkpoint, not the best checkpoint from a historical STAR run.

These messages are expected:

~~~text
Position interpolate ... 13x13 to 23x23 (torch bicubic)
unexpected_keys: [... bbox_head ..., text_encoder.cls.predictions ...]
~~~

They come from loading a broader pretraining checkpoint and adapting 224px weights to 384px.
The legacy BERT gradient-checkpointing warning is already handled by STAR; visual checkpointing
remains enabled.

## 6. Create the clean 448px package locally

Run this locally from the private engineering archive. It never modifies 1024px originals. The
dev,report option is required: valid paper evaluation needs frozen dev/report gallery images in
addition to the selected 30K train anchors.

~~~bash
cd /path/to/STAR

python scripts/make_video_disjoint_splits.py \
  --annotation-dir /private/data/annotation/train_processed \
  --output-dir manifests/paper/splits --seed 2026

test -s manifests/paper/splits/video_splits.json
python - <<'PY'
import json
p = json.load(open('manifests/paper/splits/split_summary.json'))
print('split rows:', p['row_counts'])
assert p['hard_pair_leakage'] == 0
PY

python scripts/sample_clean_subsets.py \
  --annotation-dir /private/data/annotation/train_processed \
  --video-splits manifests/paper/splits/video_splits.json \
  --output-dir manifests/paper/subsets \
  --image-root /private/data/train_webp \
  --sizes 10k,30k,50k --seed 2026

for size in 10k 30k 50k; do
  test -s "manifests/paper/subsets/train_${size}_hard.jsonl"
  wc -l "manifests/paper/subsets/train_${size}_hard.jsonl"
done

python scripts/sample_clean_eval.py \
  --annotation-dir /private/data/annotation/train_processed \
  --video-splits manifests/paper/splits/video_splits.json \
  --output-dir manifests/paper/eval \
  --dev-queries 1000 --dev-gallery 5000 \
  --report-queries 2000 --report-gallery 10000 \
  --seed 2026

cat manifests/paper/eval/eval_summary.json

python scripts/prepare_subset_448.py \
  --subset manifests/paper/subsets/train_30k_hard.jsonl \
  --data-root /private/data \
  --subsets-dir manifests/paper/subsets \
  --image-root /private/data/train_webp \
  --annotation-dir /private/data/annotation/train_processed \
  --video-splits manifests/paper/splits/video_splits.json \
  --include-eval-splits dev,report \
  --eval-dir manifests/paper/eval \
  --output-root exports/paper --size 448 --quality 92 \
  --workers 16 --shard-size 2048 --archive zst
~~~

If an earlier preparation was interrupted, use `--resume` only when the export directory already
contains valid 448px files. If it is empty or stale, rerun the same command with `--overwrite`;
the archive is created only after the export directory and checksum manifest are complete.

The resulting export contains only the train subset, hard endpoints, and fixed compact dev/report
query-gallery suites (5K dev gallery; 10K report gallery). It includes checksums and 2,048-image
shards. Upload the .tar.zst archive only, then extract it on the A100:

~~~bash
mkdir -p /workspace/paper-data
tar --zstd -xf /workspace/uploads/train_30k_hard_448_data.tar.zst \
  -C /workspace/paper-data

export RUN_ROOT=/workspace/paper-data/train_30k_hard_448_data
test -d "$RUN_ROOT/data/train_webp"
test -d "$RUN_ROOT/data/annotation/train_processed"
test -f "$RUN_ROOT/data/splits/video_splits.json"
(
  cd "$RUN_ROOT"
  sha256sum -c checksums.sha256
)
~~~

## 7. Caption, bbox, and manifest preparation

Run the caption stage from Section 4 first. Then make bboxes for selected train images only.
They support online augmentation; evaluation is augmentation-free and does not need bboxes.

~~~bash
cd /workspace/STAR
source /workspace/venvs/star-pe/bin/activate

python scripts/extract_person_bboxes.py \
  --subset "$RUN_ROOT/data/subsets_v2/train_30k_hard.jsonl" \
  --image-root "$RUN_ROOT/data/train_webp" \
  --output "$RUN_ROOT/artifacts/train_30k_hard_yolo11s.json" \
  --model yolo11s.pt --device cuda:0 --imgsz 640 --batch-size 128 \
  --conf 0.20 --iou 0.70 --max-det 20 \
  --qa-dir "$RUN_ROOT/artifacts/bbox_qa" --qa-samples 500 --resume

python scripts/build_training_manifest.py \
  --subset "$RUN_ROOT/data/subsets_v2/train_30k_hard.jsonl" \
  --annotation-dir "$RUN_ROOT/data/annotation/train_processed" \
  --video-splits "$RUN_ROOT/data/splits/video_splits.json" \
  --eval-dir "$RUN_ROOT/data/eval" \
  --bbox-json "$RUN_ROOT/artifacts/train_30k_hard_yolo11s.json" \
  --output "$RUN_ROOT/manifests/train_30k_hard.parquet" \
  --use-enhanced --dedupe-eval-captions

RUN_ROOT="$RUN_ROOT" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

p = Path(os.environ["RUN_ROOT"]) / "manifests/train_30k_hard.parquet"
df = pd.read_parquet(p)
print(df.groupby("split").size())
assert set(df["split"]) == {"train", "dev", "report"}
assert (df.loc[df.split.eq("train"), "caption"].str.strip() != "").all()
print("manifest validation: OK")
PY
deactivate
~~~

## 8. Smoke checks

Do not start a scale or ablation run until a 500-1,000-row smoke run verifies:

1. manifest/image-path validation;
2. Qwen examples manually inspected;
3. YOLO overlay QA inspected;
4. PE text cache and one forward/backward step;
5. X-VLM --overfit-one-batch;
6. one completed epoch, last checkpoint, and exact resume;
7. W&B URL, TensorBoard events, JSONL, and NVML telemetry.

Define all paths once:

~~~bash
export MANIFEST="$RUN_ROOT/manifests/train_30k_hard.parquet"
export IMAGE_ROOT="$RUN_ROOT"
export XVLM_REPO=/workspace/third_party/X-VLM
export XVLM_CKPT=/workspace/checkpoints/xvlm_16m_base.th
~~~

## 9. PE 30K reference

~~~bash
cd /workspace/STAR
source /workspace/venvs/star-pe/bin/activate

python scripts/precompute_pe_text.py \
  --manifest "$MANIFEST" \
  --output "$RUN_ROOT/artifacts/pe_30k_text.pt" \
  --model hf-hub:timm/PE-Core-bigG-14-448 --device cuda --batch-size 256

python scripts/train_pe.py \
  --manifest "$MANIFEST" --image-root "$IMAGE_ROOT" \
  --text-cache "$RUN_ROOT/artifacts/pe_30k_text.pt" \
  --out-dir /workspace/STAR/outputs/paper/30k/pe \
  --auto-batch --epochs 4 --xbm-size 4096 --selection-split dev \
  --log-wandb --wandb-group 30k-main \
  --wandb-run-name pe-reference-seed2026 \
  --wandb-tags paper,30k,pe,reference

# Exact resume only from an epoch-complete last checkpoint:
python scripts/train_pe.py ... \
  --resume /workspace/STAR/outputs/paper/30k/pe/last_pe.pth
deactivate
~~~

PE chooses best_dev_pe.pth by dev R@10, breaking ties by dev R@1. A changed loss, epoch budget,
or architecture is a new run: use --init-from best_dev_pe.pth, never --resume.

## 10. X-VLM 30K reference

~~~bash
cd /workspace/STAR
source /workspace/venvs/star-xvlm/bin/activate

python scripts/calibrate_xvlm_batch.py \
  --config configs/paper/xvlm_30k.yaml \
  --output /workspace/STAR/outputs/paper/30k/xvlm/batch_calibration.json \
  --set \
    data.manifest="$MANIFEST" \
    data.image_root="$IMAGE_ROOT" \
    model.xvlm_repo="$XVLM_REPO" \
    model.checkpoint="$XVLM_CKPT"

python scripts/train.py --config configs/paper/xvlm_30k.yaml \
  --set \
    data.manifest="$MANIFEST" \
    data.image_root="$IMAGE_ROOT" \
    model.xvlm_repo="$XVLM_REPO" \
    model.checkpoint="$XVLM_CKPT"

# Exact resume only from an epoch-complete last.pth with the same recipe:
python scripts/train.py --config configs/paper/xvlm_30k.yaml \
  --set data.manifest="$MANIFEST" data.image_root="$IMAGE_ROOT" \
        model.xvlm_repo="$XVLM_REPO" model.checkpoint="$XVLM_CKPT" \
  --resume /workspace/STAR/outputs/paper/30k/xvlm/last.pth
deactivate
~~~

Pair batching uses physical_batch_size / 2 explicit hard pairs and no random fillers. For batch 64,
this is 32 pairs / 64 rows. Each pair contributes one ITM positive and two directed ITM negatives.
The logged X-VLM loss is exactly ITC + 2.0 * ITM + 0.2 * Smooth-AP; MLM is disabled.

## 11. Freeze on dev, report once, and archive

Select all settings and best_dev only on dev. Record the selection in experiments/ledger.csv, freeze
the config and checkpoint hash, and evaluate once on report.

~~~bash
cd /workspace/STAR
source /workspace/venvs/star-xvlm/bin/activate

python scripts/evaluate.py \
  --config configs/paper/xvlm_30k.yaml \
  --ckpt outputs/paper/30k/xvlm/best_dev.pth \
  --split report \
  --output outputs/paper/30k/xvlm/report_metrics.json \
  --set data.manifest="$MANIFEST" data.image_root="$IMAGE_ROOT" \
        model.xvlm_repo="$XVLM_REPO" model.checkpoint="$XVLM_CKPT"

python scripts/write_run_metadata.py \
  --config configs/paper/xvlm_30k.yaml \
  --manifest "$MANIFEST" \
  --out-dir outputs/paper/30k/xvlm \
  --checkpoint outputs/paper/30k/xvlm/best_dev.pth
deactivate
~~~

Keep private copies of resolved commands/configs, split and manifest hashes, environment freezes,
train_metrics.jsonl, TensorBoard events, W&B URL, last and best_dev checkpoints, checksums, dev
metrics, and report metrics. Do not upload multi-GB checkpoints to W&B.

### 11.1 Two-stage PE -> X-VLM ITM evaluation

The standalone X-VLM bi-encoder metric is an ablation, not the final STAR metric. PE performs
global retrieval; the X-VLM ITM head only reranks PE's frozen Top-K. These models use incompatible
legacy/modern dependency stacks, so export candidates in the PE environment and consume them in
the X-VLM environment.

First select the fusion/postprocessing configuration on DEV only:

~~~bash
# Modern PE environment: one PE encode pass, CPU candidate cache.
source /workspace/venvs/star-pe/bin/activate
python scripts/export_pe_candidates.py \
  --manifest "$MANIFEST" --image-root "$RUN_ROOT" \
  --text-cache "$RUN_ROOT/artifacts/pe_30k_text.pt" \
  --ckpt outputs/paper/30k/pe/best_dev_pe.pth \
  --split dev --topk 50 --output outputs/paper/30k/two_stage/dev_pe_candidates.pt
deactivate

# Legacy X-VLM environment: ITM only on PE candidates, then CPU-only alpha/SCA/GS sweep.
source /workspace/venvs/star-xvlm/bin/activate
python scripts/rerank_xvlm_candidates.py \
  --config configs/paper/xvlm_30k.yaml \
  --xvlm-ckpt outputs/paper/30k/xvlm/best_dev.pth \
  --pe-candidates outputs/paper/30k/two_stage/dev_pe_candidates.pt \
  --split dev --output-dir outputs/paper/30k/two_stage/dev \
  --set data.manifest="$MANIFEST" data.image_root="$RUN_ROOT" \
        model.xvlm_repo="$XVLM_REPO" model.checkpoint="$XVLM_CKPT"
deactivate
~~~

After the frozen REPORT result exists, the predeclared assignment ablation can reuse its
``itm_cache.pt``. This does not encode images or make a new selection decision: it reports the
three fixed policies at the DEV-selected fusion alpha.

~~~bash
source /workspace/venvs/star-xvlm/bin/activate
python scripts/rerank_xvlm_candidates.py \
  --config configs/paper/xvlm_30k.yaml \
  --xvlm-ckpt outputs/paper/30k/xvlm/best_dev.pth \
  --pe-candidates outputs/paper/30k/two_stage/report_pe_candidates.pt \
  --split report --frozen-config outputs/paper/30k/two_stage/dev/best_dev_config.json \
  --report-postprocess-ablation --postprocess none,greedy_sca,gale_shapley \
  --reuse-itm-cache --output-dir outputs/paper/30k/two_stage/report \
  --set data.manifest="$MANIFEST" data.image_root="$RUN_ROOT" \
        model.xvlm_repo="$XVLM_REPO" model.checkpoint="$XVLM_CKPT"
deactivate
~~~

`best_dev_config.json` is frozen once. Only then generate REPORT candidates and run exactly that
configuration. The report command must never sweep or select a new setting:

~~~bash
source /workspace/venvs/star-pe/bin/activate
python scripts/export_pe_candidates.py \
  --manifest "$MANIFEST" --image-root "$RUN_ROOT" \
  --text-cache "$RUN_ROOT/artifacts/pe_30k_text.pt" \
  --ckpt outputs/paper/30k/pe/best_dev_pe.pth \
  --split report --topk 50 --output outputs/paper/30k/two_stage/report_pe_candidates.pt
deactivate

source /workspace/venvs/star-xvlm/bin/activate
python scripts/rerank_xvlm_candidates.py \
  --config configs/paper/xvlm_30k.yaml \
  --xvlm-ckpt outputs/paper/30k/xvlm/best_dev.pth \
  --pe-candidates outputs/paper/30k/two_stage/report_pe_candidates.pt \
  --split report --frozen-config outputs/paper/30k/two_stage/dev/best_dev_config.json \
  --output-dir outputs/paper/30k/two_stage/report \
  --set data.manifest="$MANIFEST" data.image_root="$RUN_ROOT" \
        model.xvlm_repo="$XVLM_REPO" model.checkpoint="$XVLM_CKPT"
deactivate
~~~

## 12. Failure guide

| Symptom | Cause | Correct response |
|---|---|---|
| No module named pip | venv lacks ensurepip | Recreate with python3.10 -m venv --upgrade-deps; otherwise run python -m ensurepip --upgrade. |
| pkgutil.ImpImporter or missing pkg_resources | Python 3.12 / incompatible setuptools | Recreate Python 3.10 environment with pinned bootstrap packages. |
| NumPy _ARRAY_API or dtype-size error | NumPy 2 ABI mismatch | Recreate the isolated environment; requirements pin NumPy 1.26.4. |
| MMCV, MMDetection, or MMPose build fails | Unneeded OpenMMLab path | Stop. This pipeline uses YOLO11s/RT-DETR only. |
| ModuleNotFoundError: models | X-VLM source path missing | Run setup_xvlm_legacy.py and pass absolute model.xvlm_repo. |
| BertModel gradient checkpointing warning | Legacy Transformers behavior | Expected and handled; do not patch BERT. |
| Relative-position interpolation warning | 224px X-VLM weights to 384px | Expected if it says torch bicubic. |
| vLLM engine failure | stale GPU job or memory reservation | Check nvidia-smi, then retry caption smoke at 0.75 GPU utilization. |

## 13. Scale progression

After the 30K reference and ablations, freeze the dev-selected recipe. Run the identical package,
preprocessing, PE, and X-VLM procedure for 10K and 50K. Do not alter model settings, loss weights,
retrieval K, or postprocessing while producing the data-scale table. A second 50K seed is a
reproducibility run, not another tuning opportunity.
