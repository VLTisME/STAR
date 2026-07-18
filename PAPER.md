# STAR: Action-Aware Two-Stage Retrieval for Fine-Grained Person Anomaly Retrieval

**Authors (camera-ready):** Tuan Vo-Lan, Khanh Tran-Quoc, Phuc Nguyen-Ngoc, Thanh Vo-Thai  
**Affiliation:** University of Science, Vietnam National University Ho Chi Minh City

> This is the living scientific draft and evidence tracker. Values marked `[TBD]` must be
> filled only from the clean video-disjoint protocol in `docs/experiment_protocol.md`.
> Historical evaluation artifacts and tuning runs are not paper evidence.

## Abstract

Fine-grained person anomaly retrieval requires matching a textual description to an image
whose decisive evidence may be an action, interaction, or event state rather than its global
scene. This is difficult when candidate images share people, objects, and background while
differing only in a small action transition. We present STAR, an action-aware two-stage
retrieval framework for this setting. STAR enriches anomaly-side training captions with
source event information, adapts a PE-Core visual retriever with Cross-Batch Memory to obtain
a high-recall candidate set, and applies an X-VLM cross-encoder trained with contrastive,
image-text matching, and Smooth-AP losses for fine-grained reranking. A Gale-Shapley assignment
stage resolves conflicting Top-1 image choices under the one-to-one structure of retrieval.
On a video-disjoint held-out report split, STAR achieves `[TBD]` mAP and `[TBD]` R@1, while
ablations quantify the contribution of caption enhancement, XBM, rank-aware learning, and
global assignment. `[TBD: add one OOD sentence only if UCF-Crime distractor experiments run.]`

## 1. Introduction

### Problem

Text-based person anomaly retrieval asks a system to retrieve the image corresponding to a
natural-language description of an event. In many hard cases, candidates share a scene,
subjects, attire, and objects; the correct decision depends on action phase, body state,
object contact, affected subject, or event outcome. A global contrastive embedding can retrieve
the right neighborhood but still confuse near-identical frames.

### Motivation

Synthetic captions frequently describe visible surroundings while omitting the anomaly or
action label that distinguishes the correct image. This creates supervision that is visually
rich but event-poor. Separately, independent query-wise Top-1 predictions can conflict when
multiple descriptions select the same gallery image despite the benchmark's one-to-one target
structure.

### Contributions

1. **Action-aware two-stage retrieval.** A PE-Core retrieval stage produces a high-recall
   candidate set; an X-VLM cross-encoder reranks it with action-sensitive image-text matching.
   The reranker jointly optimizes ITC, ITM, and Smooth-AP.
2. **Action-focused caption enhancement.** We augment anomaly-side synthetic captions with
   source event descriptions while preserving normal captions, improving the alignment between
   training language and the action-centric retrieval target.
3. **Cross-Batch Memory for retrieval adaptation.** PE-Core visual adaptation uses detached
   FIFO image/text queues to expose more informative contrastive negatives than one batch alone.
4. **One-to-one final assignment.** Gale-Shapley matching resolves duplicate Top-1 image
   conflicts while retaining score-ordered ranks 2--10.

## 2. Related Work

### 2.1 Text-Image Retrieval

Discuss CLIP-style dual encoders, contrastive objectives, large-scale pretrained vision-language
models, and the efficiency/precision trade-off motivating a retriever plus cross-encoder.

### 2.2 Fine-Grained Retrieval and Hard Negatives

Discuss hard-negative mining and pair-aware sampling. Position STAR as using hard-pair structure
as training support, rather than claiming it as a new contribution.

### 2.3 Rank-Aware Objectives

Discuss differentiable ranking and Smooth-AP. Clarify that with a single correct item per query,
average precision equals reciprocal rank, so improving early ranks directly matters.

### 2.4 Caption Augmentation

Discuss synthetic caption limitations and task-aware textual enrichment. State that enhancement
is restricted to anomaly-side captions to avoid inventing changes for normal examples.

### 2.5 Global Assignment

Discuss SCA and stable matching for one-to-one retrieval. Compare Greedy SCA, unmodified scores,
and Gale-Shapley experimentally.

## 3. Method

### 3.1 Data Preparation and Caption Enhancement

Clean labels, construct a video-disjoint split, and generate nested 10K/30K/50K hard subsets
from the training partition. For an anomaly row, generate `caption_enhanced` and
`hard_c_enhanced` from its original caption, anomaly label, and source event description. For
normal rows, the enhanced field is exactly the original caption. The original annotation fields
remain unchanged.

### 3.2 Overall Architecture

```text
processed PAB rows
  -> action-focused caption enhancement
  -> PE-Core visual retrieval + XBM
  -> Top-K candidate set
  -> X-VLM cross-encoder ITM reranking
  -> score-ordered candidate lists
  -> Gale-Shapley Top-1 assignment
  -> final Top-10 rankings
```

PE-Core-bigG-14-448 is the first-stage retriever. Its text representation is cached and frozen;
visual LoRA and alignment layers are optimized. X-VLM consumes the Top-K candidates at 384px and
uses its cross-encoder to verify fine-grained image-text compatibility.

### 3.3 Learning Objective

The X-VLM objective is:

```math
L = L_ITC + lambda_ITM L_ITM + lambda_SAP L_SmoothAP.
```

`L_ITC` aligns global image and text features, `L_ITM` distinguishes positive pairs from explicit
hard image/text mismatches, and `L_SmoothAP` supplies a differentiable rank-aware signal.
The selected values of `lambda_ITM` and `lambda_SAP` will be chosen only on `dev`.

### 3.4 Cross-Batch Memory

The PE stage stores normalized detached image/text embeddings with instance and caption hashes
in a FIFO queue. Queue entries enlarge the negative set while duplicate captions and same-instance
items are masked. `[TBD: queue size and final PE adaptation schedule.]`

### 3.5 Gale-Shapley Assignment

Queries propose to candidate images in descending reranker score order. Each image tentatively
holds its preferred proposal and rejects inferior ones until convergence. The accepted image is
placed at rank one; ranks two through ten preserve their score ordering after removing duplicates.

### 3.6 Optional Multi-Retriever Fusion

RRF fusion remains optional. It will appear in the final method only if selected on `dev` and
reproduced on `report`; otherwise it is an appendix ablation.

## 4. Experiments

### 4.1 Dataset, Split, and Metrics

Use PAB training data only. Split video-connected components into train/dev/report at 80/10/10.
All selection is performed on dev, while report is used exactly once for a frozen configuration.
Report mAP, R@1, R@5, and R@10.

### 4.2 Experimental Settings

Describe 448px PE retrieval, 384px X-VLM reranking, LoRA, XBM, augmentation, batch size,
optimizer, schedule, seeds, compute, and checkpoint selection. All values come from tracked
config files and the experiment ledger.

### 4.3 Planned Main Tables

| Table | Comparison | Evaluation split |
|---|---|---|
| Main system | retriever / reranker / assignment variants | report |
| Data scale | 10K, 30K, 50K full pipeline | report |
| Component ablation | caption enhancement, XBM, Smooth-AP, ITM weight | report |
| Postprocessing | none, Greedy SCA, Gale-Shapley, locked GS | report |
| Optional OOD | report with fixed UCF-Crime distractors | report + OOD gallery |

### 4.4 Qualitative Analysis

Show action-focused caption examples, retrieval wins over the base system, and remaining failures.
Use train/report images only. Do not include held-out competition examples or tuning-selected cases.

## 5. Limitations and Reproducibility

Caption enhancement may preserve or amplify mistakes when the source event description is
ambiguous. Two-stage retrieval adds latency relative to a single dual encoder. The public release
provides code and configuration but not PAB data, pretrained weights, or official evaluation
artifacts. Every paper number must have a matching ledger row, configuration hash, split hash,
and checkpoint checksum.

## Evidence Tracker

| Claim | Required evidence | Status |
|---|---|---|
| Caption enhancement helps | 30K no-enhancement vs enhancement on report | pending |
| XBM helps PE retrieval | 30K PE no-XBM vs XBM on report | pending |
| Smooth-AP helps reranking | 30K Smooth-AP off/on on report | pending |
| Gale-Shapley helps assignment | same score matrix, four postprocessors | pending |
| More data helps | 10K/30K/50K frozen recipe on report | pending |
