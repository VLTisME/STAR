#!/usr/bin/env python3
"""Sample nested, hard-pair-safe paper training subsets from the train split only.

This intentionally has no ``dev`` or ``report`` sampling mode.  Those partitions
are immutable products of ``make_video_disjoint_splits.py`` and are never used to
choose training examples or tune subset composition.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEED = 2026
BAD_LABEL = "__bad_label__"
SIZES = (10_000, 30_000, 50_000)
BUCKETS = ("goal", "full", "wentwrong")
BUCKET_RATIOS = {"goal": 0.37109, "full": 0.34398, "wentwrong": 0.28493}
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Row:
    raw: dict
    image_id: str
    hard_i_id: str
    video_id: int
    bucket: str
    label_type: str
    label: str
    caption_key: str
    hard_caption_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/train_webp"))
    parser.add_argument("--sizes", default="10k,30k,50k")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalise_text(value: object) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip().lower())


def iter_rows(annotation_dir: Path) -> Iterable[dict]:
    files = sorted(annotation_dir.glob("attr_*.json"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No attr_*.json files in {annotation_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def bucket_from_path(value: str) -> str:
    for bucket in BUCKETS:
        if f"/{bucket}/" in f"/{value.replace(chr(92), '/')}":
            return bucket
    raise ValueError(f"Cannot infer bucket from image path: {value}")


def label_info(row: dict) -> tuple[str, str]:
    if row.get("anomaly") is not None:
        return "anomaly", str(row["anomaly"])
    if row.get("normal") is not None:
        return "normal", str(row["normal"])
    raise ValueError(f"Missing normal/anomaly label for {row.get('image_id')}")


def to_webp_path(value: str) -> str:
    value = value.replace("\\", "/").removeprefix("train/")
    return (Path("data/train_webp") / Path(value).with_suffix(".webp")).as_posix()


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        text = part.strip().lower().removesuffix("k")
        sizes.append(int(text) * 1000)
    if not sizes or any(size <= 0 for size in sizes) or sizes != sorted(set(sizes)):
        raise ValueError("--sizes must be increasing unique positive values, e.g. 10k,30k,50k")
    return sizes


def quotas(size: int) -> dict[str, int]:
    raw = {bucket: size * BUCKET_RATIOS[bucket] for bucket in BUCKETS}
    result = {bucket: int(raw[bucket]) for bucket in BUCKETS}
    for bucket in sorted(BUCKETS, key=lambda name: raw[name] - result[name], reverse=True)[: size - sum(result.values())]:
        result[bucket] += 1
    return result


def hard_score(anchor: Row, target: Row) -> float:
    score = 0.0
    pair = {anchor.bucket, target.bucket}
    if pair in ({"goal", "full"}, {"goal", "wentwrong"}):
        score += 5.0
    if anchor.video_id == target.video_id:
        score += 2.0
    if anchor.raw.get("action") != target.raw.get("action"):
        score += 2.0
    if anchor.raw.get("scene") and anchor.raw.get("scene") == target.raw.get("scene"):
        score += 1.5
    return score


def candidate_order(candidates: list[Row], rows_by_id: dict[str, Row], seed: int) -> list[Row]:
    rng = random.Random(seed)
    label_counts = Counter(row.label for row in candidates)
    keyed = []
    for row in candidates:
        # Hardness is primary; a light rare-label boost prevents the common labels from
        # entirely consuming every bucket while keeping the distribution recognisable.
        rarity = label_counts[row.label] ** -0.25
        keyed.append((-hard_score(row, rows_by_id[row.hard_i_id]), -rarity, rng.random(), row.image_id, row))
    keyed.sort()
    return [item[-1] for item in keyed]


def choose_nested(
    ordered_by_bucket: dict[str, list[Row]], sizes: list[int]
) -> dict[int, list[Row]]:
    selected: list[Row] = []
    selected_ids: set[str] = set()
    used_text: set[str] = set()
    cursors = {bucket: 0 for bucket in BUCKETS}
    outputs: dict[int, list[Row]] = {}

    for size in sizes:
        required = quotas(size)
        existing = Counter(row.bucket for row in selected)
        for bucket in BUCKETS:
            need = required[bucket] - existing[bucket]
            while need > 0:
                options = ordered_by_bucket[bucket]
                if cursors[bucket] >= len(options):
                    raise RuntimeError(
                        f"Insufficient unique-caption {bucket} candidates while building {size:,}: "
                        f"need {need:,} more"
                    )
                row = options[cursors[bucket]]
                cursors[bucket] += 1
                texts = {row.caption_key, row.hard_caption_key}
                if not row.caption_key or not row.hard_caption_key or len(texts) != 2:
                    continue
                if row.image_id in selected_ids or texts & used_text:
                    continue
                selected.append(row)
                selected_ids.add(row.image_id)
                used_text.update(texts)
                need -= 1
        if len(selected) != size:
            raise AssertionError(f"nested subset has {len(selected):,}, expected {size:,}")
        outputs[size] = list(selected)
    return outputs


def edge_relation(anchor: Row, target: Row) -> list[str]:
    return [
        "goal_full" if {anchor.bucket, target.bucket} == {"goal", "full"}
        else "goal_wentwrong" if {anchor.bucket, target.bucket} == {"goal", "wentwrong"}
        else f"{anchor.bucket}_{target.bucket}",
        "same_video" if anchor.video_id == target.video_id else "different_video",
        "different_action" if anchor.raw.get("action") != target.raw.get("action") else "same_action",
        "same_scene" if anchor.raw.get("scene") and anchor.raw.get("scene") == target.raw.get("scene") else "different_scene",
    ]


def write_subset(
    selected: list[Row],
    name: str,
    output_dir: Path,
    image_root: Path,
    check_images: bool,
) -> dict:
    selected_by_id = {row.image_id: row for row in selected}
    incident = Counter()
    edges = []
    missing = Counter()
    for row in selected:
        target = selected_by_id.get(row.hard_i_id)
        if target is None:
            continue
        edges.append(
            {
                "edge_id": f"{row.image_id}->{target.image_id}",
                "anchor_image_id": row.image_id,
                "hard_image_id": target.image_id,
                "anchor_bucket": row.bucket,
                "hard_bucket": target.bucket,
                "video_id": row.video_id,
                "relation": edge_relation(row, target),
            }
        )
        incident[row.image_id] += 1
        incident[target.image_id] += 1

    subset_path = output_dir / f"{name}.jsonl"
    with subset_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            output = dict(row.raw)
            image_webp = to_webp_path(str(row.raw["image"]))
            hard_image_webp = to_webp_path(str(row.raw["hard_i"]))
            output.update(
                {
                    "subset_name": name,
                    "bucket": row.bucket,
                    "label_type": row.label_type,
                    "label": row.label,
                    "image_webp": image_webp,
                    "hard_image_webp": hard_image_webp,
                    "hard_edge_count": int(incident[row.image_id]),
                }
            )
            if check_images:
                if not (image_root / Path(image_webp).relative_to("data/train_webp")).exists():
                    missing["image_webp"] += 1
                if not (image_root / Path(hard_image_webp).relative_to("data/train_webp")).exists():
                    missing["hard_image_webp"] += 1
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")

    edge_path = output_dir / f"hard_edges_{name.removeprefix('train_')}.jsonl"
    with edge_path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False, separators=(",", ":")) + "\n")

    captions = [row.caption_key for row in selected]
    hard_captions = [row.hard_caption_key for row in selected]
    all_texts = captions + hard_captions
    return {
        "rows": len(selected),
        "bucket_counts": dict(Counter(row.bucket for row in selected)),
        "label_type_counts": dict(Counter(row.label_type for row in selected)),
        "top_labels": Counter(row.label for row in selected).most_common(20),
        "top_scenes": Counter(str(row.raw.get("scene") or "") for row in selected).most_common(20),
        "unique_video_ids": len({row.video_id for row in selected}),
        "unique_source_captions": len({str(row.raw.get("source_caption") or "") for row in selected}),
        "unique_texts_across_caption_and_hard_c": len(set(all_texts)),
        "duplicate_text_rows_across_caption_and_hard_c": len(all_texts) - len(set(all_texts)),
        "hard_edge_count": len(edges),
        "hard_edge_endpoint_check": all(
            edge["anchor_image_id"] in selected_by_id and edge["hard_image_id"] in selected_by_id
            for edge in edges
        ),
        "image_path_missing": dict(missing),
        "manifest": str(subset_path),
        "edge_manifest": str(edge_path),
    }


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    if tuple(sizes) != SIZES:
        print(f"warning: non-paper sizes requested: {sizes}")
    split_payload = json.loads(args.video_splits.read_text(encoding="utf-8"))
    split_by_video = {str(key): value for key, value in split_payload["split_by_video"].items()}

    rows_by_id: dict[str, Row] = {}
    for raw in iter_rows(args.annotation_dir):
        label_type, label = label_info(raw)
        row = Row(
            raw=raw,
            image_id=str(raw["image_id"]),
            hard_i_id=str(raw.get("hard_i_id") or ""),
            video_id=int(raw["video_id"]),
            bucket=bucket_from_path(str(raw["image"])),
            label_type=label_type,
            label=label,
            caption_key=normalise_text(raw.get("caption")),
            hard_caption_key=normalise_text(raw.get("hard_c")),
        )
        rows_by_id[row.image_id] = row

    candidates = []
    rejected = Counter()
    for row in rows_by_id.values():
        target = rows_by_id.get(row.hard_i_id)
        if row.label == BAD_LABEL or target is None or target.label == BAD_LABEL:
            rejected["bad_or_missing_hard_target"] += 1
        elif split_by_video.get(str(row.video_id)) != "train" or split_by_video.get(str(target.video_id)) != "train":
            rejected["outside_train_split"] += 1
        elif not row.caption_key or not row.hard_caption_key:
            rejected["empty_text"] += 1
        elif row.caption_key == row.hard_caption_key:
            rejected["same_anchor_and_hard_text"] += 1
        else:
            candidates.append(row)

    by_bucket = defaultdict(list)
    for row in candidates:
        by_bucket[row.bucket].append(row)
    ordered = {
        bucket: candidate_order(by_bucket[bucket], rows_by_id, args.seed + index)
        for index, bucket in enumerate(BUCKETS)
    }
    selected_by_size = choose_nested(ordered, sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "annotation_dir": str(args.annotation_dir),
        "video_splits": str(args.video_splits),
        "candidate_rows": len(candidates),
        "candidate_bucket_counts": dict(Counter(row.bucket for row in candidates)),
        "rejected": dict(rejected),
        "subsets": {},
        "nesting_checks": {},
    }
    previous: set[str] = set()
    for size in sizes:
        name = f"train_{size // 1000}k_hard"
        selected = selected_by_size[size]
        current = {row.image_id for row in selected}
        if previous:
            summary["nesting_checks"][f"{len(previous) // 1000}k_subset_of_{size // 1000}k"] = previous <= current
            if not previous <= current:
                raise AssertionError("nested selection violated")
        summary["subsets"][name] = write_subset(
            selected, name, args.output_dir, args.image_root, args.check_images
        )
        previous = current

    (args.output_dir / "subset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
