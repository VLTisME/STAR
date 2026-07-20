#!/usr/bin/env python3
"""Create compact, fixed video-disjoint dev/report retrieval suites.

Each output JSONL lists gallery images for one frozen split. Query targets are
included in that gallery and have ``is_query=true``; remaining rows are gallery
distractors. The suites are deterministic and are reused unchanged by every
10K/30K/50K experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


BUCKETS = ("goal", "full", "wentwrong")
BUCKET_RATIOS = {"goal": 0.37109, "full": 0.34398, "wentwrong": 0.28493}
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-queries", type=int, default=1_000)
    parser.add_argument("--dev-gallery", type=int, default=5_000)
    parser.add_argument("--report-queries", type=int, default=2_000)
    parser.add_argument("--report-gallery", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_rows(annotation_dir: Path) -> Iterable[dict]:
    paths = sorted(annotation_dir.glob("attr_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"No attr_*.json rows found in {annotation_dir}")
    for path in paths:
        yield from iter_jsonl(path)


def caption_key(value: object) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip().lower())


def bucket(row: dict) -> str:
    image = str(row.get("image") or "").replace("\\", "/")
    for name in BUCKETS:
        if f"/{name}/" in f"/{image}":
            return name
    raise ValueError(f"Cannot infer bucket for {row.get('image_id')}: {image}")


def stable_key(seed: int, split: str, purpose: str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{purpose}:{image_id}".encode()).hexdigest()


def quotas(total: int) -> dict[str, int]:
    raw = {name: total * BUCKET_RATIOS[name] for name in BUCKETS}
    result = {name: int(raw[name]) for name in BUCKETS}
    for name in sorted(BUCKETS, key=lambda key: raw[key] - result[key], reverse=True)[: total - sum(result.values())]:
        result[name] += 1
    return result


def select_queries(rows: list[dict], split: str, count: int, seed: int) -> list[dict]:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if caption_key(row.get("caption")):
            by_bucket[bucket(row)].append(row)
    for name in BUCKETS:
        by_bucket[name].sort(key=lambda row: stable_key(seed, split, "query", str(row["image_id"])))

    selected: list[dict] = []
    used_captions: set[str] = set()
    for name, need in quotas(count).items():
        for row in by_bucket[name]:
            key = caption_key(row.get("caption"))
            if key in used_captions:
                continue
            selected.append(row)
            used_captions.add(key)
            if sum(bucket(item) == name for item in selected) >= need:
                break
        if sum(bucket(item) == name for item in selected) < need:
            raise RuntimeError(f"Insufficient unique-caption {name} queries for {split}")

    if len(selected) != count:
        raise AssertionError(f"Selected {len(selected)} queries, expected {count}")
    return selected


def select_gallery(rows: list[dict], split: str, query_rows: list[dict], count: int, seed: int) -> list[dict]:
    if count < len(query_rows):
        raise ValueError(f"{split} gallery size {count} is smaller than query count {len(query_rows)}")
    selected = list(query_rows)
    selected_ids = {str(row["image_id"]) for row in selected}
    desired = quotas(count)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if str(row["image_id"]) not in selected_ids:
            by_bucket[bucket(row)].append(row)
    for name in BUCKETS:
        by_bucket[name].sort(key=lambda row: stable_key(seed, split, "gallery", str(row["image_id"])))

    existing = Counter(bucket(row) for row in selected)
    for name in BUCKETS:
        need = desired[name] - existing[name]
        if need < 0:
            raise RuntimeError(f"Query bucket quota exceeds gallery quota for {split}/{name}")
        if len(by_bucket[name]) < need:
            raise RuntimeError(f"Insufficient {name} gallery rows for {split}")
        selected.extend(by_bucket[name][:need])
    if len(selected) != count:
        raise AssertionError(f"Selected {len(selected)} gallery rows, expected {count}")
    return selected


def write_suite(path: Path, split: str, gallery_rows: list[dict], query_rows: list[dict]) -> dict:
    query_ids = {str(row["image_id"]) for row in query_rows}
    if not query_ids <= {str(row["image_id"]) for row in gallery_rows}:
        raise AssertionError(f"{split} query target missing from its gallery")
    query_captions = [caption_key(row.get("caption")) for row in query_rows]
    if len(query_captions) != len(set(query_captions)):
        raise AssertionError(f"{split} query captions are not unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(gallery_rows, key=lambda item: str(item["image_id"])):
            output = {
                "image_id": str(row["image_id"]),
                "split": split,
                "is_query": str(row["image_id"]) in query_ids,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    return {
        "gallery_rows": len(gallery_rows),
        "query_rows": len(query_rows),
        "bucket_counts": dict(Counter(bucket(row) for row in gallery_rows)),
    }


def main() -> None:
    args = parse_args()
    if min(args.dev_queries, args.dev_gallery, args.report_queries, args.report_gallery) <= 0:
        raise SystemExit("All query/gallery counts must be positive")
    payload = json.loads(args.video_splits.read_text(encoding="utf-8"))
    split_by_video = {str(key): str(value) for key, value in payload["split_by_video"].items()}
    rows_by_split: dict[str, list[dict]] = {"dev": [], "report": []}
    for row in iter_rows(args.annotation_dir):
        split = split_by_video.get(str(row.get("video_id")))
        if split in rows_by_split:
            rows_by_split[split].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "dev": (args.dev_queries, args.dev_gallery),
        "report": (args.report_queries, args.report_gallery),
    }
    summary = {"schema_version": 1, "seed": args.seed, "suites": {}}
    for split, (query_count, gallery_count) in settings.items():
        queries = select_queries(rows_by_split[split], split, query_count, args.seed)
        gallery = select_gallery(rows_by_split[split], split, queries, gallery_count, args.seed)
        summary["suites"][split] = write_suite(
            args.output_dir / f"{split}_eval.jsonl", split, gallery, queries
        )
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
