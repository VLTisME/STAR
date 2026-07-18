#!/usr/bin/env python3
"""Create deterministic, hard-pair-safe video-disjoint paper splits.

The split unit is a connected component of the video graph.  Every annotation
edge ``image_id -> hard_i_id`` joins the two endpoint videos, so hard-pair
training never crosses train/dev/report boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "dev", "report")
DEFAULT_RATIOS = {"train": 0.80, "dev": 0.10, "report": 0.10}


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def add(self, item: int) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: int) -> int:
        root = self.parent[item]
        if root != item:
            self.parent[item] = self.find(root)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def iter_rows(annotation_dir: Path) -> Iterable[dict]:
    files = sorted(annotation_dir.glob("attr_*.json"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No attr_*.json files in {annotation_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def file_fingerprint(annotation_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(annotation_dir.glob("attr_*.json"), key=lambda item: item.name):
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--report-ratio", type=float, default=0.10)
    return parser.parse_args()


def choose_split(counts: Counter, targets: dict[str, float], component_size: int) -> str:
    """Choose the partition with the greatest normalized remaining capacity."""
    deficits = {
        name: (targets[name] - counts[name]) / max(targets[name], 1.0)
        for name in SPLITS
    }
    # Deterministic split order resolves equal deficits.
    return max(SPLITS, key=lambda name: (deficits[name], -counts[name], -SPLITS.index(name)))


def build_split(rows: list[dict], ratios: dict[str, float], seed: int):
    image_to_video: dict[str, int] = {}
    image_to_hard: dict[str, str] = {}
    video_rows: Counter = Counter()
    dsu = DisjointSet()
    for row in rows:
        image_id = str(row["image_id"])
        video_id = int(row["video_id"])
        if image_id in image_to_video and image_to_video[image_id] != video_id:
            raise ValueError(f"image_id {image_id} appears in multiple videos")
        image_to_video[image_id] = video_id
        image_to_hard[image_id] = str(row.get("hard_i_id") or "")
        video_rows[video_id] += 1
        dsu.add(video_id)

    cross_video_edges = 0
    missing_targets = []
    for image_id, hard_id in image_to_hard.items():
        if not hard_id:
            continue
        target_video = image_to_video.get(hard_id)
        if target_video is None:
            missing_targets.append((image_id, hard_id))
            continue
        source_video = image_to_video[image_id]
        if source_video != target_video:
            cross_video_edges += 1
        dsu.union(source_video, target_video)
    if missing_targets:
        sample = ", ".join(f"{a}->{b}" for a, b in missing_targets[:5])
        raise ValueError(f"{len(missing_targets):,} hard targets are absent from annotations, e.g. {sample}")

    components: dict[int, list[int]] = defaultdict(list)
    for video_id in sorted(video_rows):
        components[dsu.find(video_id)].append(video_id)

    items = []
    rng = random.Random(seed)
    for root, videos in components.items():
        size = sum(video_rows[video] for video in videos)
        items.append((size, rng.random(), root, sorted(videos)))
    items.sort(key=lambda item: (-item[0], item[1], item[2]))

    total_rows = len(rows)
    targets = {name: total_rows * ratios[name] for name in SPLITS}
    counts: Counter = Counter()
    split_by_video: dict[str, str] = {}
    component_rows = []
    for size, _, root, videos in items:
        split = choose_split(counts, targets, size)
        counts[split] += size
        for video in videos:
            split_by_video[str(video)] = split
        component_rows.append(
            {"component": root, "split": split, "rows": size, "videos": videos}
        )

    violations = 0
    for image_id, hard_id in image_to_hard.items():
        if hard_id and split_by_video[str(image_to_video[image_id])] != split_by_video[str(image_to_video[hard_id])]:
            violations += 1
    if violations:
        raise AssertionError(f"hard-pair split leakage: {violations}")

    summary = {
        "schema_version": 1,
        "seed": seed,
        "ratios": ratios,
        "rows": total_rows,
        "row_counts": dict(counts),
        "row_ratios": {name: counts[name] / total_rows for name in SPLITS},
        "video_counts": dict(Counter(split_by_video.values())),
        "components": len(component_rows),
        "largest_component_rows": max(item["rows"] for item in component_rows),
        "cross_video_hard_edges": cross_video_edges,
        "hard_pair_leakage": violations,
    }
    return split_by_video, component_rows, summary


def main() -> None:
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "dev": args.dev_ratio,
        "report": args.report_ratio,
    }
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise SystemExit("Split ratios must be positive and sum to 1.0")

    rows = list(iter_rows(args.annotation_dir))
    split_by_video, components, summary = build_split(rows, ratios, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = file_fingerprint(args.annotation_dir)
    payload = {
        "schema_version": 1,
        "annotation_dir": str(args.annotation_dir),
        "annotation_fingerprint": fingerprint,
        "seed": args.seed,
        "ratios": ratios,
        "split_by_video": split_by_video,
    }
    (args.output_dir / "video_splits.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "components.json").write_text(
        json.dumps(components, indent=2) + "\n", encoding="utf-8"
    )
    summary["annotation_fingerprint"] = fingerprint
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote: {args.output_dir / 'video_splits.json'}")


if __name__ == "__main__":
    main()
