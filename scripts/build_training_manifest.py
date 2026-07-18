#!/usr/bin/env python3
"""Build a single clean STAR manifest from a train-only subset and frozen splits.

``train`` contains the selected anchors plus each required hard-pair endpoint.
``dev`` is the sole checkpoint/hyperparameter selection split. ``report`` is
evaluation-only and must not be read by either training script.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, default=None)
    parser.add_argument("--pose-json", type=Path, default=None)
    parser.add_argument("--use-enhanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe-eval-captions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def annotation_rows(annotation_dir: Path) -> dict[str, dict]:
    output = {}
    for path in sorted(annotation_dir.glob("attr_*.json"), key=lambda item: item.name):
        for row in iter_jsonl(path):
            image_id = str(row["image_id"])
            if image_id in output:
                raise ValueError(f"Duplicate annotation image_id: {image_id}")
            output[image_id] = row
    if not output:
        raise FileNotFoundError(f"No attr_*.json rows in {annotation_dir}")
    return output


def load_items(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload)
    if isinstance(items, list):
        return {str(item["image_id"]): item for item in items}
    return {str(key): value for key, value in items.items()}


def normalise_path(value: str) -> str:
    value = str(value).replace("\\", "/").removeprefix("train/")
    return (Path("data/train_webp") / Path(value).with_suffix(".webp")).as_posix()


def label_info(row: dict) -> tuple[str, str]:
    if row.get("anomaly") is not None:
        return "anomaly", str(row["anomaly"])
    if row.get("normal") is not None:
        return "normal", str(row["normal"])
    return "unknown", ""


def bucket(path: str) -> str:
    for value in ("goal", "full", "wentwrong"):
        if f"/{value}/" in f"/{path}/":
            return value
    return "unknown"


def xyxy_to_xywh(value) -> list[float] | None:
    if not value or len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    return [x1, y1, max(1e-6, x2 - x1), max(1e-6, y2 - y1)]


def flat_keypoints(item: dict | None) -> list[float] | None:
    if not item or item.get("status") != "ok":
        return None
    instances = item.get("instances") or []
    if not instances:
        return None
    keypoints = instances[0].get("keypoints_xyc")
    if not keypoints or len(keypoints) != 17:
        return None
    width, height = float(item.get("width") or 1), float(item.get("height") or 1)
    out = []
    for x, y, confidence in keypoints:
        out.extend([float(x) / width, float(y) / height, float(confidence)])
    return out


def caption(row: dict, base: str, enhanced: str, use_enhanced: bool) -> str:
    if use_enhanced and str(row.get(enhanced) or "").strip():
        return str(row[enhanced]).strip()
    return str(row.get(base) or "").strip()


def record(
    row: dict,
    *,
    split: str,
    text: str,
    pair_image_id: str = "",
    is_hard_target: bool = False,
    anchor_image_id: str = "",
    boxes: dict[str, dict],
    poses: dict[str, dict],
) -> dict:
    image_id = str(row["image_id"])
    image_path = normalise_path(str(row["image"]))
    label_type, label = label_info(row)
    bbox_item = boxes.get(image_id, {})
    pose_item = poses.get(image_id)
    item = {
        "image_id": image_id,
        "pair_image_id": pair_image_id,
        "image_path": image_path,
        "caption": text,
        "caption_raw": str(row.get("caption") or ""),
        "sequence_id": f"v{row.get('video_id')}_{bucket(image_path)}",
        "label": label,
        "label_type": label_type,
        "bucket": bucket(image_path),
        "scene": str(row.get("scene") or ""),
        "video_id": int(row["video_id"]),
        "action": str(row.get("action") or ""),
        "source_caption": str(row.get("source_caption") or ""),
        "split": split,
        "is_hard_target": is_hard_target,
        "anchor_image_id": anchor_image_id,
        "bbox": xyxy_to_xywh(
            bbox_item.get("union_bbox_norm_xyxy") or bbox_item.get("primary_bbox_norm_xyxy")
        ),
        "bbox_status": bbox_item.get("status", "missing"),
    }
    keypoints = flat_keypoints(pose_item)
    if keypoints is not None:
        item["keypoints"] = keypoints
    return item


def assert_split(image_id: str, row: dict, expected: str, split_by_video: dict[str, str]) -> None:
    actual = split_by_video.get(str(row["video_id"]))
    if actual != expected:
        raise ValueError(f"{image_id} belongs to {actual!r}, expected {expected!r}")


def main() -> None:
    args = parse_args()
    selected = list(iter_jsonl(args.subset))
    if not selected:
        raise ValueError("Subset is empty")
    annotations = annotation_rows(args.annotation_dir)
    split_payload = json.loads(args.video_splits.read_text(encoding="utf-8"))
    split_by_video = {str(key): value for key, value in split_payload["split_by_video"].items()}
    boxes, poses = load_items(args.bbox_json), load_items(args.pose_json)

    train_records: dict[str, dict] = {}
    target_text: dict[str, tuple[str, str]] = {}
    for subset_row in selected:
        image_id = str(subset_row["image_id"])
        hard_id = str(subset_row["hard_i_id"])
        anchor = annotations.get(image_id)
        target = annotations.get(hard_id)
        if anchor is None or target is None:
            raise ValueError(f"Subset endpoint absent from annotation: {image_id}->{hard_id}")
        assert_split(image_id, anchor, "train", split_by_video)
        assert_split(hard_id, target, "train", split_by_video)
        if image_id in train_records:
            raise ValueError(f"Duplicate selected anchor: {image_id}")
        # The subset owns its enhancement fields. Annotation rows remain immutable originals.
        anchor_with_enhancement = dict(anchor)
        anchor_with_enhancement.update(
            {
                key: subset_row[key]
                for key in ("caption_enhanced", "hard_c_enhanced")
                if key in subset_row
            }
        )
        train_records[image_id] = record(
            anchor_with_enhancement,
            split="train",
            text=caption(anchor_with_enhancement, "caption", "caption_enhanced", args.use_enhanced),
            pair_image_id=hard_id,
            boxes=boxes,
            poses=poses,
        )
        text = caption(anchor_with_enhancement, "hard_c", "hard_c_enhanced", args.use_enhanced)
        previous = target_text.get(hard_id)
        if previous and previous[0] != text:
            raise ValueError(
                f"Conflicting hard caption for target {hard_id}: anchors {previous[1]} and {image_id}"
            )
        target_text[hard_id] = (text, image_id)

    for target_id, (text, anchor_id) in target_text.items():
        if target_id not in train_records:
            train_records[target_id] = record(
                annotations[target_id],
                split="train",
                text=text,
                is_hard_target=True,
                anchor_image_id=anchor_id,
                boxes=boxes,
                poses=poses,
            )

    train_ids = set(train_records)
    missing_partners = [
        row["image_id"] for row in train_records.values()
        if row["pair_image_id"] and row["pair_image_id"] not in train_ids
    ]
    if missing_partners:
        raise AssertionError(f"{len(missing_partners)} train rows lack in-manifest hard partners")

    records = list(train_records.values())
    eval_duplicate_counts = Counter()
    for split in ("dev", "report"):
        seen_captions: set[str] = set()
        for row in annotations.values():
            if split_by_video.get(str(row["video_id"])) != split:
                continue
            text = str(row.get("caption") or "").strip()
            key = " ".join(text.lower().split())
            if args.dedupe_eval_captions and key and key in seen_captions:
                text = ""
                eval_duplicate_counts[split] += 1
            elif key:
                seen_captions.add(key)
            records.append(record(row, split=split, text=text, boxes=boxes, poses=poses))

    frame = pd.DataFrame(records)
    if frame["image_id"].duplicated().any():
        duplicate = frame.loc[frame["image_id"].duplicated(), "image_id"].iat[0]
        raise AssertionError(f"Manifest duplicate image_id: {duplicate}")
    split_counts = frame.groupby("split").size().to_dict()
    if not all(split_counts.get(name, 0) for name in ("train", "dev", "report")):
        raise AssertionError(f"Empty split in manifest: {split_counts}")
    if (frame.loc[frame["split"].eq("train"), "caption"].str.strip() == "").any():
        raise AssertionError("Train split contains an empty caption")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    summary = {
        "subset": str(args.subset),
        "rows": int(len(frame)),
        "split_rows": {key: int(value) for key, value in split_counts.items()},
        "train_anchor_rows": len(selected),
        "train_hard_target_only_rows": int(frame.loc[frame["split"].eq("train"), "is_hard_target"].sum()),
        "eval_duplicate_caption_rows_made_gallery_only": dict(eval_duplicate_counts),
        "bbox_status": dict(Counter(frame["bbox_status"])),
        "pose_rows": int(frame.get("keypoints", pd.Series(dtype=object)).notna().sum()),
        "selection_split": "dev",
        "report_split": "report",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest: {args.output}")


if __name__ == "__main__":
    main()
