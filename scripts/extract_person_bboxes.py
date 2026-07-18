from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract person boxes for one sampled subset with YOLO11.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/train_webp"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--rtdetr-fallback", action="store_true")
    parser.add_argument("--rtdetr-model", default="PekingU/rtdetr_r18vd")
    parser.add_argument("--qa-dir", type=Path, default=None)
    parser.add_argument("--qa-samples", type=int, default=500)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_path(value: str, image_root: Path) -> Path:
    text = str(value).replace("\\", "/")
    if text.startswith("data/train_webp/"):
        text = text[len("data/train_webp/") :]
    elif text.startswith("train_webp/"):
        text = text[len("train_webp/") :]
    path = Path(text)
    return path if path.is_absolute() else image_root / path


def collect_assets(rows: list[dict], image_root: Path) -> list[dict]:
    by_path: dict[str, dict] = {}
    for row in rows:
        for role, path_field, id_field in (
            ("image_webp", "image_webp", "image_id"),
            ("hard_image_webp", "hard_image_webp", "hard_i_id"),
        ):
            path = resolve_path(str(row[path_field]), image_root)
            key = str(path.resolve())
            entry = by_path.setdefault(
                key,
                {
                    "image_id": str(row[id_field]),
                    "image_webp": str(row[path_field]),
                    "image_path": str(path),
                    "source_roles": [],
                },
            )
            if role not in entry["source_roles"]:
                entry["source_roles"].append(role)
    return sorted(by_path.values(), key=lambda item: item["image_id"])


def load_partial(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    items = {}
    for row in iter_jsonl(path):
        items[str(row["image_id"])] = row
    return items


def normalize_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def format_result(asset: dict, result) -> dict:
    with Image.open(asset["image_path"]) as image:
        width, height = image.size
    instances = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None:
        coords = boxes.xyxy.detach().float().cpu().tolist()
        scores = boxes.conf.detach().float().cpu().tolist()
        classes = boxes.cls.detach().long().cpu().tolist()
        for xyxy, score, cls in zip(coords, scores, classes):
            if int(cls) != 0:
                continue
            norm = normalize_box([float(v) for v in xyxy], width, height)
            instances.append({"bbox_norm_xyxy": norm, "score": float(score)})

    item = {
        **asset,
        "width": width,
        "height": height,
        "status": "ok" if instances else "no_person",
        "detector": "yolo11",
        "primary_bbox_norm_xyxy": None,
        "union_bbox_norm_xyxy": None,
        "instances": instances,
    }
    if not instances:
        return item

    primary = max(instances, key=lambda inst: inst["score"] * max(box_area(inst["bbox_norm_xyxy"]), 1e-8))
    union = [
        min(inst["bbox_norm_xyxy"][0] for inst in instances),
        min(inst["bbox_norm_xyxy"][1] for inst in instances),
        max(inst["bbox_norm_xyxy"][2] for inst in instances),
        max(inst["bbox_norm_xyxy"][3] for inst in instances),
    ]
    item["primary_bbox_norm_xyxy"] = primary["bbox_norm_xyxy"]
    item["union_bbox_norm_xyxy"] = (
        primary["bbox_norm_xyxy"] if box_area(union) > 0.85 else union
    )
    return item


def run_rtdetr_fallback(
    assets: list[dict],
    model_name: str,
    device: str,
    conf: float,
    batch_size: int,
) -> list[dict]:
    try:
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection
    except ImportError as exc:
        raise SystemExit("RT-DETR fallback requires torch and transformers") from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device).eval()
    person_ids = {
        int(index)
        for index, label in model.config.id2label.items()
        if str(label).lower() == "person"
    }
    output = []
    for start in range(0, len(assets), batch_size):
        batch = assets[start : start + batch_size]
        images = []
        sizes = []
        for asset in batch:
            with Image.open(asset["image_path"]) as image:
                rgb = image.convert("RGB")
                images.append(rgb.copy())
                sizes.append((rgb.height, rgb.width))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            predictions = model(**inputs)
        results = processor.post_process_object_detection(
            predictions,
            threshold=conf,
            target_sizes=torch.tensor(sizes, device=device),
        )
        for asset, result, image in zip(batch, results, images):
            instances = []
            for box, score, label in zip(
                result["boxes"].cpu().tolist(),
                result["scores"].cpu().tolist(),
                result["labels"].cpu().tolist(),
            ):
                if int(label) not in person_ids:
                    continue
                instances.append(
                    {
                        "bbox_norm_xyxy": normalize_box(
                            [float(value) for value in box], image.width, image.height
                        ),
                        "score": float(score),
                    }
                )
            item = {
                **asset,
                "width": image.width,
                "height": image.height,
                "status": "ok" if instances else "no_person",
                "detector": "rtdetr",
                "primary_bbox_norm_xyxy": None,
                "union_bbox_norm_xyxy": None,
                "instances": instances,
            }
            if instances:
                primary = max(
                    instances,
                    key=lambda inst: inst["score"]
                    * max(box_area(inst["bbox_norm_xyxy"]), 1e-8),
                )
                union = [
                    min(inst["bbox_norm_xyxy"][0] for inst in instances),
                    min(inst["bbox_norm_xyxy"][1] for inst in instances),
                    max(inst["bbox_norm_xyxy"][2] for inst in instances),
                    max(inst["bbox_norm_xyxy"][3] for inst in instances),
                ]
                item["primary_bbox_norm_xyxy"] = primary["bbox_norm_xyxy"]
                item["union_bbox_norm_xyxy"] = (
                    primary["bbox_norm_xyxy"] if box_area(union) > 0.85 else union
                )
            output.append(item)
    return output


def write_qa(items: list[dict], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return
    stride = max(1, len(items) // max(1, limit))
    for item in items[::stride][:limit]:
        with Image.open(item["image_path"]) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for instance in item.get("instances", []):
            x1, y1, x2, y2 = instance["bbox_norm_xyxy"]
            draw.rectangle(
                [x1 * canvas.width, y1 * canvas.height, x2 * canvas.width, y2 * canvas.height],
                outline="yellow",
                width=3,
            )
        primary = item.get("primary_bbox_norm_xyxy")
        if primary:
            x1, y1, x2, y2 = primary
            draw.rectangle(
                [x1 * canvas.width, y1 * canvas.height, x2 * canvas.width, y2 * canvas.height],
                outline="red",
                width=4,
            )
        canvas.save(output_dir / f"{item['image_id']}.jpg", quality=90)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install bbox dependencies with: pip install ultralytics") from exc

    rows = list(iter_jsonl(args.subset))
    assets = collect_assets(rows, args.image_root)
    missing = [item["image_path"] for item in assets if not Path(item["image_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing):,} images are missing. First: {missing[:10]}")
    if args.limit is not None:
        assets = assets[: args.limit]

    partial = args.output.with_suffix(".partial.jsonl")
    if args.overwrite:
        partial.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)
    done = load_partial(partial) if args.resume else {}
    pending = [item for item in assets if item["image_id"] not in done]

    model = YOLO(args.model)
    status = Counter(item.get("status", "unknown") for item in done.values())
    started = time.time()
    partial.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and partial.exists() else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            results = model.predict(
                source=[item["image_path"] for item in batch],
                batch=len(batch),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                classes=[0],
                device=args.device,
                half=str(args.device).startswith("cuda"),
                verbose=False,
                stream=False,
            )
            for asset, result in zip(batch, results):
                item = format_result(asset, result)
                done[item["image_id"]] = item
                status[item["status"]] += 1
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            processed = min(start + len(batch), len(pending))
            total_done = len(done)
            if total_done % args.progress_every < len(batch) or processed == len(pending):
                speed = processed / max(time.time() - started, 1e-6)
                print(
                    f"bbox {processed:,}/{len(pending):,}; total {total_done:,}/{len(assets):,}; "
                    f"{speed:.1f} img/s; status {dict(status)}"
                )

    if args.rtdetr_fallback:
        fallback_assets = [
            asset for asset in assets if done.get(asset["image_id"], {}).get("status") == "no_person"
        ]
        if fallback_assets:
            print(f"RT-DETR fallback: {len(fallback_assets):,} no-person assets")
            recovered = run_rtdetr_fallback(
                fallback_assets,
                args.rtdetr_model,
                args.device,
                args.conf,
                max(1, min(args.batch_size, 32)),
            )
            with partial.open("a", encoding="utf-8") as handle:
                for item in recovered:
                    old_status = done[item["image_id"]]["status"]
                    status[old_status] -= 1
                    status[item["status"]] += 1
                    done[item["image_id"]] = item
                    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "subset": str(args.subset),
            "model": args.model,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "status": dict(status),
        },
        "items": {image_id: done[image_id] for image_id in sorted(done)},
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(args.output)
    coverage = status["ok"] / max(1, sum(status.values()))
    print(f"output: {args.output}")
    print(f"person coverage: {coverage:.2%}")
    if coverage < 0.97:
        print("WARNING: coverage <97%. Review no_person images; optionally run RT-DETR-R18 on them.")
    if args.qa_dir:
        write_qa(list(payload["items"].values()), args.qa_dir, args.qa_samples)
        print(f"QA overlays: {args.qa_dir}")


if __name__ == "__main__":
    main()
