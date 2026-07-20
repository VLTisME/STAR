"""Evaluate a PE checkpoint on a frozen manifest split without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.data import BBoxAwareTransform  # noqa: E402
from star.metrics import full_report  # noqa: E402
from star.pe import PEManifestDataset, PEVisionRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PE checkpoint on dev or report.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="report")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def evaluate(model, dataset, transform, args, device) -> dict[str, float]:
    def collate(batch):
        return {
            "image": torch.stack(
                [transform.apply(item["image"], item["bbox"]) for item in batch]
            ),
            "text": torch.stack([item["text_feature"] for item in batch]),
            "image_id": [item["image_id"] for item in batch],
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
    )
    image_features, text_features, image_ids = [], [], []
    model.eval()
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            image_features.append(model.encode_image(batch["image"].to(device)).float().cpu())
        text_features.append(F.normalize(batch["text"].float(), dim=-1))
        image_ids.extend(batch["image_id"])

    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    id_to_gallery, gallery_rows = {}, []
    for index, image_id in enumerate(image_ids):
        if image_id not in id_to_gallery:
            id_to_gallery[image_id] = len(gallery_rows)
            gallery_rows.append(index)
    frame = dataset.df
    query_rows = [
        index for index, caption in enumerate(frame["caption"].fillna("")) if str(caption).strip()
    ]
    sim = text_features[query_rows] @ image_features[gallery_rows].t()
    gt = torch.tensor([id_to_gallery[image_ids[index]] for index in query_rows])
    return full_report(sim, gt, ks=(1, 5, 10, 50, 200))


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt)
    saved_args = checkpoint.get("args") or {}
    model_id = saved_args.get("model", "hf-hub:timm/PE-Core-bigG-14-448")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PEVisionRetriever(model_id).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = PEManifestDataset(args.manifest, args.image_root, args.text_cache, split=args.split)
    transform = BBoxAwareTransform(
        size=448,
        enabled=False,
        mean=model.image_mean,
        std=model.image_std,
    )
    report = evaluate(model, dataset, transform, args, device)
    payload = {
        "split": args.split,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "model": model_id,
        **report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
