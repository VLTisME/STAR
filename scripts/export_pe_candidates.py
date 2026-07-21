"""Export frozen PE Top-K candidates for the legacy X-VLM environment.

Run this in the modern ``star-pe`` environment.  The resulting .pt contains
only CPU tensors and can be consumed by ``rerank_xvlm_candidates.py`` under the
isolated legacy X-VLM environment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.data import BBoxAwareTransform  # noqa: E402
from star.metrics import full_report  # noqa: E402
from star.pe import PEManifestDataset, PEVisionRetriever  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PE Top-K candidates for X-VLM ITM.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "report"))
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def encode(model, dataset, transform, cfg, device):
    def collate(batch):
        return {
            "image": torch.stack([transform.apply(item["image"], item["bbox"]) for item in batch]),
            "text": torch.stack([item["text_feature"] for item in batch]),
            "image_id": [item["image_id"] for item in batch],
        }

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers else None,
    )
    visual, text, image_ids = [], [], []
    model.eval()
    for step, batch in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            visual.append(model.encode_image(batch["image"].to(device, non_blocking=True)).float().cpu())
        text.append(F.normalize(batch["text"].float(), dim=-1))
        image_ids.extend(batch["image_id"])
        if step % 50 == 0:
            print(f"PE encode {step}/{len(loader)} batches")
    return torch.cat(visual), torch.cat(text), image_ids


def main() -> None:
    cfg = args()
    output = Path(cfg.output)
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    checkpoint = load(cfg.ckpt)
    model_id = (checkpoint.get("args") or {}).get("model", "hf-hub:timm/PE-Core-bigG-14-448")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PEVisionRetriever(model_id).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = PEManifestDataset(cfg.manifest, cfg.image_root, cfg.text_cache, split=cfg.split)
    transform = BBoxAwareTransform(size=448, enabled=False, mean=model.image_mean, std=model.image_std)
    image_features, text_features, image_ids = encode(model, dataset, transform, cfg, device)

    gallery_rows, gallery_by_id = [], {}
    for row, image_id in enumerate(image_ids):
        if image_id not in gallery_by_id:
            gallery_by_id[image_id] = len(gallery_rows)
            gallery_rows.append(row)
    captions = dataset.df["caption"].fillna("").astype(str).tolist()
    query_rows = [row for row, caption in enumerate(captions) if caption.strip()]
    sim = text_features[query_rows] @ image_features[gallery_rows].t()
    gt = torch.tensor([gallery_by_id[image_ids[row]] for row in query_rows], dtype=torch.long)
    if cfg.topk > sim.size(1):
        raise ValueError(f"topk={cfg.topk} exceeds gallery={sim.size(1)}")
    candidate_scores, candidate_indices = torch.topk(sim, k=cfg.topk, dim=1)
    report = full_report(sim, gt, ks=(1, 5, 10, 50, 200))
    payload = {
        "format": "star-pe-candidates-v1",
        "split": cfg.split,
        "topk": cfg.topk,
        "model": model_id,
        "pe_checkpoint": str(Path(cfg.ckpt).resolve()),
        "pe_checkpoint_sha256": sha256(cfg.ckpt),
        "manifest": str(Path(cfg.manifest).resolve()),
        "row_image_ids": image_ids,
        "gallery_image_ids": [image_ids[row] for row in gallery_rows],
        "query_rows": query_rows,
        "query_image_ids": [image_ids[row] for row in query_rows],
        "gt_index": gt,
        "pe_scores": sim.float(),
        "candidate_indices": candidate_indices.long(),
        "candidate_scores": candidate_scores.float(),
        "pe_metrics": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({"output": str(output), "queries": len(query_rows), "gallery": len(gallery_rows),
           "topk": cfg.topk, "pe_metrics": report})
    del model, image_features, text_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
