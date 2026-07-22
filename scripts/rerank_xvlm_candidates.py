"""Run X-VLM ITM on frozen PE candidates and select a clean dev configuration.

This script deliberately runs in the legacy ``star-xvlm`` environment.  It never
loads PE; the PE candidate payload is produced by ``export_pe_candidates.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import _merge, load_config, parse_overrides  # noqa: E402
from star.data import PABDataset  # noqa: E402
from star.inference.two_stage import (  # noqa: E402
    apply_postprocess,
    evaluate_order,
    minmax_per_query,
    rank_candidates,
)
from star.models import STARModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PE Top-K + X-VLM ITM reranking.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--xvlm-ckpt", required=True)
    parser.add_argument("--pe-candidates", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "report"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-config", default=None,
                        help="dev-selected JSON config; required for a final report run")
    parser.add_argument(
        "--report-postprocess-ablation",
        action="store_true",
        help=(
            "evaluate predeclared none/Greedy-SCA/Gale-Shapley variants at the DEV-frozen "
            "fusion alpha. This is REPORT-only measurement, never a new selection sweep."
        ),
    )
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2",
                        help="ITM + alpha * per-query min-max PE score")
    parser.add_argument("--postprocess", default="none,greedy_sca,gale_shapley")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--itm-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--reuse-itm-cache", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def encode_xvlm(model, dataset, gallery_ids, query_rows, batch_size, workers, device):
    gallery_position = {image_id: pos for pos, image_id in enumerate(gallery_ids)}
    first_gallery_row = {}
    for row, image_id in enumerate(dataset.df["image_id"].astype(str).tolist()):
        first_gallery_row.setdefault(image_id, row)
    expected_gallery_rows = {row: gallery_position[image_id] for image_id, row in first_gallery_row.items()}
    query_position = {row: pos for pos, row in enumerate(query_rows)}

    def collate(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        }

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=4 if workers else None,
    )
    gallery_embeds = query_embeds = query_masks = None
    model.eval()
    for step, batch in enumerate(loader, 1):
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            image_embeds, _ = model.backbone.encode_image(image)
            text_embeds, _ = model.backbone.encode_text(ids, mask)
        if gallery_embeds is None:
            gallery_embeds = torch.empty(
                (len(gallery_ids), *image_embeds.shape[1:]), dtype=torch.float16
            )
            query_embeds = torch.empty(
                (len(query_rows), *text_embeds.shape[1:]), dtype=torch.float16
            )
            query_masks = torch.empty((len(query_rows), mask.size(1)), dtype=torch.long)
        for local, row in enumerate(batch["row_index"].tolist()):
            if row in expected_gallery_rows:
                gallery_embeds[expected_gallery_rows[row]] = image_embeds[local].detach().cpu().half()
            if row in query_position:
                query_embeds[query_position[row]] = text_embeds[local].detach().cpu().half()
                query_masks[query_position[row]] = mask[local].detach().cpu()
        if step % 50 == 0:
            print(f"X-VLM encode {step}/{len(loader)} batches")
    assert gallery_embeds is not None and query_embeds is not None and query_masks is not None
    return gallery_embeds, query_embeds, query_masks


@torch.inference_mode()
def compute_itm(model, gallery_embeds, query_embeds, query_masks, candidate_indices, batch_size, device):
    query_count, topk = candidate_indices.shape
    flat_q = torch.arange(query_count).repeat_interleave(topk)
    flat_g = candidate_indices.reshape(-1)
    scores = torch.empty(flat_g.numel(), dtype=torch.float32)
    for start in range(0, flat_g.numel(), batch_size):
        end = min(start + batch_size, flat_g.numel())
        q = flat_q[start:end]
        g = flat_g[start:end]
        image = gallery_embeds[g].to(device, non_blocking=True)
        text = query_embeds[q].to(device, non_blocking=True)
        mask = query_masks[q].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.backbone.itm_logits(image, text, mask)
        scores[start:end] = (logits[:, 1] - logits[:, 0]).float().cpu()
        if (start // batch_size + 1) % 100 == 0:
            print(f"ITM pairs {end:,}/{flat_g.numel():,}")
    return scores.view(query_count, topk)


def evaluate_variant(pe_scores, gt, candidate_indices, candidate_scores, itm_scores, alpha, method):
    fused = itm_scores + float(alpha) * minmax_per_query(candidate_scores)
    order, ordered_scores = rank_candidates(candidate_indices, fused)
    order = apply_postprocess(order, ordered_scores, method)
    return {"kind": "xvlm_itm", "alpha": float(alpha), "postprocess": method,
            **evaluate_order(pe_scores, gt, order)}, order


def select_best(rows, pe_baseline):
    safe = [row for row in rows if row["R@10"] >= pe_baseline["R@10"] - 1e-9]
    candidates = safe or rows
    return max(candidates, key=lambda row: (row["mAP"], row["R@1"], row["R@5"], -row["top1_conflicts"]))


def main() -> None:
    args = parse_args()
    payload = load(args.pe_candidates)
    if payload.get("format") != "star-pe-candidates-v1":
        raise ValueError("Unsupported PE candidate cache format")
    if payload["split"] != args.split:
        raise ValueError(f"Candidate cache is for split={payload['split']}, not {args.split}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    itm_cache_path = output / "itm_cache.pt"

    cfg = load_config(args.config, parse_overrides(args.set))
    raw = load(args.xvlm_ckpt)
    embedded = (raw.get("extra") or {}).get("cfg")
    if embedded and "model" in embedded:
        _merge(cfg.model, embedded["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STARModel(cfg).to(device)
    message = model.load_state_dict(raw["model"], strict=False)
    print(f"loaded X-VLM: missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}")
    dataset = PABDataset(cfg.data.manifest, cfg.data.image_root, model.backbone.tokenizer,
                         split=args.split, image_size=cfg.data.image_size,
                         max_token=cfg.data.max_token, train=False)
    row_ids = dataset.df["image_id"].astype(str).tolist()
    if row_ids != payload["row_image_ids"]:
        raise ValueError("Manifest row order/image IDs differ from the PE candidate cache")
    query_rows = [row for row, caption in enumerate(dataset.df["caption"].fillna("")) if str(caption).strip()]
    if query_rows != payload["query_rows"]:
        raise ValueError("Query rows differ from the PE candidate cache")

    if args.reuse_itm_cache and itm_cache_path.exists():
        itm_payload = load(itm_cache_path)
        itm_scores = itm_payload["itm_scores"].float()
        if tuple(itm_scores.shape) != tuple(payload["candidate_indices"].shape):
            raise ValueError("Cached ITM scores do not match candidate shape")
        print(f"reused ITM cache: {itm_cache_path}")
    else:
        gallery_embeds, query_embeds, query_masks = encode_xvlm(
            model, dataset, payload["gallery_image_ids"], query_rows,
            args.encode_batch_size, args.workers, device,
        )
        itm_scores = compute_itm(
            model, gallery_embeds, query_embeds, query_masks,
            payload["candidate_indices"], args.itm_batch_size, device,
        )
        torch.save({"format": "star-xvlm-itm-v1", "itm_scores": itm_scores,
                    "candidate_shape": tuple(itm_scores.shape),
                    "xvlm_checkpoint": str(Path(args.xvlm_ckpt).resolve())}, itm_cache_path)
        del gallery_embeds, query_embeds, query_masks
        gc.collect()
        torch.cuda.empty_cache()

    pe_scores = payload["pe_scores"].float()
    gt = payload["gt_index"].long()
    candidate_indices = payload["candidate_indices"].long()
    candidate_scores = payload["candidate_scores"].float()
    pe_order = candidate_indices
    pe_baseline = {"kind": "pe_raw", "alpha": None, "postprocess": "none",
                   **evaluate_order(pe_scores, gt, pe_order)}
    rows = [pe_baseline]

    if args.frozen_config:
        frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
        # DEV can legitimately select PE alone.  Keep REPORT faithful to that
        # frozen decision instead of inventing an ITM fusion weight.
        if frozen.get("kind") == "pe_raw":
            variants = []
        elif args.report_postprocess_ablation:
            methods = [value.strip() for value in args.postprocess.split(",") if value.strip()]
            variants = [(frozen["alpha"], method) for method in methods]
        else:
            variants = [(frozen["alpha"], frozen["postprocess"])]
    else:
        alphas = [float(value) for value in args.alphas.split(",")]
        methods = [value.strip() for value in args.postprocess.split(",") if value.strip()]
        variants = [(alpha, method) for alpha in alphas for method in methods]
    best_order = pe_order
    for alpha, method in variants:
        row, order = evaluate_variant(pe_scores, gt, candidate_indices, candidate_scores, itm_scores, alpha, method)
        rows.append(row)
        print(row)
        if args.frozen_config or row is select_best(rows, pe_baseline):
            best_order = order
    if args.frozen_config:
        best = next(
            (
                row for row in rows
                if row["kind"] == frozen.get("kind")
                and row["alpha"] == frozen.get("alpha")
                and row["postprocess"] == frozen.get("postprocess")
            ),
            pe_baseline,
        )
    else:
        best = select_best(rows, pe_baseline)
    # Rebuild the selected order rather than relying on loop-object identity or
    # the order in which a fixed report ablation was requested.
    if best["kind"] == "pe_raw":
        best_order = pe_order
    else:
        _, best_order = evaluate_variant(pe_scores, gt, candidate_indices, candidate_scores, itm_scores,
                                          best["alpha"], best["postprocess"])
        (output / "best_dev_config.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")

    result = {"split": args.split, "pe_baseline": pe_baseline, "selected": best,
              "variants": rows, "topk": int(candidate_indices.size(1))}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    torch.save({"gallery_image_ids": payload["gallery_image_ids"],
                "query_image_ids": payload["query_image_ids"], "order": best_order.cpu(),
                "selected": best, "topk": int(candidate_indices.size(1))}, output / "best_ranking.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
