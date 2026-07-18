"""Evaluate a checkpoint on a frozen split (bi-encoder retrieval: mAP / MRR / R@K).

The trainer embeds the run config inside the checkpoint (`extra.cfg`), so the exact model
architecture (incl. any --set overrides used at train time) is rebuilt automatically here;
the YAML --config still controls data paths/split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import _merge, load_config, parse_overrides  # noqa: E402
from star.data import PABDataset                 # noqa: E402
from star.engine import evaluate_retrieval       # noqa: E402
from star.models import STARModel                # noqa: E402
from star.utils import get_logger                # noqa: E402

log = get_logger("star.eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=None, help="defaults to data.report_split")
    ap.add_argument("--output", default=None, help="write metrics JSON for table/ledger tooling")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.manifest=...")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    split = args.split or cfg.data.report_split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(args.ckpt, map_location="cpu")
    embedded = (raw.get("extra") or {}).get("cfg")
    if embedded and "model" in embedded:
        _merge(cfg.model, embedded["model"])     # rebuild the EXACT trained architecture
        log.info("using model config embedded in the checkpoint")

    model = STARModel(cfg).to(device)
    msg = model.load_state_dict(raw["model"], strict=False)
    log.info(f"loaded ckpt: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    ds = PABDataset(cfg.data.manifest, cfg.data.image_root, model.backbone.tokenizer,
                    split=split, image_size=cfg.data.image_size,
                    max_token=cfg.data.max_token, train=False)
    report = evaluate_retrieval(model, ds, device, num_workers=cfg.data.num_workers)
    payload = {"split": split, "checkpoint": str(Path(args.ckpt).resolve()), **report}
    log.info(f"[{split}] {report}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info(f"saved metrics: {output}")


if __name__ == "__main__":
    main()
