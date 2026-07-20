#!/usr/bin/env python3
"""Fail-fast environment checks for STAR's isolated server environments."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def require_python_310() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"Expected Python 3.10; found {sys.version.split()[0]}. "
            "Do not use the host Python 3.12 for STAR model environments."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("pe", "bbox", "caption", "xvlm"), required=True)
    parser.add_argument("--xvlm-dir", type=Path, default=None)
    args = parser.parse_args()

    require_python_310()
    import numpy as np
    import torch

    if int(np.__version__.split(".")[0]) >= 2:
        raise SystemExit(f"NumPy must be <2 for the pinned stacks, found {np.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment.")

    print("python:", sys.version.split()[0], sys.executable)
    print("numpy:", np.__version__)
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))

    if args.stack in {"pe", "bbox"}:
        import torchvision

        print("torchvision:", torchvision.__version__)
        for name in ("timm", "open_clip", "transformers", "pandas", "pyarrow", "wandb"):
            print(f"{name}:", version(name))
    if args.stack == "pe":
        print("[ok] modern PE environment imports succeeded")
    elif args.stack == "bbox":
        print("ultralytics:", version("ultralytics"))
        from ultralytics import YOLO

        _ = YOLO
        print("[ok] YOLO11/RT-DETR-compatible bbox environment imports succeeded")
    elif args.stack == "caption":
        print("transformers:", version("transformers"))
        print("vllm:", version("vllm"))
        print("[ok] Qwen caption-enhancement environment imports succeeded")
    else:
        import torchvision

        print("torchvision:", torchvision.__version__)
        print("transformers:", version("transformers"))
        print("timm:", version("timm"))
        if version("transformers") != "4.12.5" or version("timm") != "0.4.9":
            raise SystemExit("X-VLM requires transformers==4.12.5 and timm==0.4.9")
        from transformers import BertTokenizer

        _ = BertTokenizer
        if args.xvlm_dir is None:
            raise SystemExit("--xvlm-dir is required for --stack xvlm")
        source = args.xvlm_dir / "models" / "model_retrieval.py"
        if not source.exists():
            raise SystemExit(f"X-VLM source missing: {source}")
        sys.path.insert(0, str(args.xvlm_dir))
        from models.model_retrieval import XVLM

        _ = XVLM
        print("[ok] legacy X-VLM environment and external source imports succeeded")


if __name__ == "__main__":
    main()
