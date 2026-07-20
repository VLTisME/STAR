#!/usr/bin/env python3
"""Prepare one external X-VLM checkout for STAR's pinned legacy environment.

Run this only inside the dedicated Python 3.10 ``star-xvlm`` virtual environment after
installing ``requirements-xvlm.txt`` plus ``transformers==4.12.5`` and ``timm==0.4.9`` with
``--no-deps``. The script is idempotent and records no private paths in this repository.

Why these patches exist:
  * Transformers 4.12.5 hard-pins a tokenizers version with no practical modern wheel. X-VLM
    uses the slow pure-Python BertTokenizer, so we safely relax that import-time version check.
  * The original CIDEr import is unrelated to retrieval and requires unavailable caption-eval deps.
  * SciPy removed interp2d; use the standard torch bicubic interpolation for Swin rel-pos tables.
  * Newer torch defaults can reject trusted checkpoint metadata unless weights_only=False is explicit.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


X_VLM_URL = "https://github.com/zengyan-97/X-VLM.git"
X_VLM_COMMIT = "cb4fff15bcc30bba710a7717318e6b963e6935a4"
PATCH_MARKER = "STAR compatibility patch: torch bicubic relative-position interpolation"

BICUBIC_PATCH = f'''\n\n# --- {PATCH_MARKER} ---
import torch as _star_torch


def interpolate_relative_pos_embed(rel_pos_bias, dst_num_pos, param_name=""):
    src_num_pos, num_heads = rel_pos_bias.size()
    src_size = int(src_num_pos ** 0.5)
    dst_size = int(dst_num_pos ** 0.5)
    if src_size != dst_size:
        print(
            "Position interpolate %s from %dx%d to %dx%d (torch bicubic)"
            % (param_name, src_size, src_size, dst_size, dst_size)
        )
        rel = rel_pos_bias.detach().float().permute(1, 0).reshape(
            1, num_heads, src_size, src_size
        )
        rel = _star_torch.nn.functional.interpolate(
            rel, size=(dst_size, dst_size), mode="bicubic", align_corners=False
        )
        rel_pos_bias = rel.reshape(num_heads, dst_size * dst_size).permute(1, 0).to(
            rel_pos_bias.dtype
        )
    return rel_pos_bias
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xvlm-dir", type=Path, required=True)
    parser.add_argument(
        "--clone-if-missing",
        action="store_true",
        help="Clone the pinned upstream X-VLM revision when --xvlm-dir is absent.",
    )
    return parser.parse_args()


def run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_xvlm_source(path: Path, clone_if_missing: bool) -> None:
    if not path.exists():
        if not clone_if_missing:
            raise FileNotFoundError(
                f"X-VLM checkout missing: {path}. Re-run with --clone-if-missing or clone it manually."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", X_VLM_URL, str(path))
    if not (path / ".git").exists():
        raise RuntimeError(f"{path} exists but is not a Git checkout.")
    run("git", "-C", str(path), "fetch", "--depth", "1", "origin", X_VLM_COMMIT)
    run("git", "-C", str(path), "checkout", "--detach", X_VLM_COMMIT)
    required = path / "models" / "model_retrieval.py"
    if not required.exists():
        raise RuntimeError(f"Pinned X-VLM checkout is incomplete; missing {required}.")


def patch_transformers_tokenizer_requirement() -> None:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "transformers is missing. Install transformers==4.12.5 with --no-deps first."
        )
    table = Path(spec.origin).parent / "dependency_versions_table.py"
    content = table.read_text(encoding="utf-8")
    patched = re.sub(
        r'^(\s*"tokenizers":\s*).+$',
        r'\1"tokenizers",',
        content,
        flags=re.MULTILINE,
    )
    if patched == content:
        if '"tokenizers": "tokenizers",' not in content:
            raise RuntimeError(f"Could not locate the tokenizers requirement in {table}")
        print(f"[ok] transformers tokenizers requirement already relaxed: {table}")
        return
    compile(patched, str(table), "exec")
    table.write_text(patched, encoding="utf-8")
    print(f"[patched] relaxed tokenizers requirement: {table}")


def patch_optional_cider(xvlm_dir: Path) -> None:
    path = xvlm_dir / "utils" / "__init__.py"
    content = path.read_text(encoding="utf-8")
    target = "from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD"
    replacement = f"try:\n    {target}\nexcept Exception:\n    CiderD = None"
    if replacement in content:
        print(f"[ok] optional CIDEr patch already present: {path}")
        return
    if target not in content:
        raise RuntimeError(f"Could not locate CIDEr import in {path}")
    path.write_text(content.replace(target, replacement), encoding="utf-8")
    print(f"[patched] made unused CIDEr import optional: {path}")


def patch_swin_interpolation(xvlm_dir: Path) -> None:
    path = xvlm_dir / "models" / "swin_transformer.py"
    content = path.read_text(encoding="utf-8")
    if PATCH_MARKER in content:
        print(f"[ok] torch-bicubic interpolation patch already present: {path}")
        return
    path.write_text(content + BICUBIC_PATCH, encoding="utf-8")
    print(f"[patched] replaced removed scipy interp2d path with torch bicubic: {path}")


def patch_torch_load(xvlm_dir: Path) -> None:
    path = xvlm_dir / "models" / "xvlm.py"
    content = path.read_text(encoding="utf-8")
    old = "torch.load(ckpt_rpath, map_location='cpu')"
    new = "torch.load(ckpt_rpath, map_location='cpu', weights_only=False)"
    if new in content:
        print(f"[ok] trusted checkpoint load is explicit: {path}")
        return
    if old not in content:
        print(f"[note] no legacy torch.load call found in {path}; no patch needed")
        return
    path.write_text(content.replace(old, new), encoding="utf-8")
    print(f"[patched] made trusted checkpoint load compatible with torch>=2.6: {path}")


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"X-VLM setup requires Python 3.10, found {sys.version.split()[0]}. "
            "Use /usr/bin/python3.10 to create the dedicated environment."
        )
    args = parse_args()
    xvlm_dir = args.xvlm_dir.resolve()
    ensure_xvlm_source(xvlm_dir, args.clone_if_missing)
    patch_transformers_tokenizer_requirement()
    patch_optional_cider(xvlm_dir)
    patch_swin_interpolation(xvlm_dir)
    patch_torch_load(xvlm_dir)
    print("[done] X-VLM source and legacy transformers compatibility are ready.")


if __name__ == "__main__":
    main()
