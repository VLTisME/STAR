#!/usr/bin/env python3
"""Write reproducibility metadata next to a paper experiment run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=("smoke", "pe", "xvlm", "evaluation"))
    args = parser.parse_args()
    for path in (args.config, args.manifest, args.video_splits):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.resolved.yaml").write_bytes(args.config.read_bytes())
    payload = {
        "run_id": args.run_id,
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "files": {
            str(args.config): sha256(args.config),
            str(args.manifest): sha256(args.manifest),
            str(args.video_splits): sha256(args.video_splits),
        },
    }
    (args.out_dir / "data_fingerprint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
