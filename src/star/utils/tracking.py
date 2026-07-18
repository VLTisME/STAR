"""Optional, failure-tolerant experiment tracking.

JSONL and TensorBoard remain the durable local record. Weights & Biases is an
additional private dashboard, never a dependency required to complete a run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(self, *, enabled: bool, config: dict[str, Any], out_dir: Path, run_kind: str):
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            train = config.get("train", {})
            mode = train.get("wandb_mode") or os.environ.get("WANDB_MODE") or "online"
            self.run = wandb.init(
                project=train.get("wandb_project") or "star-aicity-paper",
                entity=train.get("wandb_entity") or os.environ.get("WANDB_ENTITY"),
                group=train.get("wandb_group") or run_kind,
                name=train.get("wandb_run_name"),
                tags=list(train.get("wandb_tags") or ()),
                mode=mode,
                dir=str(out_dir),
                config=config,
                save_code=False,
            )
            if self.run and getattr(self.run, "url", None):
                (out_dir / "wandb_url.txt").write_text(f"{self.run.url}\n", encoding="utf-8")
        except Exception as exc:
            print(f"[tracking] W&B disabled: {type(exc).__name__}: {exc}")

    @classmethod
    def from_config(cls, cfg, out_dir: Path, run_kind: str) -> "ExperimentTracker":
        from ..config import to_dict

        return cls(
            enabled=bool(cfg.train.log_wandb),
            config=to_dict(cfg),
            out_dir=out_dir,
            run_kind=run_kind,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as exc:
            print(f"[tracking] W&B logging failed; continuing locally: {exc}")

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
            self.run = None
