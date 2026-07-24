from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LABEL_FIELDS = ("normal", "anomaly")
PREFIX_RE = re.compile(r"^\s*(enhanced caption|caption|rewrite|final caption|patched caption)\s*:\s*", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")
MARKDOWN_RE = re.compile(r"(^|\s)([*_`#>-]{1,3})(\s|$)|```", re.MULTILINE)


@dataclass(slots=True)
class Task:
    row_index: int
    output_field: str
    original_text: str
    label_type: str
    action_label: str
    source_caption: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance subset captions with natural action-label integration using a local HF instruct model."
    )
    parser.add_argument("--subset", type=Path, required=True, help="Subset JSONL to update or smoke-test.")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/annotation/train_processed"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", type=int, default=10, help="Rows to print when not using --write.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process.")
    parser.add_argument("--write", action="store_true", help="Update the subset JSONL in place and create .bak.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a --write run from its local sidecar progress file.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Optional JSONL sidecar for resumable completed caption tasks.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=512,
        help="Persist the subset every N generated tasks during --write; 0 writes only at the end.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing enhanced fields.")
    parser.add_argument(
        "--label-type",
        choices=("anomaly", "normal", "all"),
        default="anomaly",
        help="Which labels to enhance. Default: anomaly only.",
    )
    parser.add_argument(
        "--clear-unselected",
        action="store_true",
        help="When writing, reset non-selected enhanced fields to their original caption text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def label_from_row(row: dict) -> tuple[str | None, str | None]:
    for field in LABEL_FIELDS:
        value = row.get(field)
        if value:
            return field, str(value)
    field = row.get("label_type")
    value = row.get("label")
    if field and value:
        return str(field), str(value)
    return None, None


def annotation_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def annotation_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("attr_*.json"), key=annotation_index)


def load_label_map(annotation_dir: Path) -> dict[str, dict]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation dir does not exist: {annotation_dir}")

    labels: dict[str, dict] = {}
    for path in annotation_files(annotation_dir):
        for row in iter_jsonl(path):
            label_type, label = label_from_row(row)
            if label:
                labels[str(row["image_id"])] = {
                    "label_type": label_type,
                    "label": label,
                    "source_caption": str(row.get("source_caption") or ""),
                }
    return labels


def build_prompt(
    original_text: str,
    action_label: str,
    source_caption: str,
    label_type: str | None = None,
) -> str:
    return (
        "Rewrite the caption with minimal changes so it naturally describes the anomalous "
        "action indicated by the event description. Preserve the visible facts and return only "
        "the caption.\n\n"
        f"Caption: {original_text}\n"
        f"Event description: {source_caption}\n"
        f"Action: {action_label}"
    )


def chat_or_plain_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful caption editor for pedestrian anomaly image retrieval.",
        },
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_name: str, device_arg: str, trust_remote_code: bool):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "enhance_caption.py needs torch and transformers. Install them in the environment that will run caption enhancement."
        ) from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_generation(text: str) -> str:
    text = text.strip()
    text = PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.strip("` \t\r\n")
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return " ".join(text.split())


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))


def output_is_bad(output: str, original_text: str, action_label: str) -> bool:
    if not output.strip():
        return True
    if MARKDOWN_RE.search(output):
        return True

    original_count = token_count(original_text)
    output_count = token_count(output)
    if original_count and not 0.75 * original_count <= output_count <= 1.35 * original_count:
        return True
    return False


def generate_vllm(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    batch_size: int,
) -> list[str]:
    outputs = []
    for start, batch_outputs in generate_vllm_batches(
        model_name,
        prompts,
        max_new_tokens,
        temperature,
        trust_remote_code,
        gpu_memory_utilization,
        batch_size,
    ):
        outputs.extend(batch_outputs)
        print(f"enhanced {min(start + len(batch_outputs), len(prompts)):,}/{len(prompts):,}")
    return outputs


def generate_vllm_batches(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    batch_size: int,
) -> Iterable[tuple[int, list[str]]]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit("The vLLM backend failed to import. See the traceback above.") from exc

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    for start in range(0, len(prompts), batch_size):
        rendered = [
            chat_or_plain_prompt(tokenizer, prompt)
            for prompt in prompts[start : start + batch_size]
        ]
        results = llm.generate(rendered, params, use_tqdm=False)
        yield start, [result.outputs[0].text.strip() for result in results]


def generate_batch(tokenizer, model, device: str, prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]:
    import torch

    rendered = [chat_or_plain_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(device)
    input_width = inputs["input_ids"].shape[1]
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        kwargs.update({"do_sample": False})

    with torch.inference_mode():
        outputs = model.generate(**inputs, **kwargs)

    return [tokenizer.decode(output[input_width:], skip_special_tokens=True).strip() for output in outputs]


def task_key(task: Task) -> str:
    payload = {
        "row_index": task.row_index,
        "output_field": task.output_field,
        "original_text": task.original_text,
        "label_type": task.label_type,
        "action_label": task.action_label,
        "source_caption": task.source_caption,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_progress_path(subset: Path) -> Path:
    return subset.with_suffix(subset.suffix + ".enhance_progress.jsonl")


def load_progress(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    completed: dict[str, str] = {}
    for line_number, line in enumerate(path.open("r", encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = str(record["task_key"])
            output = str(record["output"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid progress record at {path}:{line_number}") from exc
        completed[key] = output
    return completed


def append_progress(path: Path, tasks: list[Task], outputs: list[str]) -> None:
    if len(tasks) != len(outputs):
        raise ValueError("Progress task/output length mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for task, output in zip(tasks, outputs):
            record = {
                "task_key": task_key(task),
                "row_index": task.row_index,
                "output_field": task.output_field,
                "output": output,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def apply_saved_outputs(rows: list[dict], tasks: list[Task], completed: dict[str, str]) -> int:
    restored = 0
    for task in tasks:
        output = completed.get(task_key(task))
        if output is None:
            continue
        rows[task.row_index][task.output_field] = output
        restored += 1
    return restored


def selected_label_type(label_type: str | None, target: str) -> bool:
    return target == "all" or label_type == target


def build_tasks(
    rows: list[dict],
    label_map: dict[str, dict],
    overwrite: bool,
    limit: int | None,
    label_type_target: str,
) -> list[Task]:
    tasks: list[Task] = []
    selected_rows = rows[:limit] if limit is not None else rows
    for row_index, row in enumerate(selected_rows):
        anchor_label_type, anchor_label = label_from_row(row)
        hard_meta = label_map.get(str(row.get("hard_i_id", "")), {})
        hard_label_type = hard_meta.get("label_type")
        hard_label = hard_meta.get("label")
        hard_source_caption = str(hard_meta.get("source_caption") or row.get("source_caption") or "")

        if (
            selected_label_type(anchor_label_type, label_type_target)
            and anchor_label
            and row.get("caption")
            and (overwrite or not row.get("caption_enhanced"))
        ):
            original = str(row["caption"])
            tasks.append(
                Task(
                    row_index=row_index,
                    output_field="caption_enhanced",
                    original_text=original,
                    label_type=str(anchor_label_type),
                    action_label=anchor_label,
                    source_caption=str(row.get("source_caption") or ""),
                    prompt=build_prompt(
                        original,
                        anchor_label,
                        str(row.get("source_caption") or ""),
                        anchor_label_type,
                    ),
                )
            )

        if (
            selected_label_type(str(hard_label_type) if hard_label_type else None, label_type_target)
            and hard_label
            and row.get("hard_c")
            and (overwrite or not row.get("hard_c_enhanced"))
        ):
            original = str(row["hard_c"])
            tasks.append(
                Task(
                    row_index=row_index,
                    output_field="hard_c_enhanced",
                    original_text=original,
                    label_type=str(hard_label_type),
                    action_label=str(hard_label),
                    source_caption=hard_source_caption,
                    prompt=build_prompt(
                        original,
                        str(hard_label),
                        hard_source_caption,
                        str(hard_label_type),
                    ),
                )
            )

    return tasks


def finalize_output(task: Task, output: str) -> str:
    raw_output = output or task.original_text
    if MARKDOWN_RE.search(raw_output):
        return task.original_text
    output = clean_generation(raw_output)
    if task.label_type == "anomaly" and output_is_bad(output, task.original_text, task.action_label):
        output = task.original_text
    return output


def apply_outputs(rows: list[dict], tasks: list[Task], outputs: list[str]) -> None:
    for task, output in zip(tasks, outputs):
        rows[task.row_index][task.output_field] = finalize_output(task, output)


def initialize_enhanced_fields(rows: list[dict], overwrite: bool, limit: int | None) -> None:
    selected_rows = rows[:limit] if limit is not None else rows
    for row in selected_rows:
        if row.get("caption") and (overwrite or not row.get("caption_enhanced")):
            row["caption_enhanced"] = str(row["caption"])
        if row.get("hard_c") and (overwrite or not row.get("hard_c_enhanced")):
            row["hard_c_enhanced"] = str(row["hard_c"])


def reset_unselected_fields(rows: list[dict], label_map: dict[str, dict], label_type_target: str, limit: int | None) -> None:
    if label_type_target == "all":
        return
    selected_rows = rows[:limit] if limit is not None else rows
    for row in selected_rows:
        anchor_label_type, _ = label_from_row(row)
        if not selected_label_type(anchor_label_type, label_type_target):
            row["caption_enhanced"] = str(row.get("caption", ""))

        hard_meta = label_map.get(str(row.get("hard_i_id", "")), {})
        hard_label_type = hard_meta.get("label_type")
        if not selected_label_type(str(hard_label_type) if hard_label_type else None, label_type_target):
            row["hard_c_enhanced"] = str(row.get("hard_c", ""))


def print_smoke(rows: list[dict], tasks: list[Task], outputs: list[str], max_rows: int) -> None:
    by_row: dict[int, list[tuple[Task, str]]] = {}
    for task, output in zip(tasks, outputs):
        by_row.setdefault(task.row_index, []).append((task, output))

    printed = 0
    for row_index in sorted(by_row):
        row = rows[row_index]
        print("=" * 88)
        print(f"image_id: {row.get('image_id')}  hard_i_id: {row.get('hard_i_id')}")
        for task, output in by_row[row_index]:
            print(f"\nfield: {task.output_field}")
            print(f"label_type: {task.label_type}")
            print(f"label: {task.action_label}")
            print(f"old: {task.original_text}")
            print(f"new: {finalize_output(task, output)}")
        printed += 1
        if printed >= max_rows:
            break


def main() -> None:
    args = parse_args()
    if args.resume and not args.write:
        raise SystemExit("--resume requires --write")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be >= 0")
    rows = list(iter_jsonl(args.subset))
    if not rows:
        raise SystemExit(f"No rows found in {args.subset}")

    label_map = load_label_map(args.annotation_dir)
    row_limit = args.limit if args.write else args.limit
    tasks = build_tasks(
        rows,
        label_map,
        overwrite=args.overwrite or args.resume,
        limit=row_limit,
        label_type_target=args.label_type,
    )
    if not args.write:
        tasks = tasks[: max(1, args.smoke * 2)]
    if not tasks:
        if args.write:
            backup_path = args.subset.with_suffix(args.subset.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(args.subset, backup_path)
            initialize_enhanced_fields(rows, overwrite=args.overwrite, limit=args.limit)
            if args.clear_unselected:
                reset_unselected_fields(rows, label_map, args.label_type, args.limit)
            write_jsonl(args.subset, rows)
            print("No caption fields needed LLM enhancement.")
            print(f"backup: {backup_path}")
            print(f"updated: {args.subset}")
            return
        print("No caption fields need enhancement.")
        return

    print(f"subset: {args.subset}")
    print(f"model: {args.model}")
    print(f"backend: {args.backend}")
    print(f"device: {args.device}")
    print(f"rows: {len(rows):,}")
    progress_path = args.progress_path or default_progress_path(args.subset)
    if args.write and not args.resume and progress_path.exists():
        progress_path.unlink()
    completed = load_progress(progress_path) if args.resume else {}
    if args.write:
        backup_path = args.subset.with_suffix(args.subset.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(args.subset, backup_path)
        initialize_enhanced_fields(rows, overwrite=args.overwrite or args.resume, limit=args.limit)
        if args.clear_unselected:
            reset_unselected_fields(rows, label_map, args.label_type, args.limit)
        restored = apply_saved_outputs(rows, tasks, completed)
        pending_tasks = [task for task in tasks if task_key(task) not in completed]
        # Make the normal-copy fields durable before the expensive model starts.
        write_jsonl(args.subset, rows)
    else:
        backup_path = None
        restored = 0
        pending_tasks = tasks

    print(f"tasks: {len(tasks):,}")
    if args.write:
        print(f"resumed tasks: {restored:,}; pending tasks: {len(pending_tasks):,}")
        print(f"progress: {progress_path}")
    print(f"label_type: {args.label_type}")
    print(f"write: {args.write}")

    if not args.write:
        if args.backend == "vllm":
            outputs = generate_vllm(
                args.model,
                [task.prompt for task in tasks],
                args.max_new_tokens,
                args.temperature,
                args.trust_remote_code,
                args.gpu_memory_utilization,
                args.batch_size,
            )
        else:
            tokenizer, model, device = load_model(args.model, args.device, args.trust_remote_code)
            outputs = []
            for start in range(0, len(tasks), args.batch_size):
                batch = tasks[start : start + args.batch_size]
                outputs.extend(
                    generate_batch(
                        tokenizer,
                        model,
                        device,
                        [task.prompt for task in batch],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                )
                done = min(start + args.batch_size, len(tasks))
                print(f"enhanced {done:,}/{len(tasks):,}")
        print_smoke(rows, tasks, outputs, max_rows=args.smoke)
        print("Smoke mode only; no file was modified. Add --write to update the subset JSONL.")
        return

    written_since_checkpoint = 0
    total_done = restored

    def commit(batch: list[Task], raw_outputs: list[str]) -> None:
        nonlocal written_since_checkpoint, total_done
        final_outputs = [finalize_output(task, output) for task, output in zip(batch, raw_outputs)]
        append_progress(progress_path, batch, final_outputs)
        for task, output in zip(batch, final_outputs):
            rows[task.row_index][task.output_field] = output
        written_since_checkpoint += len(batch)
        total_done += len(batch)
        if args.checkpoint_every and written_since_checkpoint >= args.checkpoint_every:
            write_jsonl(args.subset, rows)
            written_since_checkpoint = 0
            print(f"checkpointed {total_done:,}/{len(tasks):,}")

    if args.backend == "vllm":
        for start, raw_outputs in generate_vllm_batches(
            args.model,
            [task.prompt for task in pending_tasks],
            args.max_new_tokens,
            args.temperature,
            args.trust_remote_code,
            args.gpu_memory_utilization,
            args.batch_size,
        ):
            batch = pending_tasks[start : start + len(raw_outputs)]
            commit(batch, raw_outputs)
            print(f"enhanced {total_done:,}/{len(tasks):,}")
    else:
        tokenizer, model, device = load_model(args.model, args.device, args.trust_remote_code)
        for start in range(0, len(pending_tasks), args.batch_size):
            batch = pending_tasks[start : start + args.batch_size]
            raw_outputs = generate_batch(
                tokenizer,
                model,
                device,
                [task.prompt for task in batch],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            commit(batch, raw_outputs)
            print(f"enhanced {total_done:,}/{len(tasks):,}")

    write_jsonl(args.subset, rows)
    print(f"backup: {backup_path}")
    print(f"progress: {progress_path}")
    print(f"updated: {args.subset}")


if __name__ == "__main__":
    main()
