from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable 448x448 subset package without modifying source images."
    )
    parser.add_argument("--subset", required=True, help="Subset name or JSONL path.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--subsets-dir", type=Path, default=Path("data/subsets_v2"))
    parser.add_argument("--image-root", type=Path, default=Path("data/train_webp"))
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("data/annotation/train_processed"),
        help="Full processed annotation used to include labels for hard_i_id targets.",
    )
    parser.add_argument(
        "--video-splits",
        type=Path,
        default=None,
        help="Optional video_splits.json. Required with --include-eval-splits.",
    )
    parser.add_argument(
        "--include-eval-splits",
        default="",
        help="Comma-separated frozen splits to package as evaluation galleries, e.g. dev,report.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing compact dev_eval.jsonl/report_eval.jsonl suites.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("exports"))
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--require-source-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=max(4, min(32, os.cpu_count() or 8)))
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--archive", choices=("none", "zst"), default="zst")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def annotation_index(path: Path) -> int:
    """Sort attr_<integer>.json files numerically rather than lexicographically."""
    try:
        return int(path.stem.removeprefix("attr_"))
    except ValueError as exc:
        raise ValueError(f"Expected an attr_<integer>.json annotation file, got {path}") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def subset_path(value: str, subsets_dir: Path) -> Path:
    path = Path(value)
    if path.suffix == ".jsonl":
        return path
    return subsets_dir / f"{value}.jsonl"


def subset_name(value: str) -> str:
    return Path(value).stem


def source_image_path(value: str, data_root: Path, image_root: Path) -> Path:
    text = str(value).replace("\\", "/")
    if text.startswith("data/train_webp/"):
        return data_root.parent / text
    if text.startswith("train_webp/"):
        return data_root / text
    if text.startswith("imgs_"):
        return image_root / text
    if text.startswith("train/"):
        return image_root / Path(text.removeprefix("train/")).with_suffix(".webp")
    path = Path(text)
    if path.is_absolute():
        return path
    return data_root / path


def output_image_rel(value: str) -> Path:
    text = str(value).replace("\\", "/")
    if text.startswith("data/"):
        text = text[len("data/") :]
    if text.startswith("train_webp/"):
        return Path(text)
    if text.startswith("imgs_"):
        return Path("train_webp") / text
    if text.startswith("train/"):
        return Path("train_webp") / Path(text.removeprefix("train/")).with_suffix(".webp")
    parts = Path(text).parts
    if "train_webp" in parts:
        return Path(*parts[parts.index("train_webp") :])
    return Path("train_webp") / Path(text).name


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def valid_output(path: Path, size: int) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (size, size)
    except Exception:
        return False


def resize_one(
    src: Path,
    dst: Path,
    size: int,
    quality: int,
    resume: bool,
    require_source_size: int | None,
) -> tuple[str, str | None]:
    if resume and valid_output(dst, size):
        return "skipped", None
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp.webp")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            if require_source_size and image.size != (require_source_size, require_source_size):
                return (
                    "failed",
                    f"{src}: expected source {require_source_size}x{require_source_size}, got {image.size}",
                )
            rgb = image.convert("RGB")
            resized = rgb.resize((size, size), Image.Resampling.BICUBIC)
            resized.save(tmp, format="WEBP", quality=quality, method=6)
        tmp.replace(dst)
        return "resized", None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return "failed", f"{src}: {type(exc).__name__}: {exc}"


def copy_metadata(
    annotation_rows: dict[str, dict],
    manifest_path: Path,
    subsets_dir: Path,
    data_root: Path,
    export_data: Path,
    name: str,
    video_splits: Path | None,
    eval_paths: list[Path],
) -> dict:
    subset_dst = export_data / "subsets_v2"
    subset_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, subset_dst / manifest_path.name)
    for filename in (
        f"hard_edges_{name.removeprefix('train_')}.jsonl",
        "subset_summary.json",
    ):
        src = subsets_dir / filename
        if src.exists():
            shutil.copy2(src, subset_dst / filename)

    annotation_dst = export_data / "annotation"
    source_caption = data_root / "annotation" / "source_caption.json"
    if source_caption.exists():
        annotation_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_caption, annotation_dst / source_caption.name)

    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for image_id, row in annotation_rows.items():
        chunk = image_id.split("_", 1)[0]
        by_chunk[chunk].append(row)
    counts = {}
    for chunk, chunk_rows in sorted(
        by_chunk.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
    ):
        filename = f"attr_{chunk}.json"
        counts[filename] = write_jsonl(annotation_dst / "train_processed" / filename, chunk_rows)
    if video_splits is not None:
        split_dst = export_data / "splits"
        split_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_splits, split_dst / "video_splits.json")
    if eval_paths:
        eval_dst = export_data / "eval"
        eval_dst.mkdir(parents=True, exist_ok=True)
        for path in eval_paths:
            shutil.copy2(path, eval_dst / path.name)
    return counts


def parse_eval_splits(value: str) -> set[str]:
    names = {item.strip() for item in value.split(",") if item.strip()}
    unknown = names - {"dev", "report"}
    if unknown:
        raise ValueError(f"--include-eval-splits only accepts dev/report, got {sorted(unknown)}")
    return names


def collect_eval_records(
    eval_dir: Path | None,
    eval_splits: set[str],
) -> tuple[list[dict], dict[str, dict[str, int]], list[Path]]:
    if not eval_splits:
        return [], {}, []
    if eval_dir is None:
        raise ValueError("--eval-dir is required when --include-eval-splits is set")
    records, counts, paths, seen_ids = [], {}, [], set()
    for split in sorted(eval_splits):
        path = eval_dir / f"{split}_eval.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen {split} evaluation suite: {path}")
        suite = list(iter_jsonl(path))
        if not suite:
            raise ValueError(f"Evaluation suite is empty: {path}")
        query_count = 0
        for record in suite:
            image_id = str(record.get("image_id") or "")
            if not image_id or record.get("split") != split:
                raise ValueError(f"Invalid record in {path}: {record}")
            if image_id in seen_ids:
                raise ValueError(f"Evaluation image appears in multiple suites: {image_id}")
            seen_ids.add(image_id)
            query_count += bool(record.get("is_query"))
        if not query_count:
            raise ValueError(f"Evaluation suite has no queries: {path}")
        records.extend(suite)
        counts[split] = {"gallery_rows": len(suite), "query_rows": query_count}
        paths.append(path)
    return records, counts, paths


def selected_annotation_rows(annotation_dir: Path, needed_ids: set[str]) -> dict[str, dict]:
    rows = {}
    for path in sorted(annotation_dir.glob("attr_*.json"), key=annotation_index):
        for row in iter_jsonl(path):
            image_id = str(row.get("image_id"))
            if image_id in needed_ids:
                rows[image_id] = row
    missing = sorted(needed_ids - set(rows))
    if missing:
        raise RuntimeError(
            f"{len(missing):,} requested IDs are missing from {annotation_dir}. First: {missing[:10]}"
        )
    return rows


def create_shards(
    export_data: Path,
    shard_dir: Path,
    rel_paths: list[Path],
    shard_size: int,
) -> list[dict]:
    if shard_size <= 0:
        return []
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, len(rel_paths), shard_size):
        chunk = rel_paths[start : start + shard_size]
        shard_index = start // shard_size
        path = shard_dir / f"train_webp_{shard_index:05d}.tar"
        tmp = path.with_suffix(".tar.tmp")
        with tarfile.open(tmp, "w") as tar:
            for rel in chunk:
                src = export_data / rel
                tar.add(src, arcname=str(Path("data") / rel), recursive=False)
        tmp.replace(path)
        records.append(
            {
                "path": str(path),
                "images": len(chunk),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        print(f"wrote shard {shard_index + 1:,}/{(len(rel_paths) + shard_size - 1) // shard_size:,}")
    return records


def create_archive(export_dir: Path) -> Path:
    archive = export_dir.with_suffix(".tar.zst")
    tmp = archive.with_name(archive.name + ".tmp")
    subprocess.run(
        ["tar", "--zstd", "-cf", str(tmp), "-C", str(export_dir.parent), export_dir.name],
        check=True,
    )
    tmp.replace(archive)
    return archive


def main() -> None:
    args = parse_args()
    manifest_path = subset_path(args.subset, args.subsets_dir)
    name = subset_name(args.subset)
    export_dir = args.output_root / f"{name}_448_data"
    export_data = export_dir / "data"

    if export_dir.exists() and args.overwrite:
        shutil.rmtree(export_dir)
    elif export_dir.exists() and not args.resume:
        raise SystemExit(f"{export_dir} exists. Use --resume or --overwrite.")
    export_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(manifest_path))
    eval_splits = parse_eval_splits(args.include_eval_splits)
    eval_records, eval_counts, eval_paths = collect_eval_records(args.eval_dir, eval_splits)
    assets: dict[Path, Path] = {}
    missing = []
    needed_ids: set[str] = set()
    for row in rows:
        for field in ("image_webp", "hard_image_webp"):
            src = source_image_path(str(row[field]), args.data_root, args.image_root)
            rel = output_image_rel(str(row[field]))
            assets[src] = rel
            if not src.exists():
                missing.append(str(src))
        for field in ("image_id", "hard_i_id"):
            if row.get(field):
                needed_ids.add(str(row[field]))
    needed_ids.update(str(record["image_id"]) for record in eval_records)
    annotation_rows = selected_annotation_rows(args.annotation_dir, needed_ids)
    for record in eval_records:
        row = annotation_rows[str(record["image_id"])]
        image_value = str(row.get("image") or "")
        src = source_image_path(image_value, args.data_root, args.image_root)
        rel = output_image_rel(image_value)
        assets[src] = rel
        if not src.exists():
            missing.append(str(src))
    if missing:
        raise SystemExit(f"{len(missing):,} selected assets are missing. First: {missing[:10]}")

    stats = Counter()
    failures = []
    ordered_assets = sorted(assets.items(), key=lambda item: item[1].as_posix())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                resize_one,
                src,
                export_data / rel,
                args.size,
                args.quality,
                args.resume,
                args.require_source_size,
            ): (src, rel)
            for src, rel in ordered_assets
        }
        for done, future in enumerate(as_completed(futures), 1):
            status, error = future.result()
            stats[status] += 1
            if error:
                failures.append(error)
            if done % args.progress_every == 0 or done == len(futures):
                print(f"images {done:,}/{len(futures):,}: {dict(stats)}")
    if failures:
        raise SystemExit(f"{len(failures):,} resize failures. First: {failures[:10]}")

    annotation_counts = copy_metadata(
        annotation_rows,
        manifest_path,
        args.subsets_dir,
        args.data_root,
        export_data,
        name,
        args.video_splits,
        eval_paths,
    )
    rel_paths = [rel for _, rel in ordered_assets]

    checksum_path = export_dir / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for rel in rel_paths:
            path = export_data / rel
            if not valid_output(path, args.size):
                raise SystemExit(f"Invalid resized output: {path}")
            handle.write(f"{sha256(path)}  data/{rel.as_posix()}\n")

    shard_dir = args.output_root / f"{name}_448_shards"
    if shard_dir.exists() and args.overwrite:
        shutil.rmtree(shard_dir)
    shards = create_shards(export_data, shard_dir, rel_paths, args.shard_size)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subset": name,
        "source_manifest": str(manifest_path),
        "subset_rows": len(rows),
        "eval_rows": eval_counts,
        "unique_assets": len(assets),
        "output_size": [args.size, args.size],
        "webp_quality": args.quality,
        "resize_stats": dict(stats),
        "annotation_files": annotation_counts,
        "checksums": str(checksum_path),
        "shard_size": args.shard_size,
        "shards": shards,
    }
    manifest_path_out = export_dir / "export_manifest.json"
    manifest_path_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = None
    if args.archive == "zst":
        archive = create_archive(export_dir)
        manifest["archive"] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        }
        manifest_path_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"export: {export_dir}")
    print(f"subset rows: {len(rows):,}; eval rows: {eval_counts}; unique 448 assets: {len(assets):,}")
    print(f"shards: {len(shards):,}")
    if archive:
        print(f"archive: {archive}")


if __name__ == "__main__":
    main()
