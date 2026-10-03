#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO = Path(__file__).resolve().parents[1]
QA_SCRIPT = REPO / "scripts" / "run_gpt4o_visual_qa.py"
DIFFICULTIES = ("simple", "medium", "difficult")
SPLITS = ("train", "test")


def load_qa_module():
    spec = importlib.util.spec_from_file_location("visual_qa", QA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {QA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QA = load_qa_module()


DIFFICULT_EXTRA_PROMPT = """

Additional note for difficult samples:
- This sample is intentionally difficult. Be more tolerant of density, crossings, long routes, small but legible text, and local clutter.
- Still FAIL if the image is genuinely unusable or the graph structure is ambiguous.
- The final selection still requires overall PASS; do not mark a bad image as PASS.
"""


def sample_key(row: Dict[str, Any]) -> str:
    value = row.get("unified_sample_id") or row.get("sample_id")
    if not value:
        raise ValueError(f"Missing sample id in row: {row}")
    return str(value)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(dataset_root: Path, manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
    path = manifest_path or dataset_root / "manifest.jsonl"
    rows: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        row = dict(row)
        row["unified_sample_id"] = sample_key(row)
        row["image_path"] = str(dataset_root / row["image"])
        rows.append(row)
    return rows


def allocate_targets(counts: Dict[str, int], total_target: int) -> Dict[str, int]:
    total_source = sum(counts.values())
    if total_source <= 0:
        return {difficulty: 0 for difficulty in DIFFICULTIES}
    raw: Dict[str, float] = {
        difficulty: total_target * counts.get(difficulty, 0) / total_source
        for difficulty in DIFFICULTIES
    }
    targets = {difficulty: int(raw[difficulty]) for difficulty in DIFFICULTIES}
    remaining = total_target - sum(targets.values())
    order = sorted(DIFFICULTIES, key=lambda difficulty: (raw[difficulty] - targets[difficulty], difficulty), reverse=True)
    for difficulty in order[:remaining]:
        targets[difficulty] += 1
    return targets


def compute_targets(rows: List[Dict[str, Any]], target_train: int, target_test: int) -> Dict[Tuple[str, str, str], int]:
    source = Counter((row["domain"], row["split"], row["difficulty"]) for row in rows)
    domains = sorted({str(row["domain"]) for row in rows})
    targets: Dict[Tuple[str, str, str], int] = {}
    for domain in domains:
        for split, total_target in (("train", target_train), ("test", target_test)):
            diff_counts = {difficulty: source[(domain, split, difficulty)] for difficulty in DIFFICULTIES}
            for difficulty, count in allocate_targets(diff_counts, total_target).items():
                targets[(domain, split, difficulty)] = count
    return targets


def call_visual_qa(
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    difficulty: str,
    max_tokens: int,
    timeout: int,
) -> str:
    prompt = QA.VISUAL_QA_PROMPT
    if difficulty == "difficult":
        prompt += DIFFICULT_EXTRA_PROMPT
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict but practical visual QA reviewer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": QA.image_data_url(image_path)}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc
    return body["choices"][0]["message"]["content"]


def result_passed(record: Dict[str, Any]) -> bool:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    return record.get("status") == "OK" and result.get("overall") == "PASS"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def summarize_progress(
    targets: Dict[Tuple[str, str, str], int],
    selected: Dict[str, Dict[str, Any]],
    reviewed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    selected_counts = Counter()
    reviewed_counts = Counter()
    failed_counts = Counter()
    for row in selected.values():
        selected_counts[(row["domain"], row["split"], row["difficulty"])] += 1
    for row in reviewed.values():
        key = (row.get("domain"), row.get("split"), row.get("difficulty"))
        reviewed_counts[key] += 1
        if not result_passed(row):
            failed_counts[key] += 1
    groups: Dict[str, Dict[str, int]] = {}
    complete = True
    for key, target in sorted(targets.items()):
        selected_count = selected_counts[key]
        reviewed_count = reviewed_counts[key]
        if selected_count < target:
            complete = False
        groups["/".join(key)] = {
            "target": target,
            "selected": selected_count,
            "reviewed": reviewed_count,
            "failed_or_error": failed_counts[key],
            "remaining": max(0, target - selected_count),
        }
    return {
        "complete": complete,
        "selected_total": len(selected),
        "reviewed_total": len(reviewed),
        "failed_or_error_total": sum(1 for row in reviewed.values() if not result_passed(row)),
        "groups": groups,
    }


def write_outputs(
    dataset_root: Path,
    out_dir: Path,
    selected: Dict[str, Dict[str, Any]],
    reviewed: Dict[str, Dict[str, Any]],
    targets: Dict[Tuple[str, str, str], int],
) -> None:
    selected_rows = [row["manifest_row"] for row in sorted(selected.values(), key=lambda item: sample_key(item["manifest_row"]))]
    rejected_rows = [row for row in sorted(reviewed.values(), key=sample_key) if not result_passed(row)]
    write_jsonl(out_dir / "selected_manifest.jsonl", selected_rows)
    write_jsonl(out_dir / "rejected.jsonl", rejected_rows)
    (out_dir / "screening_state.json").write_text(
        json.dumps(summarize_progress(targets, selected, reviewed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_root = out_dir / "selected_dataset"
    for row in selected_rows:
        image_rel = Path(row["image"])
        ann_rel = Path(row["annotation"])
        link_or_copy(dataset_root / image_rel, selected_root / image_rel)
        link_or_copy(dataset_root / ann_rel, selected_root / ann_rel)
    write_jsonl(selected_root / "manifest.jsonl", selected_rows)


def choose_next_group(
    targets: Dict[Tuple[str, str, str], int],
    selected_counts: Counter,
    available: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    rng: random.Random,
) -> Optional[Tuple[str, str, str]]:
    pending = [
        key
        for key, target in targets.items()
        if selected_counts[key] < target and available.get(key)
    ]
    if not pending:
        return None
    weights = [max(1, targets[key] - selected_counts[key]) for key in pending]
    return rng.choices(pending, weights=weights, k=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("api_config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/final-before-screen-data"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final-before-screen-balanced-screening"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/gpt4o_visual_qa/final_before_screen_balanced_cache"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--target-train", type=int, default=3000)
    parser.add_argument("--target-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=0, help="Stop after this many new reviews. Use 0 for unlimited.")
    parser.add_argument("--max-api-errors", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.dataset_root, args.manifest)
    targets = compute_targets(rows, args.target_train, args.target_test)

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["domain"], row["split"], row["difficulty"])].append(row)

    rng = random.Random(args.seed)
    for key, group_rows in groups.items():
        rng.shuffle(group_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "screening_results.jsonl"
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    reviewed: Dict[str, Dict[str, Any]] = {sample_key(row): row for row in read_jsonl(results_path)}
    selected: Dict[str, Dict[str, Any]] = {}
    for record in reviewed.values():
        if result_passed(record):
            manifest_row = record.get("manifest_row")
            if isinstance(manifest_row, dict):
                selected[sample_key(manifest_row)] = record

    selected_counts = Counter()
    for record in selected.values():
        row = record["manifest_row"]
        selected_counts[(row["domain"], row["split"], row["difficulty"])] += 1

    available: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for key, group_rows in groups.items():
        available[key] = [row for row in group_rows if sample_key(row) not in reviewed]

    target_report = {
        "/".join(key): {
            "target": target,
            "source_candidates": len(groups.get(key, [])),
        }
        for key, target in sorted(targets.items())
    }
    print(json.dumps({"targets": target_report}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    cfg = QA.parse_simple_yaml(args.config)
    api_cfg = cfg.get("api") or cfg
    api_key = api_cfg.get("key") or cfg.get("key")
    base_url = api_cfg.get("base") or cfg.get("base")
    if not api_key or not base_url:
        raise ValueError("Could not find api.key and api.base in config")

    new_reviews = 0
    transient_error_samples: set[str] = set()
    consecutive_api_errors = 0
    api_error_path = args.output_dir / "api_errors.jsonl"
    with results_path.open("a", encoding="utf-8") as out, api_error_path.open("a", encoding="utf-8") as api_error_out:
        while True:
            filtered_available = {
                group_key: [row for row in group_rows if sample_key(row) not in transient_error_samples]
                for group_key, group_rows in available.items()
            }
            key = choose_next_group(targets, selected_counts, filtered_available, rng)
            if key is None:
                break
            domain, split, difficulty = key
            while available[key]:
                sample = available[key].pop()
                if sample_key(sample) not in transient_error_samples:
                    break
            else:
                continue
            sample_id = sample_key(sample)
            image_path = Path(sample["image_path"])
            image_hash = QA.image_sha256(image_path)
            cache_path = args.cache_dir / f"{sample_id}.json"
            record: Dict[str, Any]
            used_cache = False

            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("image_sha256") == image_hash:
                    record = cached
                    used_cache = True
                    print(f"[cache] {sample_id} {domain}/{split}/{difficulty}", flush=True)
                else:
                    cache_path.unlink()
                    record = {}
            else:
                record = {}

            if not record:
                print(
                    f"[review] {sample_id} {domain}/{split}/{difficulty} "
                    f"need={targets[key] - selected_counts[key]}",
                    flush=True,
                )
                raw_text = ""
                parse_error = None
                warnings: List[str] = []
                try:
                    raw_text = call_visual_qa(
                        str(base_url),
                        str(api_key),
                        args.model,
                        image_path,
                        difficulty,
                        args.max_tokens,
                        args.timeout,
                    )
                    (raw_dir / f"{sample_id}.raw.txt").write_text(raw_text, encoding="utf-8")
                    obj, parse_error = QA.extract_json_object(raw_text)
                    if obj is None:
                        status = "PARSE_ERROR"
                        result = {
                            "overall": "FAIL",
                            "checks": {name: "FAIL" for name in QA.CHECK_NAMES},
                            "failure_reasons": [str(parse_error)],
                            "quality_score": 1,
                        }
                    else:
                        status = "OK"
                        result, warnings = QA.normalize_result(obj)
                except Exception as exc:
                    parse_error = f"api_error:{exc}"
                    error_record = {
                        "unified_sample_id": sample_id,
                        "sample_id": sample.get("sample_id", sample_id),
                        "source_sample_id": sample.get("source_sample_id"),
                        "domain": domain,
                        "difficulty": difficulty,
                        "split": split,
                        "image": sample.get("image"),
                        "annotation": sample.get("annotation"),
                        "image_sha256": image_hash,
                        "manifest_row": {k: v for k, v in sample.items() if k != "image_path"},
                        "status": "API_ERROR",
                        "parse_error": parse_error,
                    }
                    api_error_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    api_error_out.flush()
                    transient_error_samples.add(sample_id)
                    consecutive_api_errors += 1
                    print(f"  API_ERROR {parse_error}", flush=True)
                    if consecutive_api_errors >= args.max_api_errors:
                        raise RuntimeError(f"Stopped after {consecutive_api_errors} consecutive API errors")
                    continue

                record = {
                    "unified_sample_id": sample_id,
                    "sample_id": sample.get("sample_id", sample_id),
                    "source_sample_id": sample.get("source_sample_id"),
                    "domain": domain,
                    "difficulty": difficulty,
                    "split": split,
                    "image": sample.get("image"),
                    "annotation": sample.get("annotation"),
                    "image_sha256": image_hash,
                    "manifest_row": {k: v for k, v in sample.items() if k != "image_path"},
                    "result": result,
                    "status": status,
                    "parse_error": parse_error,
                    "normalization_warnings": warnings,
                }
                cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                new_reviews += 1
                consecutive_api_errors = 0

            reviewed[sample_id] = record
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if result_passed(record):
                selected[sample_id] = record
                selected_counts[key] += 1
                print(f"  PASS selected={selected_counts[key]}/{targets[key]}", flush=True)
            else:
                result = record.get("result") or {}
                reasons = "; ".join(result.get("failure_reasons") or []) or record.get("parse_error") or "-"
                print(f"  FAIL/ERROR reasons={reasons}", flush=True)

            if args.report_every and len(reviewed) % args.report_every == 0:
                summary = summarize_progress(targets, selected, reviewed)
                print(json.dumps({k: summary[k] for k in ("selected_total", "reviewed_total", "failed_or_error_total", "complete")}, ensure_ascii=False), flush=True)
                write_outputs(args.dataset_root, args.output_dir, selected, reviewed, targets)
            if args.max_reviews and new_reviews >= args.max_reviews:
                break
            if args.sleep and not used_cache:
                time.sleep(args.sleep)

    write_outputs(args.dataset_root, args.output_dir, selected, reviewed, targets)
    summary = summarize_progress(targets, selected, reviewed)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
