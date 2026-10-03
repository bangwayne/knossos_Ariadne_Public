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
from typing import Any, Dict, Iterable, List, Tuple


REPO = Path(__file__).resolve().parents[1]
QA_SCRIPT = REPO / "scripts" / "run_gpt4o_visual_qa.py"


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


FOODWEB_TARGETS = {
    ("train", "difficult"): 945,
    ("train", "medium"): 1529,
    ("train", "simple"): 526,
    ("test", "difficult"): 105,
    ("test", "medium"): 71,
    ("test", "simple"): 24,
}


def sample_key(row: Dict[str, Any]) -> str:
    value = row.get("unified_sample_id") or row.get("sample_id")
    if not value:
        raise ValueError(f"Missing sample id: {row}")
    return str(value)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def result_passed(record: Dict[str, Any]) -> bool:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    return record.get("status") == "OK" and result.get("overall") == "PASS"


def call_visual_qa(base_url: str, api_key: str, model: str, image_path: Path, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict but practical visual QA reviewer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QA.VISUAL_QA_PROMPT + DIFFICULT_EXTRA_PROMPT},
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


def screen_foodweb_difficult(
    dataset_root: Path,
    source_rows: List[Dict[str, Any]],
    existing_records: Dict[str, Dict[str, Any]],
    results_path: Path,
    raw_dir: Path,
    cache_dir: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    api_cfg = cfg.get("api") or cfg
    api_key = api_cfg.get("key") or cfg.get("key")
    base_url = api_cfg.get("base") or cfg.get("base")
    if not api_key or not base_url:
        raise ValueError("Could not find api.key and api.base in config")

    records = dict(existing_records)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as out:
        for row in source_rows:
            key = sample_key(row)
            if key in records and result_passed(records[key]):
                continue
            image_path = dataset_root / row["image"]
            image_hash = QA.image_sha256(image_path)
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("image_sha256") == image_hash:
                    records[key] = cached
                    out.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[cache] {key}", flush=True)
                    continue
            print(f"[review] {key} foodweb/{row['split']}/difficult", flush=True)
            raw_text = ""
            parse_error = None
            warnings: List[str] = []
            try:
                raw_text = call_visual_qa(
                    str(base_url),
                    str(api_key),
                    args.model,
                    image_path,
                    args.max_tokens,
                    args.timeout,
                )
                (raw_dir / f"{key}.raw.txt").write_text(raw_text, encoding="utf-8")
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
                status = "API_ERROR"
                parse_error = f"api_error:{exc}"
                result = {
                    "overall": "ERROR",
                    "checks": {},
                    "failure_reasons": [str(parse_error)[:240]],
                    "quality_score": None,
                }
            record = {
                "unified_sample_id": key,
                "sample_id": row.get("sample_id", key),
                "source_sample_id": row.get("source_sample_id"),
                "domain": "foodweb",
                "difficulty": "difficult",
                "split": row.get("split"),
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "image_sha256": image_hash,
                "manifest_row": dict(row),
                "result": result,
                "status": status,
                "parse_error": parse_error,
                "normalization_warnings": warnings,
            }
            records[key] = record
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {result.get('overall')} score={result.get('quality_score')}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/final-before-screen-data"))
    parser.add_argument("--screening-dir", type=Path, default=Path("outputs/final-before-screen-balanced-screening"))
    parser.add_argument("--out", type=Path, default=Path("outputs/final-screened-balanced-foodweb-max-difficult"))
    parser.add_argument("--config", type=Path, default=Path("api_config.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/gpt4o_visual_qa/foodweb_max_difficult_cache"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected_rows = read_jsonl(args.screening_dir / "selected_manifest.jsonl")
    source_rows = read_jsonl(args.dataset_root / "manifest.jsonl")
    screening_records = {sample_key(row): row for row in read_jsonl(args.screening_dir / "screening_results.jsonl")}
    foodweb_source = [
        row for row in source_rows
        if row.get("domain") == "foodweb" and row.get("difficulty") == "difficult"
    ]

    target_report = {
        "foodweb_targets": {f"{split}/{difficulty}": target for (split, difficulty), target in FOODWEB_TARGETS.items()},
        "foodweb_difficult_source": len(foodweb_source),
        "current_foodweb_selected": {
            f"{key[0]}/{key[1]}": value
            for key, value in sorted(Counter((row["split"], row["difficulty"]) for row in selected_rows if row.get("domain") == "foodweb").items())
        },
    }
    print(json.dumps(target_report, ensure_ascii=False, indent=2, default=str), flush=True)
    if args.dry_run:
        return 0
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass --overwrite")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = QA.parse_simple_yaml(args.config)
    screened_records = screen_foodweb_difficult(
        args.dataset_root,
        foodweb_source,
        screening_records,
        args.out / "foodweb_extra_screening_results.jsonl",
        args.out / "raw",
        args.cache_dir,
        cfg,
        args,
    )

    pass_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row.get("domain") == "foodweb":
            pass_by_group[(row["split"], row["difficulty"])].append(row)
    for row in foodweb_source:
        key = sample_key(row)
        record = screened_records.get(key)
        if record and result_passed(record):
            pass_by_group[(row["split"], row["difficulty"])].append(row)

    deduped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for group, rows in pass_by_group.items():
        for row in rows:
            deduped[group][sample_key(row)] = row

    new_rows: List[Dict[str, Any]] = [row for row in selected_rows if row.get("domain") != "foodweb"]
    missing: Dict[str, int] = {}
    for group, target in FOODWEB_TARGETS.items():
        rows = list(deduped[group].values())
        rows.sort(key=sample_key)
        rng.shuffle(rows)
        if len(rows) < target:
            missing[f"{group[0]}/{group[1]}"] = target - len(rows)
        chosen = sorted(rows[:target], key=sample_key)
        new_rows.extend(chosen)

    if missing:
        print(json.dumps({"missing_targets": missing}, ensure_ascii=False, indent=2), flush=True)
        return 2

    new_rows = sorted(new_rows, key=lambda row: (row["domain"], row["split"], row["difficulty"], sample_key(row)))
    selected_dataset = args.out / "selected_dataset"
    for row in new_rows:
        link_or_copy(args.dataset_root / row["image"], selected_dataset / row["image"])
        link_or_copy(args.dataset_root / row["annotation"], selected_dataset / row["annotation"])
    write_jsonl(args.out / "selected_manifest.jsonl", new_rows)
    write_jsonl(selected_dataset / "manifest.jsonl", new_rows)

    summary = {
        "total": len(new_rows),
        "split": dict(sorted(Counter(row["split"] for row in new_rows).items())),
        "domain_split": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter((row["domain"], row["split"]) for row in new_rows).items())},
        "domain_difficulty": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter((row["domain"], row["difficulty"]) for row in new_rows).items())},
        "foodweb_targets": {f"{split}/{difficulty}": target for (split, difficulty), target in FOODWEB_TARGETS.items()},
        "note": "FoodWeb difficult is maximized to all 1050 available source samples; exact 1120 is impossible without generating more FoodWeb difficult samples.",
    }
    (args.out / "adjustment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
