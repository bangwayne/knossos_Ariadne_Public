#!/usr/bin/env python3


from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VISUAL_QA_PROMPT = """You are a visual quality-control reviewer for synthetic topology diagrams. Judge only whether the image is visually usable.

Important principle:
- These diagrams are intended to include different difficulty levels. Some images may be visually challenging, dense, or require careful tracing.
- Do NOT fail an image just because it is difficult, has many nodes/edges, has several crossings, uses small but still readable labels, or contains mild local clutter.
- PASS images that are visible and acceptable for a human reviewer to read with effort.
- FAIL only when the diagram is genuinely unusable: corrupted, blank, severely cropped, unreadable text, indistinguishable nodes, hidden/broken edges, blocked arrowheads, unreadable legend, or chaotic overlaps that make the graph ambiguous.

1. image_clear
PASS if the image is not corrupted, blank, severely blurred, and the diagram is visible.
PASS if the image has minor compression artifacts, slight blur, light noise, or a small amount of margin/cropping that does not affect reading the diagram.
FAIL only if broken, blank, badly cropped, extremely blurry, or visually unusable.

2. text_readable
PASS if important node labels and legend text are readable enough to identify the graph.
PASS if a few labels are small or slightly crowded but still decipherable with effort.
PASS if a small number of non-critical labels are harder to read, provided the main graph structure and most important labels remain understandable.
FAIL only if important text is too small, overlapped, clipped, or unreadable enough that nodes/legend entries cannot be identified.

3. node_visible
PASS if nodes/icons/blocks are visually distinguishable and not severely overlapped.
PASS if a few nodes/icons/blocks are close together, partially touched, or mildly cluttered, as long as they can still be separated by a human reviewer.
FAIL only if important nodes overlap so much that they cannot be separated.

4. edge_visible
PASS if edges/links/arrows are visible enough to trace with reasonable effort.
PASS if dense or difficult diagrams contain crossings, long edges, or some local congestion, as long as the actual connections remain recoverable.
PASS if a small number of edges are harder to follow, provided most connections and the overall topology remain recoverable.
FAIL only if many edges are hidden, broken, too faint, or impossible to follow.
For food-web or ecological diagrams, natural long diagonal links and a small number of crossings are acceptable if a human can still trace each connection.

5. arrow_clear
PASS if arrowheads, when present, are visible and not mostly hidden.
PASS if there are no arrowheads and links are intended to be undirected/no-arrow.
PASS if a few arrowheads are small but their direction is still reasonably inferable.
PASS if a small number of arrowheads are partly occluded or subtle, provided the intended direction can usually be inferred from the line and nearby context.
FAIL only if arrowheads are frequently blocked, clipped, or visually ambiguous enough that directions cannot be determined.

6. legend_clear
PASS if there is no legend.
PASS if the legend exists and its labels, colors/styles, and line samples are readable and distinguishable.
FAIL if the legend is unreadable, clipped, overlapped, or style distinctions are unclear.

7. overlap_acceptable
PASS if node-node, node-edge, text-edge, and edge-edge overlaps are acceptable and do not prevent reading the graph.
PASS if some local overlap exists because the graph is difficult, provided the image is not a chaotic tangle and endpoints/labels/arrowheads remain mostly interpretable.
FAIL if overlaps are severe enough that a human reviewer would be confused about the graph structure.
Do not fail only because edges cross; fail only when overlaps make endpoints, arrowheads, labels, or actual connections ambiguous.

Return ONLY valid JSON:

{
  "overall": "PASS" or "FAIL",
  "checks": {
    "image_clear": "PASS" or "FAIL",
    "text_readable": "PASS" or "FAIL",
    "node_visible": "PASS" or "FAIL",
    "edge_visible": "PASS" or "FAIL",
    "arrow_clear": "PASS" or "FAIL",
    "legend_clear": "PASS" or "FAIL",
    "overlap_acceptable": "PASS" or "FAIL"
  },
  "failure_reasons": ["short reason"],
  "quality_score": 1-5
}

Rules:
- overall is PASS only if every check is PASS.
- If PASS, failure_reasons must be empty and quality_score must be 3-5.
- If FAIL, quality_score must be 1-2."""


CHECK_NAMES = [
    "image_clear",
    "text_readable",
    "node_visible",
    "edge_visible",
    "arrow_clear",
    "legend_clear",
    "overlap_acceptable",
]


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        value = value.strip().strip("\"'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value:
            parent[key] = value
        else:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    start = text.find("{")
    if start < 0:
        return None, "missing_json_object"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : idx + 1])
                except json.JSONDecodeError as exc:
                    return None, f"json_decode_error:{exc.msg}"
                if not isinstance(obj, dict):
                    return None, "json_not_object"
                return obj, None
    return None, "truncated_json_object"


def normalize_result(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    checks = obj.get("checks") if isinstance(obj.get("checks"), dict) else {}
    normalized_checks: Dict[str, str] = {}
    for name in CHECK_NAMES:
        value = str(checks.get(name, "")).upper()
        if value not in {"PASS", "FAIL"}:
            warnings.append(f"invalid_check:{name}")
            value = "FAIL"
        normalized_checks[name] = value

    overall = str(obj.get("overall", "")).upper()
    if overall not in {"PASS", "FAIL"}:
        warnings.append("invalid_overall")
        overall = "PASS" if all(value == "PASS" for value in normalized_checks.values()) else "FAIL"

    try:
        quality_score = int(obj.get("quality_score"))
    except (TypeError, ValueError):
        warnings.append("invalid_quality_score")
        quality_score = 3 if overall == "PASS" else 2
    quality_score = max(1, min(5, quality_score))

    if overall == "PASS" and any(value != "PASS" for value in normalized_checks.values()):
        warnings.append("overall_pass_but_check_failed")
        overall = "FAIL"
    if overall == "PASS" and quality_score < 3:
        warnings.append("overall_pass_but_low_score")
        overall = "FAIL"
    if overall == "FAIL" and quality_score > 2:
        warnings.append("overall_fail_but_high_score")
        quality_score = 2

    reasons = obj.get("failure_reasons")
    if not isinstance(reasons, list):
        reasons = []
        if overall == "FAIL":
            warnings.append("missing_failure_reasons")
    reasons = [str(reason)[:240] for reason in reasons]
    if overall == "PASS" and reasons:
        warnings.append("pass_with_failure_reasons")
        reasons = []

    return (
        {
            "overall": overall,
            "checks": normalized_checks,
            "failure_reasons": reasons,
            "quality_score": quality_score,
        },
        warnings,
    )


def call_nec_chat(
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict visual QA reviewer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_QA_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
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


def sample_key(row: Dict[str, Any]) -> str:
    value = row.get("unified_sample_id") or row.get("sample_id")
    if not value:
        raise ValueError(f"Manifest row is missing unified_sample_id/sample_id: {row}")
    return str(value)


def load_manifest(
    path: Path,
    dataset_root: Path,
    domains: Optional[set[str]] = None,
    splits: Optional[set[str]] = None,
    difficulties: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["unified_sample_id"] = sample_key(row)
            if domains and str(row.get("domain")) not in domains:
                continue
            if splits and str(row.get("split")) not in splits:
                continue
            if difficulties and str(row.get("difficulty")) not in difficulties:
                continue
            image_path = dataset_root / row["image"]
            rows.append({**row, "image_path": image_path})
    return rows


def stratified_sample(rows: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return sorted(rows, key=lambda item: sample_key(item))

    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("domain", "unknown")), str(row.get("difficulty", "unknown")))].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected: List[Dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) >= limit:
                    break
    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = {key: str(value) if isinstance(value, Path) else value for key, value in row.items()}
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{sample_id}.json"


def load_cached_record(cache_dir: Path, sample_id: str) -> Optional[Dict[str, Any]]:
    path = cache_path(cache_dir, sample_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cached_record(cache_dir: Path, record: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, str(record["unified_sample_id"])).write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(records: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    by_domain = Counter()
    by_difficulty = Counter()
    pass_by_domain = Counter()
    pass_by_difficulty = Counter()
    check_failures = Counter()
    scores = Counter()
    parse_errors = Counter()
    api_errors = Counter()
    evaluated = 0

    for record in records:
        if record.get("status") == "API_ERROR":
            api_errors[str(record.get("parse_error"))] += 1
            continue
        evaluated += 1
        domain = record.get("domain", "unknown")
        difficulty = record.get("difficulty", "unknown")
        by_domain[domain] += 1
        by_difficulty[difficulty] += 1
        result = record.get("result") or {}
        if record.get("parse_error"):
            parse_errors[str(record["parse_error"])] += 1
        if result.get("overall") == "PASS":
            pass_by_domain[domain] += 1
            pass_by_difficulty[difficulty] += 1
        for check, value in (result.get("checks") or {}).items():
            if value == "FAIL":
                check_failures[check] += 1
        if "quality_score" in result:
            scores[str(result["quality_score"])] += 1

    total = evaluated
    passed = sum(1 for record in records if (record.get("result") or {}).get("overall") == "PASS")
    return {
        "model": model,
        "num_samples": len(records),
        "evaluated_samples": evaluated,
        "api_error_samples": len(records) - evaluated,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "by_domain": {
            domain: {
                "samples": count,
                "passed": pass_by_domain[domain],
                "pass_rate": pass_by_domain[domain] / count if count else 0.0,
            }
            for domain, count in sorted(by_domain.items())
        },
        "by_difficulty": {
            difficulty: {
                "samples": count,
                "passed": pass_by_difficulty[difficulty],
                "pass_rate": pass_by_difficulty[difficulty] / count if count else 0.0,
            }
            for difficulty, count in sorted(by_difficulty.items())
        },
        "check_failures": dict(check_failures),
        "quality_scores": dict(scores),
        "parse_errors": dict(parse_errors),
        "api_errors": dict(api_errors),
    }


def print_running_summary(records: List[Dict[str, Any]]) -> None:
    evaluated = [record for record in records if record.get("status") != "API_ERROR"]
    passed = [record for record in evaluated if (record.get("result") or {}).get("overall") == "PASS"]
    foodweb = [record for record in evaluated if record.get("domain") == "foodweb"]
    foodweb_passed = [record for record in foodweb if (record.get("result") or {}).get("overall") == "PASS"]
    print(
        "  running_summary "
        f"evaluated={len(evaluated)} pass_rate={(len(passed) / len(evaluated)) if evaluated else 0:.3f} "
        f"foodweb={len(foodweb)} foodweb_pass_rate={(len(foodweb_passed) / len(foodweb)) if foodweb else 0:.3f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("api_config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/Knossos-36K-v1"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpt4o_visual_qa/pilot_100"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/gpt4o_visual_qa/cache"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to review. Use 0 for all matching samples.")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--domain", action="append", default=None, help="Optional domain filter. Can be repeated.")
    parser.add_argument("--split", action="append", default=None, help="Optional split filter. Can be repeated.")
    parser.add_argument("--difficulty", action="append", default=None, help="Optional difficulty filter. Can be repeated.")
    args = parser.parse_args()

    cfg = parse_simple_yaml(args.config)
    api_cfg = cfg.get("api") or cfg
    api_key = api_cfg.get("key") or cfg.get("key")
    base_url = api_cfg.get("base") or cfg.get("base")
    if not api_key or not base_url:
        raise ValueError("Could not find api.key and api.base in config")

    dataset_root = args.dataset_root
    manifest_path = args.manifest or dataset_root / "manifest.jsonl"
    samples = stratified_sample(
        load_manifest(
            manifest_path,
            dataset_root,
            domains=set(args.domain) if args.domain else None,
            splits=set(args.split) if args.split else None,
            difficulties=set(args.difficulty) if args.difficulty else None,
        ),
        args.limit,
        args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    report_path = args.output_dir / "qa_report.json"
    sample_manifest_path = args.output_dir / "sample_manifest.jsonl"
    write_jsonl(sample_manifest_path, samples)

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["unified_sample_id"])] = row

    records: List[Dict[str, Any]] = []
    seen_records: set[str] = set()
    if existing:
        for record in existing.values():
            records.append(record)
            seen_records.add(str(record["unified_sample_id"]))

    with results_path.open("a" if existing else "w", encoding="utf-8") as out:
        for idx, sample in enumerate(samples, start=1):
            sample_id = sample_key(sample)
            if sample_id in existing:
                print(f"[{idx}/{len(samples)}] {sample_id} cached", flush=True)
                continue
            sample_hash = image_sha256(Path(sample["image_path"]))
            cached = load_cached_record(args.cache_dir, sample_id)
            if cached is not None:
                if cached.get("image_sha256") == sample_hash:
                    print(f"[{idx}/{len(samples)}] {sample_id} global_cache", flush=True)
                    out.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    out.flush()
                    if sample_id not in seen_records:
                        records.append(cached)
                        seen_records.add(sample_id)
                    if args.report_every and len(records) % args.report_every == 0:
                        print_running_summary(records)
                    continue
                print(f"[{idx}/{len(samples)}] {sample_id} cache_stale", flush=True)

            print(
                f"[{idx}/{len(samples)}] {sample_id} {sample['domain']} {sample['difficulty']}",
                flush=True,
            )
            raw_text = ""
            parse_error = None
            result: Dict[str, Any]
            warnings: List[str] = []
            try:
                raw_text = call_nec_chat(
                    str(base_url),
                    str(api_key),
                    args.model,
                    Path(sample["image_path"]),
                    args.max_tokens,
                    args.timeout,
                )
                raw_path = raw_dir / f"{sample_id}.raw.txt"
                raw_path.write_text(raw_text, encoding="utf-8")
                obj, parse_error = extract_json_object(raw_text)
                if obj is None:
                    status = "PARSE_ERROR"
                    result = {
                        "overall": "FAIL",
                        "checks": {name: "FAIL" for name in CHECK_NAMES},
                        "failure_reasons": [str(parse_error)],
                        "quality_score": 1,
                    }
                else:
                    status = "OK"
                    result, warnings = normalize_result(obj)
            except Exception as exc:  
                parse_error = f"api_error:{exc}"
                status = "API_ERROR"
                result = {
                    "overall": "ERROR",
                    "checks": {},
                    "failure_reasons": [str(parse_error)[:240]],
                    "quality_score": None,
                }

            record = {
                "unified_sample_id": sample_id,
                "source_sample_id": sample.get("source_sample_id"),
                "domain": sample.get("domain"),
                "difficulty": sample.get("difficulty"),
                "split": sample.get("split"),
                "image": sample.get("image"),
                "image_sha256": sample_hash,
                "annotation": sample.get("annotation"),
                "result": result,
                "status": status,
                "parse_error": parse_error,
                "normalization_warnings": warnings,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            write_cached_record(args.cache_dir, record)
            records.append(record)
            seen_records.add(sample_id)
            print(
                f"  {result['overall']} score={result['quality_score']} "
                f"reasons={'; '.join(result['failure_reasons']) if result['failure_reasons'] else '-'}",
                flush=True,
            )
            if args.report_every and len(records) % args.report_every == 0:
                print_running_summary(records)
            if args.sleep:
                time.sleep(args.sleep)

    report = {
        "summary": summarize(records, args.model),
        "prompt": VISUAL_QA_PROMPT,
        "sample_manifest": str(sample_manifest_path),
        "results_jsonl": str(results_path),
        "raw_dir": str(raw_dir),
        "cache_dir": str(args.cache_dir),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
