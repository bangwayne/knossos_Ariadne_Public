#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


TRACE_RE = re.compile(r'<TRACE\s+([^>/]*?)/>', re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def attrs(fragment: str) -> dict[str, str]:
    return dict(ATTR_RE.findall(fragment))


def parse_points(raw: str) -> list[list[float]] | None:
    try:
        value = json.loads(raw)
        if not isinstance(value, list) or len(value) != 5:
            return None
        points = [[float(point[0]), float(point[1])] for point in value]
        if any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in points):
            return None
        return points
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return None


def parse_traces(text: str) -> tuple[dict[str, list[list[float]] | None], dict[str, Any]]:
    traces: dict[str, list[list[float]] | None] = {}
    duplicates = 0
    raw_count = 0
    for match in TRACE_RE.finditer(text):
        raw_count += 1
        item = attrs(match.group(1))
        edge_id = item.get("edge_id", "")
        if not edge_id:
            continue
        if edge_id in traces:
            duplicates += 1
            continue
        traces[edge_id] = parse_points(item.get("points", ""))
    return traces, {
        "has_open": "<TRACES>" in text,
        "has_close": "</TRACES>" in text,
        "raw_trace_count": raw_count,
        "duplicate_edge_ids": duplicates,
    }


def point_box_distance(point: Sequence[float], box: Sequence[float]) -> float:
    if len(box) != 4:
        return math.inf
    x, y = point
    x1, y1, x2, y2 = map(float, box)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def aligned_l2(pred: Sequence[Sequence[float]], gold: Sequence[Sequence[float]]) -> float:
    return sum(math.hypot(px - gx, py - gy) for (px, py), (gx, gy) in zip(pred, gold)) / 5


def chamfer(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    def directed(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]]) -> float:
        return sum(min(math.hypot(px - qx, py - qy) for qx, qy in y) for px, py in x) / len(x)
    return 0.5 * (directed(a, b) + directed(b, a))


def evaluate_one(row: dict[str, Any], text: str) -> dict[str, Any]:
    predicted, parse = parse_traces(text)
    targets = {str(item["edge_id"]): item for item in row["trace_targets"]}
    counts: Counter[str] = Counter()
    counts["groups"] = 1
    counts["gold_traces"] = len(targets)
    counts["pred_traces"] = len(predicted)
    counts["raw_traces"] = int(parse["raw_trace_count"])
    counts["duplicate_ids"] = int(parse["duplicate_edge_ids"])
    counts["unknown_ids"] = sum(edge_id not in targets for edge_id in predicted)
    counts["joined"] = sum(edge_id in targets for edge_id in predicted)
    counts["count_exact"] = int(len(predicted) == len(targets))
    counts["id_set_exact"] = int(set(predicted) == set(targets))
    counts["closed"] = int(parse["has_close"])
    counts["opened"] = int(parse["has_open"])
    for edge_id, target in targets.items():
        points = predicted.get(edge_id)
        if points is None:
            continue
        counts["valid5"] += 1
        start_distance = point_box_distance(points[0], target["source_box"])
        end_distance = point_box_distance(points[-1], target["target_box"])
        counts["start_within_12"] += int(start_distance <= 12)
        counts["end_within_12"] += int(end_distance <= 12)
        counts["both_within_12"] += int(start_distance <= 12 and end_distance <= 12)
        counts["start_within_20"] += int(start_distance <= 20)
        counts["end_within_20"] += int(end_distance <= 20)
        gold = target["points"]
        counts["distance_count"] += 1
        counts["aligned_l2_sum"] += aligned_l2(points, gold)
        counts["chamfer_sum"] += chamfer(points, gold)
    return {
        "sample_id": row["sample_id"],
        "domain": row.get("domain"),
        "difficulty": row.get("difficulty"),
        "degree": len(targets),
        "counts": dict(counts),
        "parse": parse,
    }


def aggregate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for item in items:
        totals.update(item["counts"])
    gold = totals["gold_traces"]
    valid = totals["valid5"]
    distance_count = totals["distance_count"]
    return {
        "groups": len(items),
        "gold_traces": gold,
        "pred_traces": totals["pred_traces"],
        "trace_recall": totals["joined"] / gold if gold else 0.0,
        "valid5_per_gold": valid / gold if gold else 0.0,
        "valid5_per_joined": valid / totals["joined"] if totals["joined"] else 0.0,
        "group_count_exact": totals["count_exact"] / len(items) if items else 0.0,
        "group_id_set_exact": totals["id_set_exact"] / len(items) if items else 0.0,
        "closed_output_rate": totals["closed"] / len(items) if items else 0.0,
        "duplicate_ids": totals["duplicate_ids"],
        "unknown_ids": totals["unknown_ids"],
        "start_within_12_rate": totals["start_within_12"] / valid if valid else 0.0,
        "end_within_12_rate": totals["end_within_12"] / valid if valid else 0.0,
        "both_within_12_rate": totals["both_within_12"] / valid if valid else 0.0,
        "start_within_20_rate": totals["start_within_20"] / valid if valid else 0.0,
        "end_within_20_rate": totals["end_within_20"] / valid if valid else 0.0,
        "aligned_l2_mean_px": totals["aligned_l2_sum"] / distance_count if distance_count else None,
        "path_chamfer_mean_px": totals["chamfer_sum"] / distance_count if distance_count else None,
    }


def group_report(items: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item[key])].append(item)
    return {name: aggregate(group) for name, group in sorted(grouped.items())}


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        Glm4vForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True, trust_remote_code=False)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
    }
    if args.load_in_4bit:
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model_type = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False).model_type
    model_classes = {
        "glm4v": Glm4vForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model_type={model_type}")
    base = model_classes[model_type].from_pretrained(args.model_path, **kwargs)
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor, torch


def generate(model: Any, processor: Any, torch: Any, image_path: Path, system: str, prompt: str, max_new_tokens: int) -> str:
    from transformers import StoppingCriteria, StoppingCriteriaList

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[chat], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    inputs.pop("token_type_ids", None)
    input_len = int(inputs["input_ids"].shape[1])
    stop_tag = "</TRACES>"
    stop_ids = processor.tokenizer.encode(stop_tag, add_special_tokens=False)

    class StopOnSuffix(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(stop_ids) and input_ids.shape[1] >= len(stop_ids) and input_ids[0, -len(stop_ids):].tolist() == stop_ids

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnSuffix()]),
        )
    text = processor.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
    if stop_tag in text:
        text = text[:text.index(stop_tag) + len(stop_tag)]
    return text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_outputs(output_dir: Path, items: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    with (output_dir / "per_group_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": aggregate(items),
        "by_domain": group_report(items, "domain"),
        "by_difficulty": group_report(items, "difficulty"),
        "by_degree": group_report(items, "degree"),
    }
    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--merge-root", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_root:
        items = []
        for shard in range(args.num_shards):
            items.extend(load_jsonl(args.merge_root / f"shard{shard}" / "per_group_metrics.jsonl"))
        write_outputs(args.output_dir, items, args)
        print(json.dumps(aggregate(items), indent=2))
        return 0

    if not args.model_path or not args.adapter_path:
        parser.error("--model-path and --adapter-path are required outside merge mode")
    all_rows = load_jsonl(args.manifest)
    if args.max_samples is not None:
        all_rows = all_rows[:args.max_samples]
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    model, processor, torch = load_model(args)
    items = []
    for index, row in enumerate(rows, start=1):
        path = pred_dir / f'{row["sample_id"]}.txt'
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = generate(
                model, processor, torch, args.data_root / row["image_path"],
                row["system_prompt"], row["prompt"], args.max_new_tokens,
            )
            path.write_text(text, encoding="utf-8")
        items.append(evaluate_one(row, text))
        if index == 1 or index % 20 == 0 or index == len(rows):
            summary = aggregate(items)
            print(f'[{index}/{len(rows)} shard={args.shard_index}/{args.num_shards}] valid5={summary["valid5_per_gold"]:.4f} l2={summary["aligned_l2_mean_px"]}', flush=True)
    write_outputs(args.output_dir, items, args)
    print(json.dumps(aggregate(items), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
