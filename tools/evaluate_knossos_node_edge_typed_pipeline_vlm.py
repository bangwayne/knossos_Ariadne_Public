#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


EDGE_SYSTEM_PROMPT = (
    "You are TopoAgent-GP Edge Reasoner. Given a diagram image, a fixed node "
    "list, and selected source nodes, extract only topology relations whose "
    "source is in the selected group. Treat TRACE as visual evidence and EDGE "
    "as a typed multigraph relation."
)

EDGE_PROMPT_TEMPLATE = """Extract typed topology relations for the selected source nodes.

You are given:
1. the diagram image
2. the complete fixed node list with id/name/box
3. the current source node group

Use only the provided node ids and names.
Only output relations whose source is in the current source node group.
Do not output edges from other source nodes.

For every positive relation:
- output TRACE first, then EDGE
- TRACE must contain exactly 5 points sampled along the visible connector
- point 1 must start near the source node boundary
- point 5 must end near the target node boundary or arrowhead when present
- intermediate points must follow the visible connector, not a straight-line guess
- EDGE type must preserve the exact relation/line type shown by the dataset or legend
- if the same source-target pair has multiple types, output one TRACE+EDGE pair for each type; do not merge multi-links

If a source node has no outgoing relation, output an empty CHECK_NODE.
Use sparse NO_EDGE only for hard negatives such as near_miss, crossing_not_connected, or wrong_direction.
Return only <EDGE_GRAPH>. Do not output <NODES> or FINAL_GRAPH. Do not explain.

Complete node list:
{node_list}

Current source node group:
{source_group}

Output format:
<EDGE_GRAPH>
<CHECK_NODE id="n1" name="Source A">
<TRACE target="n2" target_name="Target B" points="[[180,560],[205,520],[230,470],[200,430],[165,390]]" arrow="[165,390]"/>
<EDGE source="n1" source_name="Source A" target="n2" target_name="Target B" type="causes"/>
</CHECK_NODE>
</EDGE_GRAPH>

Rules:
- Put each relation under its source CHECK_NODE.
- Each CHECK_NODE in the current source node group must appear exactly once.
- The EDGE type is a typed multigraph label and may be domain-specific, such as directed, undirected, wire, signal, power, ground, inhibits, flows_to, or a route/line/service name.
- For visual directed arrows, include arrow="[x,y]" in TRACE at the target side.
- For visual bidirectional arrows, include arrow_start="[x,y]" and arrow_end="[x,y]" in TRACE.
- For no-arrow connectors, do not include arrow fields.
- Do not repeat node boxes in the answer."""

EDGE_ONLY_SYSTEM_PROMPT = (
    "You are TopoAgent-GP Edge Reasoner. Extract typed multigraph relations "
    "for the selected source nodes and output EDGE elements only."
)

EDGE_ONLY_PROMPT_TEMPLATE = """Extract typed topology relations for the selected source nodes.

You are given:
1. the diagram image
2. the complete fixed node list with id/name/box
3. the current source node group

Use only the provided node ids and names.
Only output relations whose source is in the current source node group.
Do not output edges from other source nodes.

For every positive relation:
- output EDGE directly; do not output TRACE
- EDGE target and type must preserve the exact visible relation shown by the dataset or legend
- if the same source-target pair has multiple types, output one EDGE for each type; do not merge multi-links

If a source node has no outgoing relation, output an empty CHECK_NODE.
Use sparse NO_EDGE only for hard negatives such as near_miss, crossing_not_connected, or wrong_direction.
Return only <EDGE_GRAPH>. Do not output <NODES> or FINAL_GRAPH. Do not explain.

Complete node list:
{node_list}

Current source node group:
{source_group}

Output format:
<EDGE_GRAPH>
<CHECK_NODE id="n1" name="Source A">
<EDGE source="n1" source_name="Source A" target="n2" target_name="Target B" type="causes"/>
</CHECK_NODE>
</EDGE_GRAPH>

Rules:
- Put each relation under its source CHECK_NODE.
- Each CHECK_NODE in the current source node group must appear exactly once.
- Output EDGE elements only; never output TRACE.
- EDGE carries the source, target, names, and exact typed multigraph label.
- Do not repeat node boxes in the answer."""


def parse_attrs(fragment: str) -> Dict[str, str]:
    return {key: value for key, value in re.findall(r'(\w+)="([^"]*)"', fragment)}


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def parse_box(value: Any) -> List[float]:
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "[]"))
        except Exception:
            raw = []
    if len(raw) != 4:
        return []
    try:
        return [float(x) for x in raw]
    except Exception:
        return []


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def parse_points(raw: str) -> Optional[List[List[float]]]:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if not isinstance(value, list) or len(value) != 5:
        return None
    out: List[List[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            out.append([float(point[0]), float(point[1])])
        except Exception:
            return None
    return out


def arrow_kind(attrs: Dict[str, str]) -> str:
    if "arrow_start" in attrs and "arrow_end" in attrs:
        return "both"
    if "arrow" in attrs:
        return "end"
    return "none"


def extract_nodes(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    match = re.search(r"<NODES>\s*(.*?)\s*(?:</NODES>|$)", text, flags=re.S)
    body = match.group(1) if match else text
    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in re.finditer(r"<NODE\s+([^>/]+?)/>", body):
        attrs = parse_attrs(item.group(1))
        node_id, name = attrs.get("id"), attrs.get("name")
        if not node_id or not name or node_id in seen:
            continue
        seen.add(node_id)
        nodes.append({"id": node_id, "name": name, "box": parse_box(attrs.get("box"))})
    if nodes:
        return nodes, None if match else "recovered_missing_nodes_tag"
    return [], "missing_nodes_tag"


def extract_edge_graph(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info = {
        "has_edge_graph_open": "<EDGE_GRAPH>" in text,
        "has_edge_graph_close": "</EDGE_GRAPH>" in text,
        "num_trace_tags": 0,
        "num_check_nodes": 0,
    }
    match = re.search(r"<EDGE_GRAPH>\s*(.*?)\s*(?:</EDGE_GRAPH>|$)", text, flags=re.S)
    if not match:
        info["format_error"] = "missing_edge_graph"
        return [], info
    edges: List[Dict[str, Any]] = []
    for block in re.finditer(r"<CHECK_NODE\s+([^>]*)>(.*?)(?=<CHECK_NODE\s+|</EDGE_GRAPH>|$)", match.group(1), flags=re.S):
        info["num_check_nodes"] += 1
        check_source = parse_attrs(block.group(1)).get("id", "")
        pending_trace: Optional[Dict[str, Any]] = None
        for tag in re.finditer(r"<(TRACE|EDGE)\s+([^>/]*?)/>", block.group(2), flags=re.S):
            attrs = parse_attrs(tag.group(2))
            if tag.group(1) == "TRACE":
                info["num_trace_tags"] += 1
                points = parse_points(attrs.get("points", ""))
                pending_trace = {
                    "target": attrs.get("target", ""),
                    "points": points,
                    "points_valid": points is not None,
                    "arrow_kind": arrow_kind(attrs),
                }
            else:
                src = attrs.get("source") or check_source
                tgt = attrs.get("target")
                typ = attrs.get("type")
                if not src or not tgt or not typ:
                    pending_trace = None
                    continue
                trace = pending_trace if pending_trace and pending_trace.get("target") == tgt else None
                edges.append(
                    {
                        "source": src,
                        "source_name": attrs.get("source_name", ""),
                        "target": tgt,
                        "target_name": attrs.get("target_name", ""),
                        "type": typ,
                        "trace": trace,
                    }
                )
                pending_trace = None
    if not info["has_edge_graph_close"]:
        info["format_error"] = "missing_edge_graph_close"
    return edges, info


def load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_stem(sample_id: str) -> str:
    return sample_id.split("_edge_trace5_typed_")[0]


def relation_iter(graph: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for node in graph.get("nodes", []) or []:
        for rel in node.get("relations", []) or []:
            if isinstance(rel, dict):
                yield rel


def greedy_node_mapping(pred_nodes: Sequence[Dict[str, Any]], gt_nodes: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    gt_by_name: Dict[str, List[int]] = defaultdict(list)
    for idx, node in enumerate(gt_nodes):
        key = normalize_name(node.get("name"))
        if key:
            gt_by_name[key].append(idx)
    used_gt: set[int] = set()
    mapping: Dict[str, str] = {}
    for pred in pred_nodes:
        key = normalize_name(pred.get("name"))
        candidates = [idx for idx in gt_by_name.get(key, []) if idx not in used_gt]
        if not candidates:
            continue
        best = max(candidates, key=lambda idx: box_iou(pred.get("box", []), gt_nodes[idx].get("box", [])))
        used_gt.add(best)
        mapping[str(pred["id"])] = str(gt_nodes[best]["id"])
    return mapping


def edge_key(edge: Dict[str, Any], id_map: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
    src = str(edge.get("source", ""))
    tgt = str(edge.get("target", ""))
    if id_map is not None:
        src = id_map.get(src, f"UNMATCHED:{src}")
        tgt = id_map.get(tgt, f"UNMATCHED:{tgt}")
    return (src, tgt, str(edge.get("type", "")))


def endpoint_key(edge: Dict[str, Any], id_map: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    src = str(edge.get("source", ""))
    tgt = str(edge.get("target", ""))
    if id_map is not None:
        src = id_map.get(src, f"UNMATCHED:{src}")
        tgt = id_map.get(tgt, f"UNMATCHED:{tgt}")
    return (src, tgt)


def prf(tp: int, pred: int, gt: int) -> Dict[str, float]:
    p = tp / pred if pred else 0.0
    r = tp / gt if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def counter_overlap(pred: Counter, gt: Counter) -> int:
    return sum((pred & gt).values())


def pair_type_multiset(edges: Iterable[Dict[str, Any]], id_map: Optional[Dict[str, str]] = None) -> Dict[Tuple[str, str], Counter]:
    out: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for edge in edges:
        out[endpoint_key(edge, id_map)][str(edge.get("type", ""))] += 1
    return out


def node_name_counts(nodes: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(normalize_name(n.get("name")) for n in nodes if normalize_name(n.get("name")))


def evaluate_image(
    row: Dict[str, Any],
    pred_nodes: Sequence[Dict[str, Any]],
    pred_edges: Sequence[Dict[str, Any]],
    edge_parse_infos: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gt_graph = row["target"]
    gt_nodes = gt_graph.get("nodes", []) or []
    id_map = greedy_node_mapping(pred_nodes, gt_nodes)
    gt_edges = list(relation_iter(gt_graph))

    pred_node_counter = node_name_counts(pred_nodes)
    gt_node_counter = node_name_counts(gt_nodes)
    node_tp = counter_overlap(pred_node_counter, gt_node_counter)

    pred_endpoint = Counter(endpoint_key(e, id_map) for e in pred_edges)
    gt_endpoint = Counter(endpoint_key(e) for e in gt_edges)
    pred_typed = Counter(edge_key(e, id_map) for e in pred_edges)
    gt_typed = Counter(edge_key(e) for e in gt_edges)
    pred_pair_types = pair_type_multiset(pred_edges, id_map)
    gt_pair_types = pair_type_multiset(gt_edges)
    pair_union = set(pred_pair_types) | set(gt_pair_types)
    pair_exact = sum(1 for pair in pair_union if pred_pair_types.get(pair, Counter()) == gt_pair_types.get(pair, Counter()))

    trace_present = sum(1 for e in pred_edges if e.get("trace"))
    trace_valid5 = sum(1 for e in pred_edges if (e.get("trace") or {}).get("points_valid"))
    format_errors = Counter(info.get("format_error") for info in edge_parse_infos if info.get("format_error"))

    return {
        "sample_id": row["sample_id"],
        "domain": row.get("domain"),
        "difficulty": row.get("difficulty"),
        "meta": row.get("meta") or {},
        "counts": {
            "node_tp": node_tp,
            "node_pred": sum(pred_node_counter.values()),
            "node_gt": sum(gt_node_counter.values()),
            "node_count_exact": int(sum(pred_node_counter.values()) == sum(gt_node_counter.values())),
            "node_mapped": len(id_map),
            "endpoint_tp": counter_overlap(pred_endpoint, gt_endpoint),
            "endpoint_pred": sum(pred_endpoint.values()),
            "endpoint_gt": sum(gt_endpoint.values()),
            "typed_tp": counter_overlap(pred_typed, gt_typed),
            "typed_pred": sum(pred_typed.values()),
            "typed_gt": sum(gt_typed.values()),
            "pair_exact": pair_exact,
            "pair_total": len(pair_union),
            "sample_multigraph_exact": int(pred_typed == gt_typed),
            "trace_present": trace_present,
            "trace_valid5": trace_valid5,
            "pred_edges": len(pred_edges),
            "edge_groups": len(edge_parse_infos),
            "edge_format_errors": sum(format_errors.values()),
        },
        "edge_format_errors": dict(format_errors),
    }


def aggregate(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals = Counter()
    errors = Counter()
    for item in items:
        totals.update(item["counts"])
        errors.update(item.get("edge_format_errors") or {})
    return {
        "num_images": len(items),
        "node": {**prf(totals["node_tp"], totals["node_pred"], totals["node_gt"]), "tp": totals["node_tp"], "pred": totals["node_pred"], "gt": totals["node_gt"]},
        "node_count_exact_rate": totals["node_count_exact"] / len(items) if items else 0.0,
        "node_mapping_rate": totals["node_mapped"] / totals["node_gt"] if totals["node_gt"] else 0.0,
        "endpoint": {**prf(totals["endpoint_tp"], totals["endpoint_pred"], totals["endpoint_gt"]), "tp": totals["endpoint_tp"], "pred": totals["endpoint_pred"], "gt": totals["endpoint_gt"]},
        "typed_edge": {**prf(totals["typed_tp"], totals["typed_pred"], totals["typed_gt"]), "tp": totals["typed_tp"], "pred": totals["typed_pred"], "gt": totals["typed_gt"]},
        "multigraph_pair_exact_rate": totals["pair_exact"] / totals["pair_total"] if totals["pair_total"] else 0.0,
        "multigraph_image_exact_rate": totals["sample_multigraph_exact"] / len(items) if items else 0.0,
        "trace_present_per_pred_edge": totals["trace_present"] / totals["pred_edges"] if totals["pred_edges"] else 0.0,
        "trace5_valid_per_pred_edge": totals["trace_valid5"] / totals["pred_edges"] if totals["pred_edges"] else 0.0,
        "edge_group_format_error_rate": totals["edge_format_errors"] / totals["edge_groups"] if totals["edge_groups"] else 0.0,
        "edge_format_errors": dict(errors),
    }


def group_by(items: Sequence[Dict[str, Any]], key_fn: Any) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        out[str(key_fn(item))].append(item)
    return dict(out)


def build_messages(system: str, prompt: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    return messages


def generate_text(model: Any, processor: Any, torch: Any, image_path: Path, system: str, prompt: str, stop_tag: str, max_new_tokens: int) -> str:
    from transformers import StoppingCriteria, StoppingCriteriaList

    image = Image.open(image_path).convert("RGB")
    chat_text = processor.apply_chat_template(build_messages(system, prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    inputs.pop("token_type_ids", None)
    input_len = int(inputs["input_ids"].shape[1])
    stop_ids = processor.tokenizer.encode(stop_tag, add_special_tokens=False)

    class StopOnSuffix(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(stop_ids) and input_ids.shape[1] >= len(stop_ids) and input_ids[0, -len(stop_ids) :].tolist() == stop_ids

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnSuffix()]),
        )
    decoded = processor.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    if stop_tag in decoded:
        decoded = decoded[: decoded.index(stop_tag) + len(stop_tag)]
    return decoded


def load_model(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
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
    kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "trust_remote_code": False, "attn_implementation": "sdpa"}
    if args.load_in_4bit:
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["device_map"] = "auto"
    model_type = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False).model_type
    model_classes = {
        "glm4v": Glm4vForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model_type={model_type}")
    base = model_classes[model_type].from_pretrained(args.model_path, **kwargs)
    model = PeftModel.from_pretrained(base, args.node_adapter_path, adapter_name="node")
    model.load_adapter(args.edge_adapter_path, adapter_name="edge")
    model.eval()
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor, torch


def chunks(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def make_edge_prompt(
    nodes: Sequence[Dict[str, Any]],
    source_nodes: Sequence[Dict[str, Any]],
    edge_only: bool = False,
) -> str:
    node_list = json.dumps(
        [{"id": n["id"], "name": n["name"], "box": [int(round(x)) for x in n.get("box", [])]} for n in nodes],
        ensure_ascii=False,
        indent=2,
    )
    source_group = json.dumps(
        [{"id": n["id"], "name": n["name"]} for n in source_nodes],
        ensure_ascii=False,
        indent=2,
    )
    template = EDGE_ONLY_PROMPT_TEMPLATE if edge_only else EDGE_PROMPT_TEMPLATE
    return template.format(node_list=node_list, source_group=source_group)


def report_for_items(items: Sequence[Dict[str, Any]], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "args": args_dict,
        "summary": aggregate(items),
        "by_domain": {key: aggregate(value) for key, value in sorted(group_by(items, lambda x: x["domain"]).items())},
        "by_difficulty": {key: aggregate(value) for key, value in sorted(group_by(items, lambda x: x["difficulty"]).items())},
        "by_domain_difficulty": {
            key: aggregate(value)
            for key, value in sorted(group_by(items, lambda x: f'{x["domain"]}/{x["difficulty"]}').items())
        },
        "by_tag": {
            tag: aggregate([item for item in items if (item.get("meta") or {}).get(tag)])
            for tag in ["has_multilink", "has_typed_edge", "has_legend", "has_arrow", "has_no_arrow"]
            if any((item.get("meta") or {}).get(tag) for item in items)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-manifest", type=Path, required=True)
    parser.add_argument("--edge-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--node-adapter-path", type=Path, required=True)
    parser.add_argument("--edge-adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-group-size", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--node-max-new-tokens", type=int, default=2048)
    parser.add_argument("--edge-max-new-tokens", type=int, default=2048)
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_pred_dir = args.output_dir / "node_predictions"
    edge_pred_dir = args.output_dir / "edge_predictions"
    node_pred_dir.mkdir(parents=True, exist_ok=True)
    edge_pred_dir.mkdir(parents=True, exist_ok=True)

    node_rows = load_jsonl(args.node_manifest, args.max_samples)
    node_rows = [row for idx, row in enumerate(node_rows) if idx % args.num_shards == args.shard_index]
    edge_rows = load_jsonl(args.edge_manifest)
    graph_by_stem: Dict[str, Dict[str, Any]] = {}
    meta_by_stem: Dict[str, Dict[str, Any]] = {}
    for row in edge_rows:
        stem = sample_stem(row["sample_id"])
        graph_by_stem.setdefault(stem, row["target"])
        meta_by_stem.setdefault(stem, {"domain": row.get("domain"), "difficulty": row.get("difficulty"), **(row.get("meta") or {})})

    model, processor, torch = load_model(args)
    details: List[Dict[str, Any]] = []
    for idx, node_row in enumerate(node_rows, start=1):
        stem = node_row["sample_id"]
        image_path = args.data_root / node_row["image_path"]
        node_text_path = node_pred_dir / f"{stem}.nodes.txt"
        if node_text_path.exists():
            node_text = node_text_path.read_text(encoding="utf-8")
        else:
            model.set_adapter("node")
            node_text = generate_text(
                model,
                processor,
                torch,
                image_path,
                node_row.get("system_prompt") or "",
                node_row["prompt"],
                "</NODES>",
                args.node_max_new_tokens,
            )
            node_text_path.write_text(node_text, encoding="utf-8")
        pred_nodes, node_error = extract_nodes(node_text)

        all_edges: List[Dict[str, Any]] = []
        parse_infos: List[Dict[str, Any]] = []
        for group_idx, source_nodes in enumerate(chunks(pred_nodes, args.source_group_size), start=1):
            edge_text_path = edge_pred_dir / f"{stem}.g{group_idx:02d}.edge_graph.txt"
            if edge_text_path.exists():
                edge_text = edge_text_path.read_text(encoding="utf-8")
            else:
                model.set_adapter("edge")
                edge_text = generate_text(
                    model,
                    processor,
                    torch,
                    image_path,
                    EDGE_ONLY_SYSTEM_PROMPT if args.edge_only else EDGE_SYSTEM_PROMPT,
                    make_edge_prompt(pred_nodes, source_nodes, args.edge_only),
                    "</EDGE_GRAPH>",
                    args.edge_max_new_tokens,
                )
                edge_text_path.write_text(edge_text, encoding="utf-8")
            edges, parse_info = extract_edge_graph(edge_text)
            all_edges.extend(edges)
            parse_infos.append(parse_info)

        eval_row = {
            **node_row,
            "target": graph_by_stem[stem],
            "meta": meta_by_stem.get(stem, {}),
        }
        item = evaluate_image(eval_row, pred_nodes, all_edges, parse_infos)
        item["node_format_error"] = node_error
        details.append(item)
        if idx == 1 or idx % 10 == 0 or idx == len(node_rows):
            agg = aggregate(details)
            print(
                f"[{idx}/{len(node_rows)} shard={args.shard_index}/{args.num_shards}] {stem} "
                f"node_f1={agg['node']['f1']:.4f} typed_f1={agg['typed_edge']['f1']:.4f} "
                f"endpoint_f1={agg['endpoint']['f1']:.4f}",
                flush=True,
            )

    with (args.output_dir / "per_image_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = report_for_items(details, args_dict)
    (args.output_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
