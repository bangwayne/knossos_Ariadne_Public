#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOP_LEVEL_REQUIRED = {
    "sample_id",
    "image",
    "split",
    "difficulty",
    "orientation",
    "canvas",
    "nodes",
    "edges",
    "negative_edges",
    "final_graph",
    "think_graph",
    "training_text",
    "render_style",
    "generation",
}

NODE_REQUIRED = {"id", "name", "box", "center", "image_box", "text_box", "asset"}
EDGE_REQUIRED = {
    "id",
    "source",
    "source_name",
    "target",
    "target_name",
    "type",
    "path",
    "visible_path",
    "start",
    "mid",
    "end",
    "arrow",
    "arrowheads",
}
NEGATIVE_REQUIRED = {"source", "source_name", "target", "target_name", "reason", "evidence_point"}
RELATION_REQUIRED = {"source", "source_name", "target", "target_name", "type"}


def is_point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) for x in value)


def is_box(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(x, int) for x in value)


def check_required(errors: list[str], label: str, data: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(data))
    if missing:
        errors.append(f"{label}: missing keys {missing}")


def validate_annotation(path: Path, *, strict_top_level: bool) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    check_required(errors, "top_level", data, TOP_LEVEL_REQUIRED)
    if strict_top_level:
        extra = sorted(set(data) - TOP_LEVEL_REQUIRED)
        if extra:
            errors.append(f"top_level: extra keys {extra}")

    canvas = data.get("canvas")
    if (
        not isinstance(canvas, dict)
        or not isinstance(canvas.get("width"), int)
        or not isinstance(canvas.get("height"), int)
        or canvas.get("width", 0) <= 0
        or canvas.get("height", 0) <= 0
    ):
        errors.append("canvas: expected positive integer width/height")
    if not isinstance(data.get("nodes"), list) or not data["nodes"]:
        errors.append("nodes: expected non-empty list")
    if not isinstance(data.get("edges"), list):
        errors.append("edges: expected list")
    if not isinstance(data.get("negative_edges"), list):
        errors.append("negative_edges: expected list")
    if not isinstance(data.get("final_graph"), dict):
        errors.append("final_graph: expected object")
    if not isinstance(data.get("think_graph"), str) or "<THINK_GRAPH>" not in data.get("think_graph", ""):
        errors.append("think_graph: missing <THINK_GRAPH>")
    if not isinstance(data.get("training_text"), str) or "<FINAL_GRAPH>" not in data.get("training_text", ""):
        errors.append("training_text: missing <FINAL_GRAPH>")

    node_ids = set()
    for idx, node in enumerate(data.get("nodes", []), 1):
        check_required(errors, f"nodes[{idx}]", node, NODE_REQUIRED)
        node_ids.add(node.get("id"))
        if not is_box(node.get("box")):
            errors.append(f"nodes[{idx}].box: expected 4 ints")
        if not is_point(node.get("center")):
            errors.append(f"nodes[{idx}].center: expected 2 ints")
        if node.get("image_box") is not None and not is_box(node.get("image_box")):
            errors.append(f"nodes[{idx}].image_box: expected 4 ints or null")
        if not is_box(node.get("text_box")):
            errors.append(f"nodes[{idx}].text_box: expected 4 ints")

    linked = set()
    for idx, edge in enumerate(data.get("edges", []), 1):
        check_required(errors, f"edges[{idx}]", edge, EDGE_REQUIRED)
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            errors.append(f"edges[{idx}]: source/target not in nodes")
        linked.update([edge.get("source"), edge.get("target")])
        for key in ("path", "visible_path"):
            if not isinstance(edge.get(key), list) or len(edge[key]) < 2 or not all(is_point(point) for point in edge[key]):
                errors.append(f"edges[{idx}].{key}: expected list of points")
        for key in ("start", "mid", "end"):
            if not is_point(edge.get(key)):
                errors.append(f"edges[{idx}].{key}: expected point")
        if edge.get("arrow") is not None and not is_point(edge.get("arrow")):
            errors.append(f"edges[{idx}].arrow: expected point or null")
        arrowheads = edge.get("arrowheads", {})
        if sorted(arrowheads.keys()) != ["source", "target"]:
            errors.append(f"edges[{idx}].arrowheads: expected source/target keys")

    isolated = sorted(node_id for node_id in node_ids if node_id not in linked)
    if isolated:
        errors.append(f"nodes: isolated nodes {isolated}")

    for idx, negative in enumerate(data.get("negative_edges", []), 1):
        check_required(errors, f"negative_edges[{idx}]", negative, NEGATIVE_REQUIRED)
        if not is_point(negative.get("evidence_point")):
            errors.append(f"negative_edges[{idx}].evidence_point: expected point")

    graph_nodes = data.get("final_graph", {}).get("nodes", [])
    if len(graph_nodes) != len(data.get("nodes", [])):
        errors.append("final_graph.nodes: length differs from nodes")
    for idx, node in enumerate(graph_nodes, 1):
        for relation in node.get("relations", []):
            check_required(errors, f"final_graph.nodes[{idx}].relations", relation, RELATION_REQUIRED)

    if data.get("training_text") != data.get("think_graph") + "\n\n<FINAL_GRAPH>\n" + json.dumps(data.get("final_graph"), indent=2) + "\n</FINAL_GRAPH>":
        errors.append("training_text: does not exactly match think_graph + final_graph block")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_dir", type=Path)
    parser.add_argument("--strict-top-level", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.annotations_dir.rglob("*.json"))
    if not paths:
        raise SystemExit(f"No annotations found under {args.annotations_dir}")

    failures = {}
    for path in paths:
        errors = validate_annotation(path, strict_top_level=args.strict_top_level)
        if errors:
            failures[str(path)] = errors

    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "checked": len(paths), "strict_top_level": args.strict_top_level}, indent=2))


if __name__ == "__main__":
    main()
