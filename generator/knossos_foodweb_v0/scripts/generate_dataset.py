#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
RENDERER = load_module("knossos_renderer_v0", SCRIPT_DIR / "renderer.py")
DIFFGEN = load_module("knossos_difficulty_v0", SCRIPT_DIR / "generate_difficulty_samples.py")


DEFAULT_COUNTS = {
    "train": {"simple": 1500, "medium": 2750, "difficult": 750},
    "val": {"simple": 50, "medium": 300, "difficult": 150},
    "test": {"simple": 50, "medium": 300, "difficult": 150},
}


SCALE_BY_DIFFICULTY = {
    "simple": (0.80, 0.86, 0.92, 0.98),
    "medium": (0.90, 0.96, 1.00, 1.06),
    "difficult": (1.00, 1.06, 1.12),
}


def reading_order(nodes) -> List:
    return sorted(nodes, key=lambda node: (node.box[1] // 30, node.box[0], node.box[1]))


def reassign_ids(nodes, edges, negatives) -> None:
    old_to_new = {}
    for idx, node in enumerate(reading_order(nodes), 1):
        old_to_new[node.id] = f"n{idx}"
    for node in nodes:
        node.id = old_to_new[node.id]
    for edge in edges:
        edge.source = old_to_new[edge.source]
        edge.target = old_to_new[edge.target]
    for negative in negatives:
        negative.source = old_to_new[negative.source]
        negative.target = old_to_new[negative.target]


def node_name_by_id(nodes) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def midpoint(path: List[Tuple[int, int]]) -> Tuple[int, int]:
    if len(path) >= 3:
        return path[len(path) // 2]
    return ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


def scale_value(value: int, scale: float) -> int:
    return int(round(value * scale))


def scale_point(point: Tuple[int, int], scale: float) -> List[int]:
    return [scale_value(point[0], scale), scale_value(point[1], scale)]


def scale_box(box: Tuple[int, int, int, int], scale: float) -> List[int]:
    return [scale_value(coord, scale) for coord in box]


def scale_path(path: List[Tuple[int, int]], scale: float) -> List[List[int]]:
    return [scale_point(point, scale) for point in path]


def output_canvas(scale: float) -> Dict[str, int]:
    return {
        "width": scale_value(RENDERER.CANVAS_W, scale),
        "height": scale_value(RENDERER.CANVAS_H, scale),
    }


def choose_image_scale(difficulty: str, nodes, edges, seed: int, sample_id: str) -> float:
    return 0.9


def edge_record(edge, names: Dict[str, str], scale: float = 1.0) -> Dict:
    visible = RENDERER.visible_edge_path(edge, clearance=12)
    arrows = {
        "target": [scale_point(point, scale) for point in RENDERER.arrowhead_points(visible)]
        if edge.edge_type in {"directed", "bidirectional"}
        else [],
        "source": [scale_point(point, scale) for point in RENDERER.arrowhead_points(RENDERER.reversed_path(visible))]
        if edge.edge_type == "bidirectional"
        else [],
    }
    return {
        "id": edge.id,
        "source": edge.source,
        "source_name": names[edge.source],
        "target": edge.target,
        "target_name": names[edge.target],
        "type": edge.edge_type,
        "path": scale_path(edge.path, scale),
        "visible_path": scale_path(visible, scale),
        "start": scale_point(visible[0], scale),
        "mid": scale_point(midpoint(visible), scale),
        "end": scale_point(visible[-1], scale),
        "arrow": scale_point(visible[-1], scale) if edge.edge_type in {"directed", "bidirectional"} else None,
        "arrowheads": arrows,
    }


def node_record(node, scale: float = 1.0) -> Dict:
    return {
        "id": node.id,
        "name": node.label,
        "box": scale_box(node.box, scale),
        "center": scale_point(node.center, scale),
        "image_box": scale_box(node.icon_box, scale) if node.icon_box else None,
        "text_box": scale_box(node.text_box, scale),
        "asset": Path(node.asset_path).name if node.asset_path else None,
    }


def final_graph(nodes, edges, scale: float = 1.0) -> Dict:
    names = node_name_by_id(nodes)
    relation_map = {node.id: [] for node in nodes}
    for edge in edges:
        relation_map[edge.source].append(
            {
                "source": edge.source,
                "source_name": names[edge.source],
                "target": edge.target,
                "target_name": names[edge.target],
                "type": edge.edge_type,
            }
        )
        if edge.edge_type == "bidirectional":
            relation_map[edge.target].append(
                {
                    "source": edge.target,
                    "source_name": names[edge.target],
                    "target": edge.source,
                    "target_name": names[edge.source],
                    "type": edge.edge_type,
                }
            )
    return {
        "nodes": [
            {
                "id": node.id,
                "name": node.label,
                "box": scale_box(node.box, scale),
                "relations": relation_map[node.id],
            }
            for node in reading_order(nodes)
        ]
    }


def think_graph(nodes, edges, negatives, scale: float = 1.0) -> str:
    names = node_name_by_id(nodes)
    outgoing = {node.id: [] for node in nodes}
    hard_negatives = {node.id: [] for node in nodes}
    for edge in edges:
        outgoing[edge.source].append(edge)
        if edge.edge_type == "bidirectional":
            reverse = RENDERER.Edge(edge.id + "_rev", edge.target, edge.source, RENDERER.reversed_path(edge.path), edge.edge_type)
            outgoing[edge.target].append(reverse)
    for negative in negatives:
        hard_negatives[negative.source].append(negative)

    lines = ["<THINK_GRAPH>", "<NODES>"]
    for node in reading_order(nodes):
        lines.append(f'<NODE id="{node.id}" name="{node.label}" box="{scale_box(node.box, scale)}"/>')
    lines.extend(["</NODES>", ""])
    for node in reading_order(nodes):
        lines.append(f'<CHECK_NODE id="{node.id}" name="{node.label}">')
        for edge in outgoing[node.id]:
            rec = edge_record(edge, names, scale)
            lines.append(
                f'<TRACE target="{rec["target"]}" target_name="{rec["target_name"]}" '
                f'start="{rec["start"]}" mid="{rec["mid"]}" end="{rec["end"]}" arrow="{rec["arrow"]}"/>'
            )
            lines.append(
                f'<EDGE source="{rec["source"]}" source_name="{rec["source_name"]}" '
                f'target="{rec["target"]}" target_name="{rec["target_name"]}" type="{rec["type"]}"/>'
            )
        for negative in hard_negatives[node.id]:
            lines.append(
                f'<NO_EDGE source="{negative.source}" source_name="{names[negative.source]}" '
                f'target="{negative.target}" target_name="{names[negative.target]}" '
                f'reason="{negative.reason}"/>'
            )
        lines.extend(["</CHECK_NODE>", ""])
    lines.append("</THINK_GRAPH>")
    return "\n".join(lines)


def training_text(nodes, edges, negatives, scale: float = 1.0) -> str:
    return think_graph(nodes, edges, negatives, scale) + "\n\n<FINAL_GRAPH>\n" + json.dumps(final_graph(nodes, edges, scale), indent=2) + "\n</FINAL_GRAPH>"


def render_image(name: str, nodes, edges, extra_paths, image_path: Path, seed: int, scale: float = 1.0) -> Dict:
    ai2d_like = any(node.shape == "image_label" for node in nodes)
    style = RENDERER.style_for_case(name, seed, ai2d_like)
    image = Image.new("RGB", (RENDERER.CANVAS_W, RENDERER.CANVAS_H), style.background)
    draw = ImageDraw.Draw(image)
    if style.grid:
        RENDERER.draw_grid(draw)
    if ai2d_like and name.startswith("difficult"):
        font = RENDERER.load_font(12)
    elif ai2d_like and name.startswith("medium"):
        font = RENDERER.load_font(14)
    else:
        font = RENDERER.load_font(16 if ai2d_like else 22)
    for path in extra_paths:
        RENDERER.draw_near_miss(draw, path)
    for edge in edges:
        RENDERER.draw_edge(draw, edge, style)
    for node in nodes:
        RENDERER.draw_node(image, draw, node, font)
    for edge in edges:
        RENDERER.draw_edge_arrowheads(draw, edge, style)
    canvas = output_canvas(scale)
    if scale != 1.0:
        image = image.resize((canvas["width"], canvas["height"]), Image.Resampling.LANCZOS)
    image.save(image_path)
    return {
        "edge_color": list(style.edge_color),
        "edge_width": style.edge_width,
        "arrow_size": style.arrow_size,
        "background": list(style.background),
        "grid": style.grid,
        "halo": style.halo,
        "image_scale": scale,
        "canonical_canvas": {"width": RENDERER.CANVAS_W, "height": RENDERER.CANVAS_H},
        "output_canvas": canvas,
    }


def has_bbox_overlap(nodes, margin: int = 8) -> bool:
    for i, a in enumerate(nodes):
        ab = a.box
        for b in nodes[i + 1 :]:
            bb = b.box
            if not (ab[2] + margin < bb[0] or ab[0] - margin > bb[2] or ab[3] + margin < bb[1] or ab[1] - margin > bb[3]):
                return True
    return False


def is_valid(nodes, edges, negatives) -> bool:
    if any(node.asset_path is None for node in nodes):
        return False
    if has_bbox_overlap(nodes):
        return False
    node_ids = {node.id for node in nodes}
    for edge in edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            return False
        if edge.edge_type == "directed" and not RENDERER.arrowhead_points(RENDERER.visible_edge_path(edge, clearance=12)):
            return False
    for negative in negatives:
        if negative.source not in node_ids or negative.target not in node_ids:
            return False
    return True


def route_repair_edge(edge_id: int, src, dst) -> object:
    start = RENDERER.boundary_point_towards(src.box, dst.center)
    end = RENDERER.boundary_point_towards(dst.box, src.center)
    mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    return RENDERER.Edge(f"e{edge_id}", src.id, dst.id, [start, mid, end], "directed")


def comes_before(a, b, orientation: str) -> bool:
    if orientation == "bottom_to_top":
        return a.center[1] > b.center[1]
    if orientation == "top_to_bottom":
        return a.center[1] < b.center[1]
    if orientation == "left_to_right":
        return a.center[0] < b.center[0]
    if orientation == "right_to_left":
        return a.center[0] > b.center[0]
    return a.center < b.center


def ensure_no_isolated(nodes, edges, orientation: str) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    seen = {(edge.source, edge.target) for edge in edges}
    for node in nodes:
        if node.id in linked:
            continue
        candidates = [other for other in nodes if other.id != node.id]
        target = min(
            candidates,
            key=lambda other: math.hypot(node.center[0] - other.center[0], node.center[1] - other.center[1]),
        )
        src, dst = (node, target) if comes_before(node, target, orientation) else (target, node)
        if (src.id, dst.id) in seen:
            src, dst = dst, src
        if (src.id, dst.id) in seen:
            continue
        seen.add((src.id, dst.id))
        edges.append(route_repair_edge(len(edges) + 1, src, dst))
        linked.update([src.id, dst.id])


def make_sample(asset_index, difficulty: str, index: int, seed: int):
    for attempt in range(40):
        _, nodes, edges, negatives, extra_paths, orientation = DIFFGEN.make_case(
            RENDERER, asset_index, difficulty, index * 100 + attempt, seed
        )
        ensure_no_isolated(nodes, edges, orientation)
        reassign_ids(nodes, edges, negatives)
        if is_valid(nodes, edges, negatives):
            return nodes, edges, negatives, extra_paths, orientation, attempt
    raise RuntimeError(f"failed to generate valid sample for {difficulty} #{index}")


def parse_counts(text: str | None) -> Dict[str, Dict[str, int]]:
    if not text:
        return DEFAULT_COUNTS
    result = {}
    for split_spec in text.split(";"):
        split, values = split_spec.split(":", 1)
        result[split] = {}
        for item in values.split(","):
            key, value = item.split("=", 1)
            result[split][key] = int(value)
    return result


def write_annotation(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="knossos_foodweb_v0/assets/renderer_asset_pool")
    parser.add_argument("--out", default="knossos_foodweb_v0/outputs/Knossos-FoodWeb-v0-6000-adaptive")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--counts", default=None, help='Optional: train:simple=1,medium=1,difficult=1;val:simple=...')
    args = parser.parse_args()

    asset_index = DIFFGEN.build_asset_index(Path(args.asset_dir))
    out_dir = Path(args.out)
    images_root = out_dir / "images"
    annotations_root = out_dir / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    counts = parse_counts(args.counts)
    summary = []
    global_index = 1
    for split, split_counts in counts.items():
        for difficulty in ["simple", "medium", "difficult"]:
            count = split_counts.get(difficulty, 0)
            image_dir = images_root / split / difficulty
            ann_dir = annotations_root / split / difficulty
            image_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            for local_idx in range(1, count + 1):
                sample_id = f"{split}_{difficulty}_{local_idx:06d}"
                nodes, edges, negatives, extra_paths, orientation, attempt = make_sample(asset_index, difficulty, global_index, args.seed)
                image_scale = choose_image_scale(difficulty, nodes, edges, args.seed, sample_id)
                canvas = output_canvas(image_scale)
                image_path = image_dir / f"{sample_id}.png"
                ann_path = ann_dir / f"{sample_id}.json"
                render_style = render_image(difficulty + "_" + sample_id, nodes, edges, extra_paths, image_path, args.seed, image_scale)
                names = node_name_by_id(nodes)
                annotation = {
                    "sample_id": sample_id,
                    "image": str(image_path.relative_to(out_dir)),
                    "split": split,
                    "difficulty": difficulty,
                    "orientation": orientation,
                    "canvas": canvas,
                    "nodes": [node_record(node, image_scale) for node in reading_order(nodes)],
                    "edges": [edge_record(edge, names, image_scale) for edge in edges],
                    "negative_edges": [
                        {
                            "source": negative.source,
                            "source_name": names[negative.source],
                            "target": negative.target,
                            "target_name": names[negative.target],
                            "reason": negative.reason,
                            "evidence_point": scale_point(negative.evidence_point, image_scale),
                        }
                        for negative in negatives
                    ],
                    "final_graph": final_graph(nodes, edges, image_scale),
                    "think_graph": think_graph(nodes, edges, negatives, image_scale),
                    "training_text": training_text(nodes, edges, negatives, image_scale),
                    "render_style": render_style,
                    "generation": {"seed": args.seed, "global_index": global_index, "attempt": attempt, "image_scale": image_scale},
                }
                write_annotation(ann_path, annotation)
                summary.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "difficulty": difficulty,
                        "orientation": orientation,
                        "nodes": len(nodes),
                        "edges": len(edges),
                        "negative_edges": len(negatives),
                        "image_scale": image_scale,
                        "canvas": canvas,
                        "image": str(image_path.relative_to(out_dir)),
                        "annotation": str(ann_path.relative_to(out_dir)),
                    }
                )
                global_index += 1
            print(f"generated {split}/{difficulty}: {count}")

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {len(summary)} samples in {out_dir.resolve()}")
    print(f"Images: {(out_dir / 'images').resolve()}")
    print(f"Annotations: {(out_dir / 'annotations').resolve()}")


if __name__ == "__main__":
    main()
