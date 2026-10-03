#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RENDERER_PATH = REPO / "knossos_foodweb_v0" / "scripts" / "renderer.py"
ASSET_DIR = ROOT / "assets" / "renderer_asset_pool"
LABELS_PATH = ROOT / "config" / "network_labels.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDERER = load_module("knossos_network_renderer_preview", RENDERER_PATH)

Point = Tuple[int, int]
_LABEL_CACHE: List[Dict[str, str]] | None = None
_ASSET_INDEX_CACHE: Dict[str, Dict[str, List[str]]] | None = None


EDGE_RATIOS = {
    "simple": [("undirected", 1.00), ("directed", 0.00)],
    "medium": [("undirected", 1.00), ("directed", 0.00), ("bidirectional", 0.00)],
    "difficult": [("undirected", 1.00), ("directed", 0.00), ("bidirectional", 0.00)],
}

NODE_COUNTS = {
    "simple": (8, 10),
    "medium": (12, 15),
    "difficult": (14, 17),
}

EDGE_RANGES = {
    "simple": (8, 12),
    "medium": (13, 19),
    "difficult": (23, 27),
}

MIN_EDGE_DISTANCE = {
    "simple": 185,
    "medium": 170,
    "difficult": 145,
}

TOPOLOGIES = {
    "simple": ["star", "layered"],
    "medium": ["layered"],
    "difficult": ["layered", "cluster_bridge", "foodweb_tier"],
}

ORIENTATIONS = ["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"]
NETWORK_EDGE_CLEARANCE = -5

NETWORK_EDGE_COLORS = [
    (36, 48, 66),
    (24, 91, 132),
    (29, 110, 85),
    (83, 72, 154),
    (128, 76, 30),
    (145, 64, 72),
    (26, 105, 122),
]

SCALE_BY_DIFFICULTY = {
    "simple": (0.80, 0.86, 0.92, 0.98),
    "medium": (0.92, 0.98, 1.04, 1.08),
    "difficult": (1.00, 1.06, 1.12, 1.16),
}


def load_labels() -> List[Dict[str, str]]:
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["network"]
    return _LABEL_CACHE


def load_asset_index() -> Dict[str, Dict[str, List[str]]]:
    global _ASSET_INDEX_CACHE
    if _ASSET_INDEX_CACHE is not None:
        return _ASSET_INDEX_CACHE
    metadata = json.loads((ASSET_DIR / "metadata.json").read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, List[str]]] = {}
    for filename, info in metadata.items():
        index.setdefault(info["label"], {}).setdefault(info["style"], []).append(str(ASSET_DIR / filename))
    for styles in index.values():
        for paths in styles.values():
            paths.sort()
    _ASSET_INDEX_CACHE = index
    return _ASSET_INDEX_CACHE


def display_name(base: str, rng: random.Random) -> str:
    if rng.random() < 0.35:
        return base
    suffixes = [
        lambda: rng.choice(["A", "B", "C", "D"]),
        lambda: f"{rng.randint(1, 12):02d}",
        lambda: rng.choice(["North", "South", "East", "West", "Core", "Edge"]),
        lambda: f"-{rng.randint(1, 4)}",
    ]
    suffix = rng.choice(suffixes)()
    if suffix.startswith("-"):
        return f"{base}{suffix}"
    return f"{base} {suffix}"


def choose_asset(label: str, asset_style: str, asset_index: Dict[str, Dict[str, List[str]]], rng: random.Random) -> str:
    by_style = asset_index[label]
    paths = by_style.get(asset_style) or next(iter(by_style.values()))
    return rng.choice(paths)


def node_size(difficulty: str) -> Tuple[int, int]:
    if difficulty == "simple":
        return (82, 94)
    if difficulty == "medium":
        return (70, 82)
    return (60, 72)


def transform_center(center: Point, orientation: str) -> Point:
    x, y = center
    w, h = RENDERER.CANVAS_W, RENDERER.CANVAS_H
    if orientation == "top_to_bottom":
        return (x, y)
    if orientation == "bottom_to_top":
        return (x, h - y)
    if orientation == "left_to_right":
        return (int(y / h * w), int(x / w * h))
    if orientation == "right_to_left":
        return (int((1 - y / h) * w), int(x / w * h))
    return (x, y)


def as_top_left(center: Point, difficulty: str) -> Point:
    width, height = node_size(difficulty)
    return (center[0] - width // 2, center[1] - height // 2)


def make_node(node_id: str, label: str, x: int, y: int, difficulty: str, asset_path: str):
    width, height = node_size(difficulty)
    node = RENDERER.Node(node_id, label, (x, y, x + width, y + height), "image_label")
    node.asset_path = asset_path
    return node


def radial_points(count: int, cx: int, cy: int, rx: int, ry: int, start: float = -math.pi / 2) -> List[Point]:
    return [
        (int(cx + rx * math.cos(start + 2 * math.pi * idx / count)), int(cy + ry * math.sin(start + 2 * math.pi * idx / count)))
        for idx in range(count)
    ]


def style_for_sample(difficulty: str, topology: str, index: int, rng: random.Random) -> str:
    if difficulty == "simple":
        return "textbook_icon"
    if difficulty == "medium":
        return ["textbook_icon", "realistic_device"][index % 2]
    if topology == "cluster_bridge":
        return "textbook_icon"
    return ["textbook_icon", "realistic_device"][index % 2]


def build_nodes(difficulty: str, topology: str, asset_style: str, orientation: str, rng: random.Random):
    if topology == "foodweb_tier":
        tier_counts = [4, rng.randint(6, 7), 5] if difficulty == "difficult" else [3, rng.randint(4, 5), 3]
        labels = rng.sample(load_labels(), sum(tier_counts))
    else:
        labels = rng.sample(load_labels(), rng.randint(*NODE_COUNTS[difficulty]))
    asset_index = load_asset_index()
    nodes = []

    if topology == "star":
        centers = [(500, 350)] + radial_points(len(labels) - 1, 500, 350, 355, 235)
    elif topology == "layered":
        count = len(labels)
        top_count = 4 if difficulty == "difficult" else (3 if difficulty == "medium" else 2)
        mid_count = 6 if difficulty == "difficult" else max(4, count // 3)
        bottom_count = count - top_count - mid_count
        row_counts = [top_count, mid_count, bottom_count]
        y_values = [115, 350, 585]
        centers = []
        used = 0
        for row, y in zip(row_counts, y_values):
            remaining = len(labels) - used
            count = min(row, remaining)
            if count <= 0:
                continue
            step = 820 // max(1, count - 1) if count > 1 else 0
            start_x = 90 if count > 1 else 500
            for idx in range(count):
                centers.append((start_x + idx * step, y))
            used += count
    elif topology == "foodweb_tier":
        centers = []
        used = 0
        if orientation in {"top_to_bottom", "bottom_to_top"}:
            y_bases = [95, 330, 565] if orientation == "top_to_bottom" else [565, 330, 95]
            for tier_idx, (count, y) in enumerate(zip(tier_counts, y_bases)):
                step = 860 // count
                for local_idx in range(count):
                    jitter = rng.randint(-10, 10) if difficulty == "difficult" else rng.randint(-16, 16)
                    x_center = 70 + step // 2 + local_idx * step + jitter
                    centers.append((x_center, y + rng.randint(-8, 8)))
                used += count
        else:
            x_bases = [80, 500, 850] if orientation == "left_to_right" else [850, 500, 80]
            for tier_idx, (count, x) in enumerate(zip(tier_counts, x_bases)):
                step = 610 // count
                for local_idx in range(count):
                    jitter = rng.randint(-8, 8) if difficulty == "difficult" else rng.randint(-14, 14)
                    y_center = 42 + step // 2 + local_idx * step + jitter
                    centers.append((x + rng.randint(-8, 8), y_center))
                used += count
    elif topology == "cluster_bridge":
        cluster_centers = [(260, 350), (740, 350)]
        centers = []
        per_cluster = math.ceil(len(labels) / len(cluster_centers))
        for cluster_idx, center in enumerate(cluster_centers):
            remaining = len(labels) - len(centers)
            count = min(per_cluster, remaining)
            if count <= 0:
                continue
            centers.append(center)
            if count > 1:
                centers.extend(radial_points(count - 1, center[0], center[1], 235, 195, start=-math.pi / 2 + cluster_idx * 0.2))
    else:
        centers = [(500, 350)] + radial_points(len(labels) - 1, 500, 350, 355, 235)

    if topology != "foodweb_tier":
        centers = [transform_center(center, orientation) for center in centers]

    for idx, item in enumerate(labels):
        base = item["display"]
        name = display_name(base, rng)
        x, y = as_top_left(centers[idx], difficulty)
        x = max(35, min(890, x))
        y = max(30, min(585, y))
        nodes.append(make_node(f"tmp{idx + 1}", name, x, y, difficulty, choose_asset(item["label"], asset_style, asset_index, rng)))
    return nodes


def reading_order(nodes) -> List:
    return sorted(nodes, key=lambda node: (node.box[1] // 30, node.box[0], node.box[1]))


def reassign_ids(nodes, edges, negatives) -> None:
    mapping = {}
    for idx, node in enumerate(reading_order(nodes), 1):
        mapping[node.id] = f"n{idx}"
    for node in nodes:
        node.id = mapping[node.id]
    for edge in edges:
        edge.source = mapping[edge.source]
        edge.target = mapping[edge.target]
    for negative in negatives:
        negative.source = mapping[negative.source]
        negative.target = mapping[negative.target]


def sample_edge_type(difficulty: str, rng: random.Random) -> str:
    value = rng.random()
    total = 0.0
    for edge_type, prob in EDGE_RATIOS[difficulty]:
        total += prob
        if value <= total:
            return edge_type
    return "undirected"


def route_edge(edge_id: int, source, target, edge_type: str, rng: random.Random, bends: bool):
    start = RENDERER.boundary_point_towards(source.box, target.center)
    end = RENDERER.boundary_point_towards(target.box, source.center)
    if bends:
        if rng.random() < 0.5:
            mid = ((start[0] + end[0]) // 2, start[1])
        else:
            mid = (start[0], (start[1] + end[1]) // 2)
        path = [start, mid, end]
    else:
        path = [start, end]
    return RENDERER.Edge(f"e{edge_id}", source.id, target.id, path, edge_type)


def center_distance(a, b) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest = (sx + t * dx, sy + t * dy)
    return math.hypot(px - closest[0], py - closest[1])


def line_keeps_node_clearance(a, b, nodes, margin: int = 6) -> bool:
    for node in nodes:
        if node.id in {a.id, b.id}:
            continue
        x1, y1, x2, y2 = node.box
        radius = max(x2 - x1, y2 - y1) / 2 + margin
        if point_segment_distance(node.center, a.center, b.center) < radius:
            return False
    return True


def add_edge(edges, seen, a, b, difficulty: str, rng: random.Random, bends: bool = False, *, allow_short: bool = False, avoid_nodes=None, node_margin: int = 6) -> bool:
    key = tuple(sorted([a.id, b.id]))
    if a.id == b.id or key in seen:
        return False
    if not allow_short and center_distance(a, b) < MIN_EDGE_DISTANCE[difficulty]:
        return False
    if avoid_nodes is not None and not line_keeps_node_clearance(a, b, avoid_nodes, margin=node_margin):
        return False
    seen.add(key)
    edges.append(route_edge(len(edges) + 1, a, b, sample_edge_type(difficulty, rng), rng, bends))
    return True


def build_edges(nodes, difficulty: str, topology: str, orientation: str, rng: random.Random):
    edges = []
    seen = set()

    if topology == "star":
        hub = min(nodes, key=lambda node: abs(node.center[0] - 500) + abs(node.center[1] - 350))
        for node in nodes:
            if node is not hub:
                add_edge(edges, seen, hub, node, difficulty, rng)
    elif topology in {"layered", "foodweb_tier"}:
        rows = {}
        horizontal_layers = orientation in {"top_to_bottom", "bottom_to_top"}
        for node in nodes:
            key_value = node.center[1] if horizontal_layers else node.center[0]
            rows.setdefault(round(key_value / 100), []).append(node)
        ordered_rows = [sorted(rows[key], key=lambda node: node.center[0]) for key in sorted(rows)]
        if not horizontal_layers:
            ordered_rows = [sorted(rows[key], key=lambda node: node.center[1]) for key in sorted(rows)]
        if len(ordered_rows) >= 3:
            top, middle, bottom = ordered_rows[0], ordered_rows[1], ordered_rows[2]
        else:
            top, middle, bottom = ordered_rows[0], ordered_rows[-1], []
        if topology == "foodweb_tier":
            for producer in top:
                add_edge(edges, seen, producer, rng.choice(middle), difficulty, rng, bends=False)
            for transit in middle:
                add_edge(edges, seen, transit, rng.choice(bottom), difficulty, rng, bends=False)
            extra_edges = rng.randint(4, 6) if difficulty == "difficult" else rng.randint(2, 4)
            for _ in range(extra_edges):
                if rng.random() < 0.45:
                    source_pool, target_pool = top, middle
                else:
                    source_pool, target_pool = middle, bottom
                candidates = [(src, dst) for src in source_pool for dst in target_pool]
                if candidates:
                    src, dst = rng.choice(candidates)
                    add_edge(edges, seen, src, dst, difficulty, rng, bends=False)
        else:
            for node in middle:
                parent = min(top, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (node.center[0] if horizontal_layers else node.center[1])))
                add_edge(edges, seen, parent, node, difficulty, rng, bends=False)
            for node in bottom:
                parent = min(middle, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (node.center[0] if horizontal_layers else node.center[1])))
                add_edge(edges, seen, parent, node, difficulty, rng, bends=False)
        if difficulty in {"medium", "difficult"} and len(middle) >= 2 and len(bottom) >= 3:
            long_links = 1 if difficulty == "medium" else (1 if topology == "foodweb_tier" else 2)
            candidates = []
            for mid in middle:
                nearest = min(bottom, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (mid.center[0] if horizontal_layers else mid.center[1])))
                far_options = [node for node in bottom if node is not nearest and center_distance(mid, node) >= MIN_EDGE_DISTANCE[difficulty]]
                if far_options:
                    far = max(far_options, key=lambda node: center_distance(mid, node))
                    candidates.append((mid, far))
            rng.shuffle(candidates)
            for mid, far in candidates[:long_links]:
                add_edge(edges, seen, mid, far, difficulty, rng, bends=False, avoid_nodes=nodes)
    elif topology == "cluster_bridge":
        split_axis = 0 if orientation in {"top_to_bottom", "bottom_to_top"} else 1
        groups = sorted(nodes, key=lambda node: node.center[split_axis])
        split_a = len(groups) // 2
        clusters = [groups[:split_a], groups[split_a:]]
        hubs = []
        for cluster in clusters:
            center_x = sum(n.center[0] for n in cluster) / len(cluster)
            center_y = sum(n.center[1] for n in cluster) / len(cluster)
            hub = min(cluster, key=lambda node: abs(node.center[0] - center_x) + abs(node.center[1] - center_y))
            hubs.append(hub)
            for node in cluster:
                if node is not hub:
                    add_edge(edges, seen, hub, node, difficulty, rng)
        for left_hub, right_hub in zip(hubs, hubs[1:]):
            add_edge(edges, seen, left_hub, right_hub, difficulty, rng, bends=False)
        for first, second in zip(clusters, clusters[1:]):
            if difficulty == "difficult" and len(first) > 3 and len(second) > 3 and rng.random() < 0.45:
                left_candidates = sorted(first, key=lambda node: node.center[split_axis], reverse=True)
                right_candidates = sorted(second, key=lambda node: node.center[split_axis])
                for left_alt in left_candidates:
                    for right_alt in right_candidates:
                        if add_edge(edges, seen, left_alt, right_alt, difficulty, rng, bends=False):
                            break
                    else:
                        continue
                    break

    lo, hi = EDGE_RANGES[difficulty]
    target_edges = rng.randint(lo, hi)
    attempts = 0
    while len(edges) < target_edges and attempts < 700:
        if topology == "foodweb_tier":
            candidates = []
            if "top" in locals() and "middle" in locals() and "bottom" in locals():
                candidates.extend((src, dst) for src in top for dst in middle)
                candidates.extend((src, dst) for src in middle for dst in bottom)
            rng.shuffle(candidates)
            for a, b in candidates:
                if center_distance(a, b) >= MIN_EDGE_DISTANCE[difficulty]:
                    add_edge(edges, seen, a, b, difficulty, rng, bends=False)
                    break
            attempts += 1
            continue
        if topology == "star":
            hub = min(nodes, key=lambda node: abs(node.center[0] - 500) + abs(node.center[1] - 350))
            candidates = [node for node in nodes if node is not hub]
            if len(candidates) < 2:
                break
            a, b = rng.sample(candidates, 2)
        else:
            a, b = rng.sample(nodes, 2)
        if center_distance(a, b) >= MIN_EDGE_DISTANCE[difficulty]:
            add_edge(edges, seen, a, b, difficulty, rng, bends=False, avoid_nodes=nodes)
        attempts += 1
    if len(edges) > hi:
        edges = edges[:hi]
        for idx, edge in enumerate(edges, 1):
            edge.id = f"e{idx}"
        seen = {tuple(sorted([edge.source, edge.target])) for edge in edges}
    if topology == "foodweb_tier" and "top" in locals() and "middle" in locals() and "bottom" in locals():
        ensure_no_isolated_foodweb_tiers([top, middle, bottom], edges, seen, difficulty, rng, nodes)
        ensure_no_isolated_cross_layers_from_positions(nodes, edges, seen, difficulty, rng, orientation)
    else:
        ensure_no_isolated_nodes(nodes, edges, seen, difficulty, rng)
    return edges


def position_tiers(nodes, orientation: str):
    horizontal_layers = orientation in {"top_to_bottom", "bottom_to_top"}
    groups = {}
    for node in nodes:
        key_value = node.center[1] if horizontal_layers else node.center[0]
        groups.setdefault(round(key_value / 100), []).append(node)
    tiers = [groups[key] for key in sorted(groups)]
    if len(tiers) <= 3:
        return tiers
    merged = [[], [], []]
    for idx, tier in enumerate(tiers):
        merged[min(2, round(idx * 2 / max(1, len(tiers) - 1)))].extend(tier)
    return [tier for tier in merged if tier]


def ensure_no_isolated_cross_layers_from_positions(nodes, edges, seen, difficulty: str, rng: random.Random, orientation: str) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    tiers = position_tiers(nodes, orientation)
    tier_by_id = {node.id: idx for idx, tier in enumerate(tiers) for node in tier}
    isolated = [node for node in nodes if node.id not in linked]
    for node in isolated:
        tier_idx = tier_by_id.get(node.id)
        if tier_idx is None:
            continue
        candidate_tiers = []
        if tier_idx > 0:
            candidate_tiers.append(tiers[tier_idx - 1])
        if tier_idx + 1 < len(tiers):
            candidate_tiers.append(tiers[tier_idx + 1])
        candidates = [
            other
            for tier in candidate_tiers
            for other in tier
            if tuple(sorted([node.id, other.id])) not in seen
        ]
        if candidates:
            target = min(candidates, key=lambda other: center_distance(node, other))
            add_edge(edges, seen, node, target, difficulty, rng, allow_short=True)


def ensure_no_isolated_foodweb_tiers(tiers, edges, seen, difficulty: str, rng: random.Random, nodes) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    tier_by_id = {node.id: idx for idx, tier in enumerate(tiers) for node in tier}
    isolated = [node for tier in tiers for node in tier if node.id not in linked]
    rng.shuffle(isolated)
    for node in isolated:
        tier_idx = tier_by_id[node.id]
        candidate_tiers = []
        if tier_idx > 0:
            candidate_tiers.append(tiers[tier_idx - 1])
        if tier_idx + 1 < len(tiers):
            candidate_tiers.append(tiers[tier_idx + 1])
        candidates = [
            other
            for tier in candidate_tiers
            for other in tier
            if tuple(sorted([node.id, other.id])) not in seen
        ]
        clear = [other for other in candidates if line_keeps_node_clearance(node, other, nodes)]
        if clear:
            target = min(clear, key=lambda other: center_distance(node, other))
            if add_edge(edges, seen, node, target, difficulty, rng, allow_short=True, avoid_nodes=nodes):
                continue
        elif candidates:
            target = min(candidates, key=lambda other: center_distance(node, other))
            add_edge(edges, seen, node, target, difficulty, rng, allow_short=True)
            continue
        if candidates:
            target = min(candidates, key=lambda other: center_distance(node, other))
            add_edge(edges, seen, node, target, difficulty, rng, allow_short=True)


def ensure_no_isolated_nodes(nodes, edges, seen, difficulty: str, rng: random.Random) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    isolated = [node for node in nodes if node.id not in linked]
    rng.shuffle(isolated)
    for node in isolated:
        candidates = [
            other
            for other in nodes
            if other is not node
            and tuple(sorted([node.id, other.id])) not in seen
            and center_distance(node, other) >= MIN_EDGE_DISTANCE[difficulty]
        ]
        if not candidates:
            continue
        target = min(candidates, key=lambda other: center_distance(node, other))
        add_edge(edges, seen, node, target, difficulty, rng)


def make_negatives(nodes, difficulty: str, rng: random.Random):
    return [], []


def node_names(nodes) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def midpoint(path: Sequence[Point]) -> Point:
    if len(path) >= 3:
        return path[len(path) // 2]
    return ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


def scale_value(value: int, scale: float) -> int:
    return int(round(value * scale))


def scale_point(point: Point, scale: float) -> List[int]:
    return [scale_value(point[0], scale), scale_value(point[1], scale)]


def scale_box(box, scale: float) -> List[int]:
    return [scale_value(coord, scale) for coord in box]


def scale_path(path: Sequence[Point], scale: float) -> List[List[int]]:
    return [scale_point(point, scale) for point in path]


def output_canvas(scale: float) -> Dict[str, int]:
    return {
        "width": scale_value(RENDERER.CANVAS_W, scale),
        "height": scale_value(RENDERER.CANVAS_H, scale),
    }


def choose_image_scale(difficulty: str, nodes, edges, seed: int, sample_id: str) -> float:
    return 0.9


def final_graph(nodes, edges, scale: float = 1.0):
    names = node_names(nodes)
    relation_map = {node.id: [] for node in nodes}
    for edge in edges:
        relation_map[edge.source].append(
            {"source": edge.source, "source_name": names[edge.source], "target": edge.target, "target_name": names[edge.target], "type": edge.edge_type}
        )
        if edge.edge_type in {"undirected", "bidirectional"}:
            relation_map[edge.target].append(
                {"source": edge.target, "source_name": names[edge.target], "target": edge.source, "target_name": names[edge.source], "type": edge.edge_type}
            )
    return {
        "nodes": [
            {"id": node.id, "name": node.label, "box": scale_box(node.box, scale), "relations": relation_map[node.id]}
            for node in reading_order(nodes)
        ]
    }


def expanded_edges_for_reasoning(edges: Sequence) -> List:
    expanded = []
    for edge in edges:
        expanded.append(edge)
        if edge.edge_type in {"undirected", "bidirectional"}:
            reverse = RENDERER.Edge(edge.id + "_rev", edge.target, edge.source, RENDERER.reversed_path(edge.path), edge.edge_type)
            expanded.append(reverse)
    return expanded


def edge_record(edge, names: Dict[str, str], scale: float = 1.0):
    visible = RENDERER.visible_edge_path(edge, clearance=NETWORK_EDGE_CLEARANCE)
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
        "arrow": scale_point(visible[-1], scale) if edge.edge_type == "directed" else None,
        "arrowheads": {
            "target": [scale_point(point, scale) for point in RENDERER.arrowhead_points(visible)]
            if edge.edge_type == "directed"
            else [],
            "source": [],
        },
    }


def think_graph(nodes, edges, negatives, scale: float = 1.0) -> str:
    names = node_names(nodes)
    outgoing = {node.id: [] for node in nodes}
    hard_negatives = {node.id: [] for node in nodes}
    for edge in expanded_edges_for_reasoning(edges):
        outgoing[edge.source].append(edge)
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


def render_style_record(style) -> Dict:
    return {
        "edge_color": list(style.edge_color),
        "edge_width": style.edge_width,
        "arrow_size": style.arrow_size,
        "label_color": list(style.label_color),
        "background": list(style.background),
        "grid": style.grid,
        "halo": style.halo,
        "edge_clearance": NETWORK_EDGE_CLEARANCE,
    }


def draw_node_image_layer(base: Image.Image, draw: ImageDraw.ImageDraw, node) -> None:
    if node.shape == "image_label" and node.icon_box:
        if node.asset_path and Path(node.asset_path).exists():
            RENDERER.paste_asset_patch(base, node.asset_path, node.icon_box)
        else:
            RENDERER.draw_image_patch(draw, node.label, node.icon_box)


def draw_node_label_layer(draw: ImageDraw.ImageDraw, node, font) -> None:
    tx1, ty1, tx2, ty2 = node.text_box
    bbox = draw.textbbox((0, 0), node.label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = 3
    pad_y = 2
    x = tx1 + max(0, (tx2 - tx1 - text_w) // 2)
    y = ty1 + max(0, (ty2 - ty1 - text_h) // 2)
    bg = [max(tx1, x - pad_x), max(ty1, y - pad_y), min(tx2, x + text_w + pad_x), min(ty2, y + text_h + pad_y)]
    draw.rounded_rectangle(bg, radius=2, fill=(255, 255, 255), outline=None)
    draw.text((x, y), node.label, fill=(20, 28, 38), font=font)


def draw_network_edge(draw: ImageDraw.ImageDraw, edge, style) -> None:
    path = RENDERER.visible_edge_path(edge, clearance=NETWORK_EDGE_CLEARANCE)
    if style.halo:
        draw.line(path, fill=(255, 255, 255), width=style.edge_width + 3, joint="curve")
    draw.line(path, fill=style.edge_color, width=style.edge_width, joint="curve")


def draw_network_edge_arrowheads(draw: ImageDraw.ImageDraw, edge, style) -> None:
    if edge.edge_type not in {"directed", "bidirectional"}:
        return
    path = RENDERER.visible_edge_path(edge, clearance=NETWORK_EDGE_CLEARANCE)
    RENDERER.draw_arrowhead(draw, path, style.edge_color, style.arrow_size, style.halo)
    if edge.edge_type == "bidirectional":
        RENDERER.draw_arrowhead(draw, RENDERER.reversed_path(path), style.edge_color, style.arrow_size, style.halo)


def node_record(node, scale: float = 1.0):
    return {
        "id": node.id,
        "name": node.label,
        "box": scale_box(node.box, scale),
        "center": scale_point(node.center, scale),
        "image_box": scale_box(node.icon_box, scale) if node.icon_box else None,
        "text_box": scale_box(node.text_box, scale),
        "asset": Path(node.asset_path).name if node.asset_path else None,
    }


def render_image(name: str, nodes, edges, extra_paths, out_path: Path, seed: int, scale: float = 1.0):
    rng = random.Random(f"{seed}:{name}:render")
    style = RENDERER.style_for_case(name, seed, ai2d_like=True)
    style.edge_color = rng.choice(NETWORK_EDGE_COLORS)
    style.edge_width = 2 if name.startswith("simple") else rng.choice([2, 3])
    style.arrow_size = 11 if name.startswith("difficult") else 14
    image = Image.new("RGB", (RENDERER.CANVAS_W, RENDERER.CANVAS_H), style.background)
    draw = ImageDraw.Draw(image)
    if style.grid:
        RENDERER.draw_grid(draw)
    font = RENDERER.load_font(11 if name.startswith("difficult") else 12)
    for node in nodes:
        draw_node_image_layer(image, draw, node)
    for path in extra_paths:
        RENDERER.draw_near_miss(draw, path)
    for edge in edges:
        draw_network_edge(draw, edge, style)
    for edge in edges:
        draw_network_edge_arrowheads(draw, edge, style)
    for node in nodes:
        draw_node_label_layer(draw, node, font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = output_canvas(scale)
    if scale != 1.0:
        image = image.resize((canvas["width"], canvas["height"]), Image.Resampling.LANCZOS)
    image.save(out_path)
    record = render_style_record(style)
    record["image_scale"] = scale
    record["canonical_canvas"] = {"width": RENDERER.CANVAS_W, "height": RENDERER.CANVAS_H}
    record["output_canvas"] = canvas
    return record


def make_contact_sheet(image_paths: List[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 252))
        thumbs.append((path.stem, image.copy()))
    font = RENDERER.load_font(16)
    cols = 2
    cell_w = 500
    cell_h = 300
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (title, image) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 18
        y = (idx // cols) * cell_h + 38
        draw.text((x, y - 24), title, fill=(28, 35, 45), font=font)
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + image.width, y + image.height], outline=(210, 215, 222), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_sample(difficulty: str, index: int, seed: int):
    rng = random.Random(f"{seed}:{difficulty}:{index}")
    if difficulty == "difficult":
        topology = ["layered", "cluster_bridge", "foodweb_tier", "foodweb_tier"][(index - 1) % 4]
    elif difficulty == "simple":
        topology = ["star", "layered"][(index - 1) % 2]
    else:
        topology = TOPOLOGIES[difficulty][0]
    orientation = rng.choice(ORIENTATIONS)
    asset_style = style_for_sample(difficulty, topology, index, rng)
    nodes = build_nodes(difficulty, topology, asset_style, orientation, rng)
    edges = build_edges(nodes, difficulty, topology, orientation, rng)
    negatives, extra_paths = make_negatives(nodes, difficulty, rng)
    reassign_ids(nodes, edges, negatives)
    return nodes, edges, negatives, extra_paths, topology, orientation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_network_v1/outputs/preview_samples")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--per-difficulty", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    image_paths = []
    summary = []
    for difficulty in ["simple", "medium", "difficult"]:
        for idx in range(1, args.per_difficulty + 1):
            sample_id = f"network_{difficulty}_{idx:03d}"
            nodes, edges, negatives, extra_paths, topology, orientation = make_sample(difficulty, idx, args.seed)
            image_path = out / "images" / difficulty / f"{sample_id}.png"
            annotation_path = out / "annotations" / difficulty / f"{sample_id}.json"
            render_style = render_image(f"{difficulty}_{sample_id}", nodes, edges, extra_paths, image_path, args.seed)
            names = node_names(nodes)
            annotation = {
                "sample_id": sample_id,
                "image": str(image_path.relative_to(out)),
                "split": "preview",
                "difficulty": difficulty,
                "orientation": orientation,
                "canvas": {"width": RENDERER.CANVAS_W, "height": RENDERER.CANVAS_H},
                "nodes": [node_record(node) for node in reading_order(nodes)],
                "edges": [edge_record(edge, names) for edge in edges],
                "negative_edges": [
                    {
                        "source": negative.source,
                        "source_name": names[negative.source],
                        "target": negative.target,
                        "target_name": names[negative.target],
                        "reason": negative.reason,
                        "evidence_point": list(negative.evidence_point),
                    }
                    for negative in negatives
                ],
                "final_graph": final_graph(nodes, edges),
                "think_graph": think_graph(nodes, edges, negatives),
                "training_text": training_text(nodes, edges, negatives),
                "render_style": render_style,
                "generation": {"seed": args.seed, "index": idx, "domain": "network", "topology": topology},
            }
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            image_paths.append(image_path)
            summary.append(
                {
                    "sample_id": sample_id,
                    "difficulty": difficulty,
                    "topology": topology,
                    "orientation": orientation,
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "negative_edges": len(negatives),
                    "image": str(image_path.relative_to(out)),
                    "annotation": str(annotation_path.relative_to(out)),
                    "edge_types": sorted({edge.edge_type for edge in edges}),
                }
            )
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(image_paths, out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
