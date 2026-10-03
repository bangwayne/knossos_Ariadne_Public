#!/usr/bin/env python3


from __future__ import annotations

import argparse
import colorsys
import importlib.util
import json
import math
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RENDERER_PATH = REPO / "knossos_foodweb_v0" / "scripts" / "renderer.py"
LABELS_PATH = ROOT / "config" / "natural_process_labels.json"
ASSET_META_PATH = ROOT / "assets" / "renderer_asset_pool" / "metadata.json"
ASSET_POOL = ROOT / "assets" / "renderer_asset_pool"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDERER = load_module("knossos_natural_renderer_base", RENDERER_PATH)

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

CANVAS_W = RENDERER.CANVAS_W
CANVAS_H = RENDERER.CANVAS_H
EDGE_CLEARANCE = 12

SCALE_BY_DIFFICULTY = {
    "simple": (0.80, 0.86, 0.92, 0.98),
    "medium": (0.92, 0.98, 1.00, 1.06),
    "difficult": (1.00, 1.06, 1.12, 1.18),
}

STYLE_MODES = ["eco_flat_icon", "natural_scene_icon", "mixed"]
RELATION_STEMS = [
    "flows_to",
    "transforms_into",
    "supports",
    "absorbs",
    "releases",
    "produces",
    "decomposes_to",
    "cycles_to",
    "deposits_into",
    "evaporates_to",
    "feeds",
    "inhibits",
    "filters",
    "carries",
    "enriches",
    "cools",
]

RELATION_PROFILE_SCHEDULES = {
    "simple": ["single", "typed", "single", "single", "typed", "single", "typed", "single", "typed", "single"],
    "medium": ["single", "typed", "single", "multi", "typed", "single", "multi", "single", "typed", "single"],
    "difficult": ["single", "typed", "multi", "single", "typed", "multi", "single", "typed", "multi", "single"],
}


@dataclass
class NaturalNode:
    id: str
    label: str
    box: BBox
    role: str
    category: str
    asset_path: str
    render_mode: str

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def icon_box(self) -> BBox:
        x1, y1, x2, y2 = self.box
        if self.render_mode == "text_only":
            return (x1 + 3, y1 + 3, x2 - 3, y2 - 3)
        return (x1 + 5, y1 + 3, x2 - 5, y2 - 24)

    @property
    def text_box(self) -> BBox | None:
        x1, _y1, x2, y2 = self.box
        if self.render_mode == "text_only":
            return (x1 + 4, y2 - 18, x2 - 4, y2 - 4)
        return (x1, y2 - 23, x2, y2)


@dataclass
class NaturalEdge:
    id: str
    source: str
    target: str
    path: List[Point]
    edge_type: str
    directed: bool = True


def edge_pair(edge: NaturalEdge) -> Tuple[str, str]:
    return (edge.source, edge.target)


def load_labels() -> List[Dict]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))["labels"]


def labels_by_category() -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for item in load_labels():
        groups.setdefault(item["category"], []).append(item)
        groups.setdefault(item["role"], []).append(item)
    return groups


def asset_lookup() -> Dict[Tuple[str, str], List[Path]]:
    meta = json.loads(ASSET_META_PATH.read_text(encoding="utf-8"))
    lookup: Dict[Tuple[str, str], List[Path]] = {}
    for filename, item in meta.items():
        lookup.setdefault((item["label"], item["style"]), []).append(ASSET_POOL / filename)
    for paths in lookup.values():
        paths.sort()
    return lookup


def choose_item(pool: Sequence[str], rng: random.Random, used: set[str]) -> Dict:
    groups = labels_by_category()
    options = []
    for key in pool:
        options.extend(groups.get(key, []))
    dedup = {item["label"]: item for item in options}
    available = [item for label, item in dedup.items() if label not in used] or list(dedup.values())
    item = rng.choice(available)
    used.add(item["label"])
    return item


def display_name(item: Dict, rng: random.Random) -> str:
    name = item["display"]
    if rng.random() < 0.28 and item["render_mode"] != "text_only":
        return f"{name} {rng.choice(['A', 'B', 'C', 'N', 'S', '1', '2', '3'])}"
    return name


def asset_for(item: Dict, style_mode: str, rng: random.Random, lookup: Dict[Tuple[str, str], List[Path]]) -> Path:
    if item["render_mode"] == "text_only":
        style = "text_card"
    elif style_mode == "mixed":
        style = rng.choice(["eco_flat_icon", "natural_scene_icon"])
    else:
        style = style_mode
    options = lookup.get((item["label"], style), [])
    if not options:
        fallback = [path for (label, _style), paths in lookup.items() if label == item["label"] for path in paths]
        options = fallback
    if not options:
        raise FileNotFoundError(f"no asset for {item['label']}")
    return rng.choice(options)


def node_size(difficulty: str, render_mode: str) -> Tuple[int, int]:
    if render_mode == "text_only":
        return {"simple": (106, 66), "medium": (98, 62), "difficult": (84, 56)}[difficulty]
    return {"simple": (98, 108), "medium": (88, 98), "difficult": (72, 82)}[difficulty]


def make_node(node_id: str, item: Dict, center: Point, difficulty: str, style_mode: str, rng: random.Random, lookup) -> NaturalNode:
    w, h = node_size(difficulty, item["render_mode"])
    x, y = center
    asset = asset_for(item, style_mode, rng, lookup)
    return NaturalNode(
        f"tmp{node_id}",
        display_name(item, rng),
        (x - w // 2, y - h // 2, x + w // 2, y + h // 2),
        item["role"],
        item["category"],
        str(asset.relative_to(ROOT)),
        item["render_mode"],
    )


def tier_counts(difficulty: str, rng: random.Random) -> Tuple[List[int], int, int, bool]:
    if difficulty == "simple":
        return [2, rng.randint(2, 3), rng.randint(2, 3), 1], rng.randint(1, 2), rng.randint(0, 1), rng.random() < 0.20
    if difficulty == "medium":
        return [rng.randint(2, 3), rng.randint(3, 4), rng.randint(3, 4), rng.randint(2, 3)], rng.randint(2, 4), rng.randint(1, 2), True
    return [rng.randint(2, 3), rng.randint(3, 4), rng.randint(3, 4), rng.randint(3, 4), rng.randint(2, 3)], rng.randint(3, 5), rng.randint(2, 3), True


def layer_base_positions(layer_count: int, orientation: str) -> List[int]:
    if orientation in {"bottom_to_top", "top_to_bottom"}:
        start, end = 92, 608
        positions = [int(round(start + idx * (end - start) / max(1, layer_count - 1))) for idx in range(layer_count)]
        return list(reversed(positions)) if orientation == "bottom_to_top" else positions
    start, end = 78, 922
    positions = [int(round(start + idx * (end - start) / max(1, layer_count - 1))) for idx in range(layer_count)]
    return list(reversed(positions)) if orientation == "right_to_left" else positions


def center_for(tier_idx: int, local_idx: int, count: int, layer_count: int, orientation: str, difficulty: str, rng: random.Random) -> Point:
    bases = layer_base_positions(layer_count, orientation)
    jitter = 4 if difficulty == "difficult" else 14
    if orientation in {"bottom_to_top", "top_to_bottom"}:
        usable_w = 850
        step = usable_w // count
        x = 70 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
        return (x, bases[tier_idx] + rng.randint(-5, 5))
    usable_h = 590
    step = usable_h // count
    y = 55 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
    return (bases[tier_idx] + rng.randint(-5, 5), y)


def expanded_box(box: BBox, margin: int = 12) -> BBox:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def point_in_box(point: Point, box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def on_segment(a: Point, b: Point, c: Point) -> bool:
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def segment_intersects_box(a: Point, b: Point, box: BBox) -> bool:
    if point_in_box(a, box) or point_in_box(b, box):
        return True
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    sides = list(zip(corners, corners[1:] + corners[:1]))
    return any(segments_intersect(a, b, c, d) for c, d in sides)


def path_block_count(path: Sequence[Point], blockers: Sequence[NaturalNode]) -> int:
    count = 0
    for a, b in zip(path, path[1:]):
        for node in blockers:
            if segment_intersects_box(a, b, expanded_box(node.box)):
                count += 1
    return count


def path_length(path: Sequence[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def choose_detour_path(start: Point, end: Point, blockers: Sequence[NaturalNode], rng: random.Random) -> List[Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    candidates: List[List[Point]] = []
    if abs(dx) >= abs(dy):
        related = [node for node in blockers if min(start[0], end[0]) - 40 <= node.center[0] <= max(start[0], end[0]) + 40]
        y_values = [node.box[1] for node in related] + [start[1], end[1]]
        y_top = max(32, min(y_values) - rng.choice([42, 56, 70]))
        y_values = [node.box[3] for node in related] + [start[1], end[1]]
        y_bottom = min(CANVAS_H - 32, max(y_values) + rng.choice([42, 56, 70]))
        candidates.extend([[start, (start[0], y_top), (end[0], y_top), end], [start, (start[0], y_bottom), (end[0], y_bottom), end]])
    else:
        related = [node for node in blockers if min(start[1], end[1]) - 40 <= node.center[1] <= max(start[1], end[1]) + 40]
        x_values = [node.box[0] for node in related] + [start[0], end[0]]
        x_left = max(32, min(x_values) - rng.choice([42, 56, 70]))
        x_values = [node.box[2] for node in related] + [start[0], end[0]]
        x_right = min(CANVAS_W - 32, max(x_values) + rng.choice([42, 56, 70]))
        candidates.extend([[start, (x_left, start[1]), (x_left, end[1]), end], [start, (x_right, start[1]), (x_right, end[1]), end]])
    return min(candidates, key=lambda path: (path_block_count(path, blockers), path_length(path)))


def is_long_same_lane(src: NaturalNode, dst: NaturalNode) -> bool:
    dx = abs(dst.center[0] - src.center[0])
    dy = abs(dst.center[1] - src.center[1])
    same_row = dy <= 72 and dx >= 220
    same_col = dx <= 72 and dy >= 220
    return same_row or same_col


def route_path(src: NaturalNode, dst: NaturalNode, rng: random.Random, bends: bool, nodes: Sequence[NaturalNode] | None = None) -> List[Point]:
    start = RENDERER.boundary_point_towards(src.box, dst.center)
    end = RENDERER.boundary_point_towards(dst.box, src.center)
    blockers = [node for node in (nodes or []) if node.id not in {src.id, dst.id}]
    if not bends:
        path = [start, end]
    else:
        mid = ((start[0] + end[0]) // 2 + rng.randint(-28, 28), (start[1] + end[1]) // 2 + rng.randint(-16, 16))
        path = [start, mid, end]
    if blockers and is_long_same_lane(src, dst) and path_block_count(path, blockers) > 0:
        return choose_detour_path(start, end, blockers, rng)
    return path


def offset_path(path: Sequence[Point], offset: float) -> List[Point]:
    if not path or abs(offset) < 0.1:
        return [tuple(p) for p in path]
    x1, y1 = path[0]
    x2, y2 = path[-1]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    return [(int(round(x + nx * offset)), int(round(y + ny * offset))) for x, y in path]


def translate_box(box: BBox, dx: int, dy: int) -> BBox:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def translate_path(path: Sequence[Point], dx: int, dy: int) -> List[Point]:
    return [(x + dx, y + dy) for x, y in path]


def translate_case(nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], dx: int, dy: int) -> None:
    for node in nodes:
        node.box = translate_box(node.box, dx, dy)
    for edge in edges:
        edge.path = translate_path(edge.path, dx, dy)


def add_edge(edges: List[NaturalEdge], seen: set[Tuple[str, str]], src: NaturalNode, dst: NaturalNode, rng: random.Random, bends: bool, directed: bool = True, nodes: Sequence[NaturalNode] | None = None) -> None:
    if src.id == dst.id or (src.id, dst.id) in seen:
        return
    seen.add((src.id, dst.id))
    edges.append(NaturalEdge(f"e{len(edges) + 1}", src.id, dst.id, route_path(src, dst, rng, bends, nodes), "process", directed))


def ensure_no_isolated(nodes: Sequence[NaturalNode], edges: List[NaturalEdge], rng: random.Random, bends: bool) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    seen = {(edge.source, edge.target) for edge in edges}
    for node in nodes:
        if node.id in linked:
            continue
        target = min([other for other in nodes if other is not node], key=lambda other: math.hypot(node.center[0] - other.center[0], node.center[1] - other.center[1]))
        add_edge(edges, seen, target, node, rng, bends, nodes=nodes)
        linked.update([target.id, node.id])


def add_multilinks(edges: List[NaturalEdge], difficulty: str, rng: random.Random) -> None:
    if not edges:
        return
    pair_counts: Dict[Tuple[str, str], int] = {}
    candidates = []
    for edge in edges:
        pair = edge_pair(edge)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if edge.directed:
            candidates.append(edge)
    rng.shuffle(candidates)
    pair_budget = {"simple": 1, "medium": rng.randint(1, 2), "difficult": rng.randint(3, 4)}[difficulty]
    for base in candidates[:pair_budget]:
        copies = 1
        if difficulty == "difficult" and rng.random() < 0.35:
            copies = 2
        family = [base]
        for _ in range(copies):
            family.append(NaturalEdge("", base.source, base.target, list(base.path), base.edge_type, base.directed))
        offsets = {2: [-7, 7], 3: [-11, 0, 11]}[len(family)]
        for edge, offset in zip(family, offsets):
            edge.path = offset_path(edge.path, offset)
        edges.extend(family[1:])


def build_case(style_mode: str, index: int, seed: int, difficulty: str, relation_profile: str):
    rng = random.Random(f"{seed}:{difficulty}:{index}:{style_mode}")
    orientation = rng.choice(["bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"])
    counts, extra_edges, reversible_edges, bends = tier_counts(difficulty, rng)
    pools = [
        ["energy", "climate", "water", "earth", "organic", "producer", "chemical"],
        ["water", "earth", "organic", "chemical", "flow"],
        ["organism", "producer", "decomposer", "process", "chemical"],
        ["process", "flow", "chemical", "organism", "consumer"],
        ["water", "earth", "organic", "chemical", "organism", "process", "consumer"],
    ]
    lookup = asset_lookup()
    used: set[str] = set()
    nodes: List[NaturalNode] = []
    by_tier: List[List[NaturalNode]] = []
    node_idx = 1
    for tier_idx, count in enumerate(counts):
        tier_nodes = []
        for local_idx in range(count):
            item = choose_item(pools[tier_idx % len(pools)], rng, used)
            node = make_node(str(node_idx), item, center_for(tier_idx, local_idx, count, len(counts), orientation, difficulty, rng), difficulty, style_mode, rng, lookup)
            nodes.append(node)
            tier_nodes.append(node)
            node_idx += 1
        by_tier.append(tier_nodes)

    edges: List[NaturalEdge] = []
    seen: set[Tuple[str, str]] = set()

    for tier_idx in range(len(by_tier) - 1):
        current = by_tier[tier_idx]
        next_tier = by_tier[tier_idx + 1]
        for src_idx, src in enumerate(current):
            mapped = round(src_idx * (len(next_tier) - 1) / max(1, len(current) - 1))
            dst_idx = max(0, min(len(next_tier) - 1, mapped + rng.choice([-1, 0, 0, 1])))
            add_edge(edges, seen, src, next_tier[dst_idx], rng, bends, nodes=nodes)
            if difficulty != "simple" and len(next_tier) > 1 and rng.random() < 0.28:
                adjacent_idx = max(0, min(len(next_tier) - 1, dst_idx + rng.choice([-1, 1])))
                if adjacent_idx != dst_idx:
                    add_edge(edges, seen, src, next_tier[adjacent_idx], rng, bends, nodes=nodes)

        current_ids = {node.id for node in current}
        for dst_idx, dst in enumerate(next_tier):
            has_incoming = any(edge.target == dst.id and edge.source in current_ids for edge in edges)
            if not has_incoming:
                mapped = round(dst_idx * (len(current) - 1) / max(1, len(next_tier) - 1))
                src = current[max(0, min(len(current) - 1, mapped))]
                add_edge(edges, seen, src, dst, rng, bends, nodes=nodes)

    for _ in range(extra_edges):
        tier_idx = rng.randrange(0, len(by_tier) - 1)
        src_tier = by_tier[tier_idx]
        dst_tier = by_tier[tier_idx + 1]
        src_idx = rng.randrange(len(src_tier))
        mapped = round(src_idx * (len(dst_tier) - 1) / max(1, len(src_tier) - 1))
        dst_idx = max(0, min(len(dst_tier) - 1, mapped + rng.choice([-1, 0, 1])))
        add_edge(edges, seen, src_tier[src_idx], dst_tier[dst_idx], rng, bends, nodes=nodes)

    for _ in range(reversible_edges):
        tier = rng.choice([items for items in by_tier if len(items) >= 2])
        idx = rng.randrange(len(tier) - 1)
        a, b = tier[idx], tier[idx + 1]
        if rng.random() < 0.5:
            a, b = b, a
        add_edge(edges, seen, a, b, rng, bends=False, directed=True, nodes=nodes)

    if relation_profile == "multi":
        add_multilinks(edges, difficulty, rng)
    reassign_ids(nodes, edges)
    for edge_idx, edge in enumerate(edges, 1):
        edge.id = f"e{edge_idx}"
    return nodes, edges, orientation


def reading_order(nodes: Sequence[NaturalNode]) -> List[NaturalNode]:
    return sorted(nodes, key=lambda n: (n.box[1] // 30, n.box[0], n.box[1]))


def reassign_ids(nodes: List[NaturalNode], edges: List[NaturalEdge]) -> None:
    mapping = {node.id: f"n{idx}" for idx, node in enumerate(reading_order(nodes), 1)}
    for node in nodes:
        node.id = mapping[node.id]
    for edge in edges:
        edge.source = mapping[edge.source]
        edge.target = mapping[edge.target]


def distinct_colors(count: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    start = rng.random()
    colors = []
    for idx in range(count):
        hue = (start + idx / max(1, count)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.58)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    rng.shuffle(colors)
    return colors


def relation_styles(edges: List[NaturalEdge], sample_id: str, seed: int, difficulty: str, relation_profile: str) -> Dict[str, Dict]:
    rng = random.Random(f"{seed}:{sample_id}:relations")
    if relation_profile == "single":
        type_count = 1
    elif relation_profile == "typed":
        type_count = {"simple": 2, "medium": rng.randint(2, 3), "difficult": rng.randint(3, 4)}[difficulty]
    else:
        type_count = {"simple": 2, "medium": rng.randint(3, 4), "difficult": rng.randint(4, 5)}[difficulty]
    type_count = min(type_count, len(RELATION_STEMS), max(1, len(edges)))
    labels = []
    while len(labels) < type_count:
        label = rng.choice(RELATION_STEMS)
        if label not in labels:
            labels.append(label)
    colors = distinct_colors(type_count, rng)
    styles = {}
    for label in labels:
        styles[label] = {"label": label, "color": colors.pop(), "width": rng.choice([2, 2, 3, 3, 4]), "arrow": True}
    used_by_pair: Dict[Tuple[str, str], set[str]] = {}
    for edge in edges:
        pair = edge_pair(edge)
        used = used_by_pair.setdefault(pair, set())
        options = [label for label in labels if label not in used] or labels
        edge.edge_type = rng.choice(options)
        used.add(edge.edge_type)
        edge.directed = True
    return styles


def visible_edge_path(edge: NaturalEdge) -> List[Point]:
    path = list(edge.path)
    path[0] = RENDERER.offset_point(path[0], path[1], EDGE_CLEARANCE)
    path[-1] = RENDERER.offset_point(path[-1], path[-2], EDGE_CLEARANCE)
    return path


def midpoint(path: Sequence[Point]) -> Point:
    return path[len(path) // 2] if len(path) >= 3 else ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


def scale_value(value: int, scale: float) -> int:
    return int(round(value * scale))


def scale_point(point: Point, scale: float) -> List[int]:
    return [scale_value(point[0], scale), scale_value(point[1], scale)]


def scale_box(box: BBox, scale: float) -> List[int]:
    return [scale_value(coord, scale) for coord in box]


def scale_path(path: Sequence[Point], scale: float) -> List[List[int]]:
    return [scale_point(point, scale) for point in path]


def output_canvas(scale: float) -> Dict[str, int]:
    return {"width": scale_value(CANVAS_W, scale), "height": scale_value(CANVAS_H, scale)}


def choose_image_scale(difficulty: str, nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], seed: int, sample_id: str) -> float:
    return 0.9


def node_names(nodes: Sequence[NaturalNode]) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def edge_record(edge: NaturalEdge, names: Dict[str, str], scale: float = 1.0) -> Dict:
    visible = visible_edge_path(edge)
    arrow = scale_point(visible[-1], scale) if edge.directed else None
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
        "arrow": arrow,
        "arrowheads": {"target": [scale_point(point, scale) for point in RENDERER.arrowhead_points(visible, size=13)] if arrow else [], "source": []},
    }


def node_record(node: NaturalNode, scale: float = 1.0) -> Dict:
    return {
        "id": node.id,
        "name": node.label,
        "box": scale_box(node.box, scale),
        "center": scale_point(node.center, scale),
        "image_box": scale_box(node.icon_box, scale),
        "text_box": scale_box(node.text_box, scale) if node.text_box else None,
        "asset": Path(node.asset_path).name,
    }


def final_graph(nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], scale: float = 1.0) -> Dict:
    names = node_names(nodes)
    relation_map = {node.id: [] for node in nodes}
    for edge in edges:
        relation_map[edge.source].append({"source": edge.source, "source_name": names[edge.source], "target": edge.target, "target_name": names[edge.target], "type": edge.edge_type})
    return {"nodes": [{"id": node.id, "name": node.label, "box": scale_box(node.box, scale), "relations": relation_map[node.id]} for node in reading_order(nodes)]}


def think_graph(nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], negatives, scale: float = 1.0) -> str:
    names = node_names(nodes)
    outgoing = {node.id: [] for node in nodes}
    for edge in edges:
        outgoing[edge.source].append(edge)
    lines = ["<THINK_GRAPH>", "<NODES>"]
    for node in reading_order(nodes):
        lines.append(f'<NODE id="{node.id}" name="{node.label}" box="{scale_box(node.box, scale)}"/>')
    lines.extend(["</NODES>", ""])
    for node in reading_order(nodes):
        lines.append(f'<CHECK_NODE id="{node.id}" name="{node.label}">')
        for edge in outgoing[node.id]:
            rec = edge_record(edge, names, scale)
            lines.append(f'<TRACE target="{rec["target"]}" target_name="{rec["target_name"]}" start="{rec["start"]}" mid="{rec["mid"]}" end="{rec["end"]}" arrow="{rec["arrow"]}"/>')
            lines.append(f'<EDGE source="{rec["source"]}" source_name="{rec["source_name"]}" target="{rec["target"]}" target_name="{rec["target_name"]}" type="{rec["type"]}"/>')
        lines.extend(["</CHECK_NODE>", ""])
    lines.append("</THINK_GRAPH>")
    return "\n".join(lines)


def training_text(nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], negatives, scale: float = 1.0) -> str:
    return think_graph(nodes, edges, negatives, scale) + "\n\n<FINAL_GRAPH>\n" + json.dumps(final_graph(nodes, edges, scale), indent=2) + "\n</FINAL_GRAPH>"


def render_style(sample_id: str, seed: int) -> Dict:
    rng = random.Random(f"{seed}:{sample_id}:style")
    return {
        "background": list(rng.choice([(255, 255, 255), (247, 251, 246), (246, 250, 255), (252, 248, 240)])),
        "grid": rng.random() < 0.35,
        "halo": True,
        "edge_width": 3,
        "arrow_size": 13,
    }


def legend_position(sample_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:{sample_id}:legend")
    return rng.choice(["top_left", "bottom_right"])


def draw_background(draw: ImageDraw.ImageDraw, style: Dict) -> None:
    if not style["grid"]:
        return
    color = (232, 238, 232)
    for x in range(0, CANVAS_W + 1, 50):
        draw.line([(x, 0), (x, CANVAS_H)], fill=color, width=1)
    for y in range(0, CANVAS_H + 1, 50):
        draw.line([(0, y), (CANVAS_W, y)], fill=color, width=1)


def wrap_label(text: str, width: int) -> List[str]:
    return (textwrap.wrap(text, width=max(8, width // 8)) or [""])[:2]


def draw_node(image: Image.Image, draw: ImageDraw.ImageDraw, node: NaturalNode, font) -> None:
    asset = Image.open(ROOT / node.asset_path).convert("RGBA")
    box = node.icon_box
    asset.thumbnail((box[2] - box[0], box[3] - box[1]))
    x = box[0] + ((box[2] - box[0]) - asset.width) // 2
    y = box[1] + ((box[3] - box[1]) - asset.height) // 2
    image.paste(asset, (x, y), asset if asset.mode == "RGBA" else None)
    if node.text_box:
        tx1, ty1, tx2, ty2 = node.text_box
        lines = wrap_label(node.label, tx2 - tx1)
        line_h = 14
        y_text = ty1 + max(0, (ty2 - ty1 - line_h * len(lines)) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((tx1 + max(0, (tx2 - tx1 - (bbox[2] - bbox[0])) // 2), y_text), line, fill=(26, 32, 40), font=font)
            y_text += line_h


def draw_edge(draw: ImageDraw.ImageDraw, edge: NaturalEdge, styles: Dict[str, Dict]) -> None:
    style = styles[edge.edge_type]
    path = visible_edge_path(edge)
    draw.line(path, fill=(255, 255, 255), width=style["width"] + 4)
    draw.line(path, fill=style["color"], width=style["width"], joint="curve")


def draw_arrow(draw: ImageDraw.ImageDraw, edge: NaturalEdge, styles: Dict[str, Dict]) -> None:
    if not edge.directed:
        return
    path = visible_edge_path(edge)
    draw.polygon(RENDERER.arrowhead_points(path, size=13), fill=styles[edge.edge_type]["color"])


def draw_legend(draw: ImageDraw.ImageDraw, font, styles: Dict[str, Dict], position: str) -> None:
    labels = list(styles)
    col_w = 210
    row_h = 32
    cols = min(2, len(labels))
    rows = math.ceil(len(labels) / cols)
    box_w = cols * col_w + 26
    box_h = rows * row_h + 40
    if position == "bottom_right":
        x = CANVAS_W - box_w + 14
        y = CANVAS_H - box_h + 16
    else:
        x, y = 34, 30
    draw.rounded_rectangle((x - 14, y - 16, x + cols * col_w + 12, y - 16 + rows * row_h + 24), radius=7, fill=(255, 255, 255), outline=(190, 198, 210), width=2)
    for idx, label in enumerate(labels):
        style = styles[label]
        x0 = x + (idx % cols) * col_w
        y0 = y + (idx // cols) * row_h
        legend_width = max(style["width"] + 1, 4)
        draw.line([(x0, y0 + 11), (x0 + 48, y0 + 11)], fill=style["color"], width=legend_width)
        if style["arrow"]:
            draw.polygon(RENDERER.arrowhead_points([(x0, y0 + 11), (x0 + 48, y0 + 11)], size=11), fill=style["color"])
        draw.text((x0 + 60, y0), label, fill=(18, 24, 32), font=font)


def render_image(sample_id: str, nodes: Sequence[NaturalNode], edges: Sequence[NaturalEdge], image_path: Path, seed: int, styles: Dict[str, Dict], legend_pos: str, scale: float = 1.0) -> Dict:
    style = render_style(sample_id, seed)
    style["legend_position"] = legend_pos
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), tuple(style["background"]))
    draw = ImageDraw.Draw(image)
    draw_background(draw, style)
    for edge in edges:
        draw_edge(draw, edge, styles)
    font = RENDERER.load_font(13)
    for node in nodes:
        draw_node(image, draw, node, font)
    for edge in edges:
        draw_arrow(draw, edge, styles)
    draw_legend(draw, RENDERER.load_font(16), styles, legend_pos)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = output_canvas(scale)
    if scale != 1.0:
        image = image.resize((canvas["width"], canvas["height"]), Image.Resampling.LANCZOS)
    image.save(image_path)
    style["image_scale"] = scale
    style["canonical_canvas"] = {"width": CANVAS_W, "height": CANVAS_H}
    style["output_canvas"] = canvas
    return style


def annotation_for(sample_id: str, split: str, style_mode: str, index: int, seed: int, image_path: Path, out_dir: Path, difficulty: str, relation_profile: str) -> Dict:
    nodes, edges, orientation = build_case(style_mode, index, seed, difficulty, relation_profile)
    styles = relation_styles(edges, sample_id, seed, difficulty, relation_profile)
    legend_pos = legend_position(sample_id, seed)
    if legend_pos == "top_left":
        translate_case(nodes, edges, 0, 32)
    else:
        translate_case(nodes, edges, 0, -28)
    image_scale = choose_image_scale(difficulty, nodes, edges, seed, sample_id)
    canvas = output_canvas(image_scale)
    style = render_image(sample_id, nodes, edges, image_path, seed, styles, legend_pos, image_scale)
    names = node_names(nodes)
    return {
        "sample_id": sample_id,
        "image": str(image_path.relative_to(out_dir)),
        "split": split,
        "difficulty": difficulty,
        "orientation": orientation,
        "canvas": canvas,
        "nodes": [node_record(node, image_scale) for node in reading_order(nodes)],
        "edges": [edge_record(edge, names, image_scale) for edge in edges],
        "negative_edges": [],
        "final_graph": final_graph(nodes, edges, image_scale),
        "think_graph": think_graph(nodes, edges, [], image_scale),
        "training_text": training_text(nodes, edges, [], image_scale),
        "render_style": style,
        "generation": {"seed": seed, "global_index": index, "attempt": 0, "layout": "foodweb_code", "relation_profile": relation_profile, "image_scale": image_scale},
    }


def make_contact_sheet(image_paths: List[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 252))
        thumbs.append((path.stem, image.copy()))
    cols = 2
    cell_w = 500
    cell_h = 300
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = RENDERER.load_font(16)
    for idx, (title, image) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 18
        y = (idx // cols) * cell_h + 38
        draw.text((x, y - 24), title, fill=(28, 35, 45), font=font)
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + image.width, y + image.height], outline=(210, 215, 222), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_natural_process_v1/outputs/preview_samples_v1")
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--per-difficulty", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out)
    image_paths: List[Path] = []
    summary = []
    for difficulty in ["simple", "medium", "difficult"]:
        for idx in range(1, args.per_difficulty + 1):
            style_mode = STYLE_MODES[(idx - 1) % len(STYLE_MODES)]
            profile_schedule = RELATION_PROFILE_SCHEDULES[difficulty]
            relation_profile = profile_schedule[(idx - 1) % len(profile_schedule)]
            sample_id = f"natural_{difficulty}_{idx:03d}"
            image_path = out / "images" / difficulty / f"{sample_id}.png"
            ann_path = out / "annotations" / difficulty / f"{sample_id}.json"
            ann = annotation_for(sample_id, "preview", style_mode, idx, args.seed, image_path, out, difficulty, relation_profile)
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")
            image_paths.append(image_path)
            summary.append({"sample_id": sample_id, "difficulty": difficulty, "nodes": len(ann["nodes"]), "edges": len(ann["edges"]), "image": ann["image"], "annotation": str(ann_path.relative_to(out)), "edge_types": sorted({e["type"] for e in ann["edges"]}), "relation_profile": relation_profile})
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(image_paths, out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
