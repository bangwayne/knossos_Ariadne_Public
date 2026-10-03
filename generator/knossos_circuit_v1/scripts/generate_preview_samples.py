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
LABELS_PATH = ROOT / "config" / "circuit_labels.json"
ASSET_META_PATH = ROOT / "assets" / "renderer_asset_pool" / "metadata.json"
ASSET_POOL = ROOT / "assets" / "renderer_asset_pool"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDERER = load_module("knossos_circuit_renderer_base", RENDERER_PATH)

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

BASE_CANVAS_W = RENDERER.CANVAS_W
BASE_CANVAS_H = RENDERER.CANVAS_H
CANVAS_W = BASE_CANVAS_W
CANVAS_H = BASE_CANVAS_H
EDGE_CLEARANCE = -3

STYLE_MODES = ["textbook_symbol", "realistic_component", "workbench_scene", "mixed"]
STANDARD_TOPOLOGIES = ["small_parallel_control", "foodweb_tiered_circuit", "controller_io", "power_distribution", "multi_link_pair"]
STANDARD_COMPLEX_TOPOLOGIES = ["complex_controller_io", "complex_power_tree", "complex_bridge_feedback", "complex_mixed_bus", "multi_link_bus"]
FOODWEB_CODE_TOPOLOGY = "foodweb_code_circuit"
LAYERED_CIRCUIT_TOPOLOGY = "layered_circuit"
NETWORK_CODE_TOPOLOGIES = ["network_star_circuit", "network_layered_circuit", "network_cluster_bridge_circuit"]
TOPOLOGIES = [LAYERED_CIRCUIT_TOPOLOGY, FOODWEB_CODE_TOPOLOGY] + NETWORK_CODE_TOPOLOGIES + STANDARD_TOPOLOGIES
COMPLEX_TOPOLOGIES = [
    LAYERED_CIRCUIT_TOPOLOGY,
    FOODWEB_CODE_TOPOLOGY,
] + NETWORK_CODE_TOPOLOGIES + STANDARD_COMPLEX_TOPOLOGIES
DIFFICULTY_TOPOLOGIES = {
    "simple": [LAYERED_CIRCUIT_TOPOLOGY],
    "medium": [LAYERED_CIRCUIT_TOPOLOGY],
    "difficult": [LAYERED_CIRCUIT_TOPOLOGY],
}

EDGE_STYLES = {
    "wire": {"color": (70, 74, 82), "width": 3, "arrow": False, "label": "Wire"},
    "power": {"color": (205, 60, 42), "width": 3, "arrow": True, "label": "Power"},
    "ground": {"color": (45, 48, 54), "width": 3, "arrow": True, "label": "Ground"},
    "signal": {"color": (35, 105, 185), "width": 3, "arrow": True, "label": "Signal"},
}

SEMANTIC_EDGE_COLORS = [
    (70, 74, 82),
    (205, 60, 42),
    (45, 48, 54),
    (35, 105, 185),
    (36, 135, 94),
    (150, 86, 170),
    (176, 115, 38),
]

LINK_TYPE_STEMS = [
    "Link",
    "Route",
    "Trace",
    "Bus",
    "Channel",
    "Path",
    "Net",
    "Flow",
    "Bridge",
    "Track",
    "Line",
    "Port",
    "Lane",
    "Arc",
    "Loop",
    "Pair",
    "Relay",
    "Hop",
]
LINK_TYPE_SUFFIXES = ["A", "B", "C", "D", "1", "2", "3", "X", "Y", "P", "Q", "I", "II"]


@dataclass
class CircuitNode:
    id: str
    label: str
    box: BBox
    role: str
    category: str
    asset_path: str | None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def icon_box(self) -> BBox | None:
        if self.asset_path is None:
            return None
        x1, y1, x2, y2 = self.box
        return (x1 + 6, y1 + 4, x2 - 6, y2 - 30)

    @property
    def text_box(self) -> BBox:
        x1, y1, x2, y2 = self.box
        return (x1, y2 - 26, x2, y2)


@dataclass
class CircuitEdge:
    id: str
    source: str
    target: str
    path: List[Point]
    edge_type: str
    directed: bool = True


def translate_box(box: BBox, dx: int, dy: int) -> BBox:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def translate_path(path: Sequence[Point], dx: int, dy: int) -> List[Point]:
    return [(x + dx, y + dy) for x, y in path]


def translate_case(nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge], dx: int, dy: int) -> None:
    if nodes:
        min_y = min(node.box[1] for node in nodes)
        max_y = max(node.box[3] for node in nodes)
        if dy < 0:
            dy = max(dy, 8 - min_y)
        elif dy > 0:
            dy = min(dy, CANVAS_H - 8 - max_y)
    for node in nodes:
        node.box = translate_box(node.box, dx, dy)
    for edge in edges:
        edge.path = translate_path(edge.path, dx, dy)


def set_canvas(width: int, height: int) -> None:
    global CANVAS_W, CANVAS_H
    CANVAS_W = width
    CANVAS_H = height


def target_canvas_for(difficulty: str, topology: str, orientation: str) -> Tuple[int, int]:
    horizontal, vertical = (900, 630), (630, 900)
    if orientation in {"top_to_bottom", "bottom_to_top"} and topology in {LAYERED_CIRCUIT_TOPOLOGY, FOODWEB_CODE_TOPOLOGY, "network_layered_circuit"}:
        return vertical
    return horizontal


def scale_point(point: Point, sx: float, sy: float) -> Point:
    return (int(round(point[0] * sx)), int(round(point[1] * sy)))


def scale_box(box: BBox, sx: float, sy: float) -> BBox:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def scale_case(nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge], width: int, height: int) -> None:
    sx = width / BASE_CANVAS_W
    sy = height / BASE_CANVAS_H
    for node in nodes:
        node.box = scale_box(node.box, sx, sy)
    for edge in edges:
        edge.path = [scale_point(point, sx, sy) for point in edge.path]


def legend_position(sample_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:{sample_id}:legend")
    return rng.choice(["top_left", "bottom_right"])


def load_labels() -> Tuple[List[Dict], List[Dict]]:
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return data["circuit"], data["edge_legend"]


def labels_by_role() -> Dict[str, List[Dict]]:
    labels, _ = load_labels()
    groups: Dict[str, List[Dict]] = {}
    for item in labels:
        groups.setdefault(item["role"], []).append(item)
    return groups


def asset_lookup() -> Dict[Tuple[str, str], List[Path]]:
    meta = json.loads(ASSET_META_PATH.read_text(encoding="utf-8"))
    lookup: Dict[Tuple[str, str], List[Path]] = {}
    for filename, item in meta.items():
        lookup.setdefault((item["label"], item["style"]), []).append(ASSET_POOL / filename)
    return lookup


def choose_label(role: str, rng: random.Random, used: set[str]) -> Dict:
    groups = labels_by_role()
    options = [item for item in groups[role] if item["label"] not in used] or groups[role]
    item = rng.choice(options)
    used.add(item["label"])
    return item


def display_name(item: Dict, rng: random.Random) -> str:
    name = item["display"]
    if rng.random() < 0.28 and item["role"] not in {"ground"}:
        return f"{name} {rng.choice(['A', 'B', 'C', '1', '2', 'X', 'Y'])}"
    return name


def foodweb_display_name(item: Dict, rng: random.Random) -> str:
    name = display_name(item, rng)
    mode = rng.choice(["upper", "lower", "title", "source"])
    if mode == "upper":
        return name.upper()
    if mode == "lower":
        return name.lower()
    if mode == "title":
        return name.title()
    return name


def asset_for(label: str, style_mode: str, rng: random.Random, lookup: Dict[Tuple[str, str], List[Path]]) -> Path | None:
    if style_mode == "block_label":
        return None
    style = rng.choice(["textbook_symbol", "realistic_component", "workbench_scene"]) if style_mode == "mixed" else style_mode
    options = lookup.get((label, style), [])
    return rng.choice(options) if options else None


def make_node(node_id: str, item: Dict, center: Point, style_mode: str, rng: random.Random, lookup, size: Tuple[int, int] | None = None) -> CircuitNode:
    w, h = size or (104, 112)
    x, y = center
    box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    asset = asset_for(item["label"], style_mode, rng, lookup)
    return CircuitNode(node_id, display_name(item, rng), box, item["role"], item["category"], str(asset.relative_to(ROOT)) if asset else None)


def make_foodweb_node(node_id: str, item: Dict, box: BBox, style_mode: str, rng: random.Random, lookup) -> CircuitNode:
    asset = asset_for(item["label"], style_mode, rng, lookup)
    return CircuitNode(node_id, foodweb_display_name(item, rng), box, item["role"], item["category"], str(asset.relative_to(ROOT)) if asset else None)


def add_duplicate_suffixes(nodes: Sequence[CircuitNode], rng: random.Random) -> None:
    groups: Dict[str, List[CircuitNode]] = {}
    for node in nodes:
        key = node.label.strip().lower()
        groups.setdefault(key, []).append(node)

    suffix_pool = ["A", "B", "C", "D", "1", "2", "3", "X", "Y"]
    for group in groups.values():
        if len(group) <= 1:
            continue
        suffixes = suffix_pool[:]
        rng.shuffle(suffixes)
        for idx, node in enumerate(group):
            suffix = suffixes[idx % len(suffixes)]
            node.label = f"{node.label} {suffix}"


def side_center(box: BBox, side: str) -> Point:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return {"left": (x1, cy), "right": (x2, cy), "top": (cx, y1), "bottom": (cx, y2)}[side]


def anchor_pair(source: CircuitNode, target: CircuitNode) -> Tuple[Point, Point]:
    sx, sy = source.center
    tx, ty = target.center
    if abs(tx - sx) >= abs(ty - sy):
        return (side_center(source.box, "right"), side_center(target.box, "left")) if tx >= sx else (side_center(source.box, "left"), side_center(target.box, "right"))
    return (side_center(source.box, "bottom"), side_center(target.box, "top")) if ty >= sy else (side_center(source.box, "top"), side_center(target.box, "bottom"))


def edge_path(source: CircuitNode, target: CircuitNode, bends: bool = False) -> List[Point]:
    start, end = anchor_pair(source, target)
    if not bends:
        return [start, end]
    mid_x = (start[0] + end[0]) // 2
    return [start, (mid_x, start[1]), (mid_x, end[1]), end]


def visible_edge_path(edge: CircuitEdge, clearance: int = EDGE_CLEARANCE) -> List[Point]:
    path = list(edge.path)
    path[0] = RENDERER.offset_point(path[0], path[1], clearance)
    path[-1] = RENDERER.offset_point(path[-1], path[-2], clearance)
    return path


def side_from_point(box: BBox, point: Point) -> str:
    x1, y1, x2, y2 = box
    x, y = point
    distances = {"left": abs(x - x1), "right": abs(x - x2), "top": abs(y - y1), "bottom": abs(y - y2)}
    return min(distances, key=distances.get)


def stagger_edge_endpoints(edges: List[CircuitEdge], nodes: Sequence[CircuitNode]) -> None:
    nodes_by_id = {node.id: node for node in nodes}
    for endpoint, node_attr, point_index, sort_index in [
        ("target", "target", -1, 0),
        ("source", "source", 0, -1),
    ]:
        grouped: Dict[Tuple[str, str], List[CircuitEdge]] = {}
        for edge in edges:
            node = nodes_by_id[getattr(edge, node_attr)]
            side = side_from_point(node.box, edge.path[point_index])
            grouped.setdefault((node.id, side), []).append(edge)
        for (node_id, side), side_edges in grouped.items():
            if len(side_edges) <= 1:
                continue
            node = nodes_by_id[node_id]
            x1, y1, x2, y2 = node.box
            ordered = sorted(side_edges, key=lambda item: (item.path[sort_index][1], item.path[sort_index][0], item.id))
            step = 11
            center = (len(ordered) - 1) / 2
            for idx, edge in enumerate(ordered):
                offset = int(round((idx - center) * step))
                x, y = edge.path[point_index]
                if side in {"left", "right"}:
                    edge.path[point_index] = (x, max(y1 + 14, min(y2 - 14, y + offset)))
                else:
                    edge.path[point_index] = (max(x1 + 14, min(x2 - 14, x + offset)), y)


def offset_parallel_multilinks(edges: List[CircuitEdge]) -> None:
    grouped: Dict[Tuple[str, str], List[CircuitEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)

    for pair_edges in grouped.values():
        if len(pair_edges) <= 1:
            continue
        ordered = sorted(pair_edges, key=lambda item: (item.edge_type, item.id))
        center = (len(ordered) - 1) / 2
        for idx, edge in enumerate(ordered):
            start = edge.path[0]
            end = edge.path[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy) or 1.0
            nx = -dy / length
            ny = dx / length
            offset = (idx - center) * 10
            edge.path = [(int(round(x + nx * offset)), int(round(y + ny * offset))) for x, y in edge.path]


def random_color(rng: random.Random) -> Tuple[int, int, int]:
    hue = rng.random()
    sat = rng.uniform(0.55, 0.88)
    val = rng.uniform(0.40, 0.82)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def distinct_colors(count: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    start = rng.random()
    hues = [((start + idx / max(1, count)) % 1.0) for idx in range(count)]
    rng.shuffle(hues)
    colors = []
    for idx, hue in enumerate(hues):
        sat = 0.68 + 0.18 * ((idx % 3) / 2)
        val = 0.48 + 0.24 * (((idx + 1) % 3) / 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def random_link_label(rng: random.Random, used: set[str]) -> str:
    for _ in range(100):
        stem = rng.choice(LINK_TYPE_STEMS)
        suffix = rng.choice(LINK_TYPE_SUFFIXES + [str(rng.randint(4, 18))])
        sep = rng.choice([" ", "-", " "])
        label = f"{stem}{sep}{suffix}"
        key = label.lower().replace("-", " ")
        if key not in used:
            used.add(key)
            return label
    label = f"Link {len(used) + 1}"
    used.add(label.lower())
    return label


def randomize_link_types(edges: List[CircuitEdge], sample_id: str, seed: int) -> Dict[str, Dict]:
    rng = random.Random(f"{seed}:{sample_id}:link-types")
    for edge in edges:
        edge.directed = True

    used: set[str] = set()
    edge_count = max(1, len(edges))
    grouped_by_pair_and_direction: Dict[Tuple[str, str, bool], List[CircuitEdge]] = {}
    for edge in edges:
        grouped_by_pair_and_direction.setdefault((edge.source, edge.target, edge.directed), []).append(edge)
    max_directed_parallel = max((len(group) for key, group in grouped_by_pair_and_direction.items() if key[2]), default=1)

    max_types = min(5, edge_count)
    min_directed_types = min(max_types, max(1, max_directed_parallel))
    min_total_types = max(1, min(max_types, min_directed_types))
    if max_types <= 2:
        total_types = max_types
    else:
        total_types = rng.randint(max(3, min_total_types), max_types)
    directed_count = total_types

    directed_labels = [random_link_label(rng, used) for _ in range(directed_count)]
    colors = distinct_colors(len(directed_labels), rng)
    edge_styles: Dict[str, Dict] = {}
    for label in directed_labels:
        edge_styles[label] = {"color": colors.pop(), "width": rng.choice([2, 2, 3, 3, 4]), "arrow": True, "label": label}

    assigned: set[str] = set()
    for group in grouped_by_pair_and_direction.values():
        if len(group) <= 1:
            continue
        available = list(directed_labels)
        rng.shuffle(available)
        for edge, label in zip(sorted(group, key=lambda item: item.id), available):
            edge.edge_type = label
            assigned.add(edge.id)

    for idx, edge in enumerate(edges):
        if edge.id in assigned:
            continue
        edge.edge_type = rng.choice(directed_labels)
    return edge_styles


def semantic_circuit_edge_styles(edges: List[CircuitEdge], sample_id: str, seed: int) -> Dict[str, Dict]:
    ordered_types = [etype for etype in ["wire", "signal", "power", "ground"] if any(edge.edge_type == etype for edge in edges)]
    counts = {etype: sum(1 for edge in edges if edge.edge_type == etype) for etype in ordered_types}
    rng = random.Random(f"{sample_id}:{seed}:semantic-directions")

    if not ordered_types:
        directed_types: set[str] = set()
    elif len(ordered_types) == 1:
        directed_types = set() if rng.random() < 0.75 else {ordered_types[0]}
    else:
        max_directed_edges = max(1, int(len(edges) * 0.45))
        shuffled = list(ordered_types)
        rng.shuffle(shuffled)
        candidates = [etype for etype in shuffled if counts[etype] <= max_directed_edges]
        if not candidates:
            candidates = [min(ordered_types, key=lambda etype: counts[etype])]

        directed_types = {candidates[0]}
        if len(candidates) > 1 and len(edges) >= 12 and rng.random() < 0.38:
            extra_options = [etype for etype in candidates[1:] if counts[candidates[0]] + counts[etype] <= max_directed_edges]
            if extra_options:
                directed_types.add(rng.choice(extra_options))

    for edge in edges:
        edge.directed = edge.edge_type in directed_types

    colors = list(SEMANTIC_EDGE_COLORS)
    rng.shuffle(colors)
    edge_styles: Dict[str, Dict] = {}
    for etype in ordered_types:
        edge_styles[etype] = dict(EDGE_STYLES[etype])
        edge_styles[etype]["color"] = colors.pop()
        edge_styles[etype]["width"] = rng.choice([2, 3, 3, 4])
        edge_styles[etype]["arrow"] = etype in directed_types
    return edge_styles


def choose_topology(topology_pool: Sequence[str], index: int, rng: random.Random) -> str:
    if len(topology_pool) == 1:
        return topology_pool[0]
    standard_pool = [item for item in topology_pool if item != FOODWEB_CODE_TOPOLOGY]
    network_pool = [item for item in topology_pool if item in NETWORK_CODE_TOPOLOGIES]
    other_pool = [item for item in standard_pool if item not in NETWORK_CODE_TOPOLOGIES]
    if not standard_pool:
        return FOODWEB_CODE_TOPOLOGY
    slot = (index - 1) % 10
    if slot < 4:
        return FOODWEB_CODE_TOPOLOGY
    if slot < 7 and network_pool:
        return network_pool[(index - 5) % len(network_pool)]
    if other_pool:
        return other_pool[(index - 8) % len(other_pool)]
    return standard_pool[(index - 1) % len(standard_pool)]


def build_foodweb_code_case(style_mode: str, index: int, seed: int, difficulty: str, rng: random.Random, lookup):
    role_groups = labels_by_role()
    tier_role_pools = [
        ["power", "sensor", "passive"],
        ["control", "active", "passive", "sensor"],
        ["load", "active", "ground"],
    ]
    orientation = rng.choice(["bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"])
    if difficulty == "simple":
        tier_counts = [rng.randint(2, 3), rng.randint(3, 4), 2]
        extra_edges = rng.randint(3, 4)
        bends = rng.random() < 0.35
    elif difficulty == "medium":
        tier_counts = [rng.randint(2, 3), rng.randint(4, 5), rng.randint(3, 4)]
        extra_edges = rng.randint(1, 3)
        bends = True
    else:
        tier_counts = [rng.randint(4, 5), rng.randint(7, 9), rng.randint(5, 6)]
        extra_edges = rng.randint(4, 6)
        bends = True

    used: set[str] = set()

    def pick_role(tier_idx: int) -> str:
        options = [role for role in tier_role_pools[tier_idx] if role_groups.get(role)]
        if tier_idx == 2 and rng.random() < 0.22:
            return "ground"
        return rng.choice(options)

    def node_size() -> Tuple[int, int]:
        if difficulty == "difficult":
            return rng.randint(56, 68), rng.randint(64, 76)
        if difficulty == "medium":
            return rng.randint(90, 104), rng.randint(102, 114)
        return rng.randint(96, 112), rng.randint(108, 120)

    def box_for(tier_idx: int, local_idx: int, count: int, width: int, height: int) -> BBox:
        if orientation in {"bottom_to_top", "top_to_bottom"}:
            y_bases = [520, 330, 110] if orientation == "bottom_to_top" else [80, 295, 510]
            step = 850 // count
            x_jitter = rng.randint(-6, 6) if difficulty == "difficult" else rng.randint(-16, 16)
            x_center = 70 + step // 2 + local_idx * step + x_jitter
            x = min(900 - width, max(45, x_center - width // 2))
            y = y_bases[tier_idx] + rng.randint(-6, 6)
            return (x, y, x + width, y + height)

        x_bases = [65, 420, 760] if orientation == "left_to_right" else [820, 470, 115]
        usable_h = 600
        step = usable_h // count
        y_jitter = 0 if difficulty == "difficult" else rng.randint(-14, 14)
        y_center = 50 + step // 2 + local_idx * step + y_jitter
        y = min(620 - height, max(35, y_center - height // 2))
        x = x_bases[tier_idx] + rng.randint(-6, 6)
        return (x, y, x + width, y + height)

    nodes: List[CircuitNode] = []
    tier_nodes: List[List[CircuitNode]] = []
    idx = 1
    for tier_idx, count in enumerate(tier_counts):
        current = []
        for local_idx in range(count):
            width, height = node_size()
            if orientation in {"left_to_right", "right_to_left"}:
                max_height = max(46, (600 // count) - 30)
                height = min(height, max_height)
            role = pick_role(tier_idx)
            item = choose_label(role, rng, used)
            node = make_foodweb_node(f"tmp{idx}", item, box_for(tier_idx, local_idx, count, width, height), style_mode, rng, lookup)
            nodes.append(node)
            current.append(node)
            idx += 1
        tier_nodes.append(current)

    edges: List[CircuitEdge] = []
    seen: set[Tuple[str, str, str]] = set()

    def type_for(src: CircuitNode, dst: CircuitNode, primary: bool = False) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power" or primary:
            return rng.choice(["power", "signal"])
        if src.role in {"sensor", "control", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    def add(src: CircuitNode, dst: CircuitNode, edge_type: str | None = None, allow_parallel: bool = False) -> None:
        if src.id == dst.id:
            return
        etype = edge_type or type_for(src, dst)
        key = (src.id, dst.id, etype)
        if key in seen or (not allow_parallel and any(edge.source == src.id and edge.target == dst.id for edge in edges)):
            return
        seen.add(key)
        start = RENDERER.boundary_point_towards(src.box, dst.center)
        end = RENDERER.boundary_point_towards(dst.box, src.center)
        if bends:
            mid = (
                (start[0] + end[0]) // 2 + rng.randint(-32, 32),
                (start[1] + end[1]) // 2 + rng.randint(-16, 16),
            )
            path = [start, mid, end]
        else:
            path = [start, end]
        edges.append(CircuitEdge(f"e{len(edges) + 1}", src.id, dst.id, path, etype))

    for src in tier_nodes[0]:
        dst = rng.choice(tier_nodes[1])
        add(src, dst, type_for(src, dst, primary=True))
    for src in tier_nodes[1]:
        add(src, rng.choice(tier_nodes[2]))
    for _ in range(extra_edges):
        if rng.random() < 0.45:
            add(rng.choice(tier_nodes[0]), rng.choice(tier_nodes[1]))
        else:
            add(rng.choice(tier_nodes[1]), rng.choice(tier_nodes[2]))

    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    for tier_idx, tier in enumerate(tier_nodes):
        for lonely in tier:
            if lonely.id in linked:
                continue
            if tier_idx == 0:
                add(lonely, rng.choice(tier_nodes[1]))
            elif tier_idx == 1:
                add(rng.choice(tier_nodes[0]), lonely)
            else:
                add(rng.choice(tier_nodes[1]), lonely)
            linked = {edge.source for edge in edges} | {edge.target for edge in edges}

    return nodes, edges, orientation


def transform_center(center: Point, orientation: str) -> Point:
    x, y = center
    if orientation == "top_to_bottom":
        return (x, y)
    if orientation == "bottom_to_top":
        return (x, CANVAS_H - y)
    if orientation == "left_to_right":
        return (int(y / CANVAS_H * CANVAS_W), int(x / CANVAS_W * CANVAS_H))
    if orientation == "right_to_left":
        return (int((1 - y / CANVAS_H) * CANVAS_W), int(x / CANVAS_W * CANVAS_H))
    return (x, y)


def radial_points(count: int, cx: int, cy: int, rx: int, ry: int, start: float = -math.pi / 2) -> List[Point]:
    return [
        (int(cx + rx * math.cos(start + 2 * math.pi * idx / count)), int(cy + ry * math.sin(start + 2 * math.pi * idx / count)))
        for idx in range(count)
    ]


def center_distance(a: CircuitNode, b: CircuitNode) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def network_node_count(difficulty: str, rng: random.Random) -> int:
    if difficulty == "simple":
        return rng.randint(7, 9)
    if difficulty == "medium":
        return rng.randint(8, 11)
    return rng.randint(15, 18)


def network_node_size(difficulty: str) -> Tuple[int, int]:
    if difficulty == "simple":
        return (86, 96)
    if difficulty == "medium":
        return (82, 94)
    return (68, 78)


def build_network_code_case(topology: str, style_mode: str, index: int, seed: int, difficulty: str, rng: random.Random, lookup):
    role_pools = {
        "source": ["power", "sensor", "passive"],
        "hub": ["control", "active", "passive"],
        "leaf": ["sensor", "passive", "active", "load", "ground"],
    }
    orientation = rng.choice(["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"])
    count = network_node_count(difficulty, rng)
    width, height = network_node_size(difficulty)

    if topology == "network_star_circuit":
        centers = [(500, 350)] + radial_points(count - 1, 500, 350, 355, 235)
        roles = ["hub"] + ["leaf"] * (count - 1)
    elif topology == "network_layered_circuit":
        top_count = rng.randint(3, 4) if difficulty == "difficult" else 2
        mid_count = rng.randint(6, 7) if difficulty == "difficult" else max(3, count // 3)
        bottom_count = count - top_count - mid_count
        row_counts = [top_count, mid_count, bottom_count]
        y_values = [115, 350, 585]
        centers = []
        roles = []
        for row_idx, (row_count, y) in enumerate(zip(row_counts, y_values)):
            if row_count <= 0:
                continue
            step = 780 // max(1, row_count - 1) if row_count > 1 else 0
            start_x = 110 if row_count > 1 else 500
            for local_idx in range(row_count):
                centers.append((start_x + local_idx * step, y))
                roles.append(["source", "hub", "leaf"][row_idx])
    else:
        cluster_centers = [(260, 350), (740, 350)]
        centers = []
        roles = []
        per_cluster = math.ceil(count / len(cluster_centers))
        for cluster_idx, center in enumerate(cluster_centers):
            remaining = count - len(centers)
            cluster_count = min(per_cluster, remaining)
            if cluster_count <= 0:
                continue
            centers.append(center)
            roles.append("hub")
            if cluster_count > 1:
                centers.extend(radial_points(cluster_count - 1, center[0], center[1], 225, 185, start=-math.pi / 2 + cluster_idx * 0.2))
                roles.extend(["leaf"] * (cluster_count - 1))

    centers = [transform_center(center, orientation) for center in centers]
    used: set[str] = set()
    nodes: List[CircuitNode] = []
    for idx, (center, role_key) in enumerate(zip(centers, roles), 1):
        role = rng.choice(role_pools[role_key])
        item = choose_label(role, rng, used)
        w, h = width, height
        x = max(35, min(890, center[0] - w // 2))
        y = max(30, min(585, center[1] - h // 2))
        nodes.append(make_foodweb_node(f"tmp{idx}", item, (x, y, x + w, y + h), style_mode, rng, lookup))

    edges: List[CircuitEdge] = []
    seen: set[Tuple[str, str]] = set()
    min_dist = {"simple": 190, "medium": 205, "difficult": 170}[difficulty]

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    def add(src: CircuitNode, dst: CircuitNode, bends: bool = False, allow_short: bool = False) -> bool:
        key = tuple(sorted([src.id, dst.id]))
        if src.id == dst.id or key in seen:
            return False
        if not allow_short and center_distance(src, dst) < min_dist:
            return False
        seen.add(key)
        start = RENDERER.boundary_point_towards(src.box, dst.center)
        end = RENDERER.boundary_point_towards(dst.box, src.center)
        if bends:
            mid = ((start[0] + end[0]) // 2, start[1]) if rng.random() < 0.5 else (start[0], (start[1] + end[1]) // 2)
            path = [start, mid, end]
        else:
            path = [start, end]
        edges.append(CircuitEdge(f"e{len(edges) + 1}", src.id, dst.id, path, type_for(src, dst)))
        return True

    if topology == "network_star_circuit":
        hub = min(nodes, key=lambda node: abs(node.center[0] - 500) + abs(node.center[1] - 350))
        for node in nodes:
            if node is not hub:
                add(hub, node, allow_short=True)
    elif topology == "network_layered_circuit":
        horizontal_layers = orientation in {"top_to_bottom", "bottom_to_top"}
        rows: Dict[int, List[CircuitNode]] = {}
        for node in nodes:
            key_value = node.center[1] if horizontal_layers else node.center[0]
            rows.setdefault(round(key_value / 100), []).append(node)
        ordered_rows = [sorted(rows[key], key=lambda node: node.center[0] if horizontal_layers else node.center[1]) for key in sorted(rows)]
        if len(ordered_rows) >= 3:
            top, middle, bottom = ordered_rows[0], ordered_rows[1], ordered_rows[2]
        else:
            top, middle, bottom = ordered_rows[0], ordered_rows[-1], []
        for item in middle:
            parent = min(top, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (item.center[0] if horizontal_layers else item.center[1])))
            add(parent, item, allow_short=True)
        for item in bottom:
            parent = min(middle, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (item.center[0] if horizontal_layers else item.center[1])))
            add(parent, item, allow_short=True)
        if difficulty in {"medium", "difficult"} and len(middle) >= 2 and len(bottom) >= 3:
            candidates = []
            for mid in middle:
                nearest = min(bottom, key=lambda candidate: abs((candidate.center[0] if horizontal_layers else candidate.center[1]) - (mid.center[0] if horizontal_layers else mid.center[1])))
                far_options = [node for node in bottom if node is not nearest and center_distance(mid, node) >= min_dist]
                if far_options:
                    candidates.append((mid, max(far_options, key=lambda node: center_distance(mid, node))))
            rng.shuffle(candidates)
            for src, dst in candidates[: 1 if difficulty == "medium" else 3]:
                add(src, dst)
    else:
        split_axis = 0 if orientation in {"top_to_bottom", "bottom_to_top"} else 1
        groups = sorted(nodes, key=lambda node: node.center[split_axis])
        split_a = len(groups) // 2
        clusters = [groups[:split_a], groups[split_a:]]
        hubs = []
        for cluster in clusters:
            center_x = sum(node.center[0] for node in cluster) / len(cluster)
            center_y = sum(node.center[1] for node in cluster) / len(cluster)
            hub = min(cluster, key=lambda node: abs(node.center[0] - center_x) + abs(node.center[1] - center_y))
            hubs.append(hub)
            for node in cluster:
                if node is not hub:
                    add(hub, node, allow_short=True)
        for left_hub, right_hub in zip(hubs, hubs[1:]):
            add(left_hub, right_hub)

    lo, hi = {"simple": (6, 10), "medium": (8, 12), "difficult": (16, 24)}[difficulty]
    attempts = 0
    while len(edges) < lo and attempts < 200 and topology != "network_star_circuit":
        src, dst = rng.sample(nodes, 2)
        add(src, dst)
        attempts += 1
    if len(edges) > hi:
        edges = edges[:hi]
    return nodes, edges, orientation


def layered_counts_for(difficulty: str, rng: random.Random) -> List[int]:
    if difficulty == "simple":
        return [2, rng.randint(2, 3), rng.randint(2, 3), rng.randint(1, 2)]
    if difficulty == "medium":
        return [rng.randint(2, 3), rng.randint(3, 4), rng.randint(3, 4), rng.randint(2, 3)]
    return [rng.randint(3, 4), rng.randint(3, 5), rng.randint(4, 5), rng.randint(3, 5), rng.randint(3, 4)]


def layered_positions(layer_count: int, orientation: str) -> List[int]:
    if orientation in {"top_to_bottom", "bottom_to_top"}:
        positions = [int(round(88 + idx * (CANVAS_H - 176) / max(1, layer_count - 1))) for idx in range(layer_count)]
        return list(reversed(positions)) if orientation == "bottom_to_top" else positions
    positions = [int(round(88 + idx * (CANVAS_W - 176) / max(1, layer_count - 1))) for idx in range(layer_count)]
    return list(reversed(positions)) if orientation == "right_to_left" else positions


def layered_center(layer_idx: int, local_idx: int, count: int, layer_count: int, orientation: str, difficulty: str, rng: random.Random) -> Point:
    bases = layered_positions(layer_count, orientation)
    jitter = 4 if difficulty == "difficult" else 10
    if orientation in {"top_to_bottom", "bottom_to_top"}:
        step = (CANVAS_W - 170) // max(1, count)
        x = 85 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
        return (x, bases[layer_idx] + rng.randint(-4, 4))
    step = (CANVAS_H - 150) // max(1, count)
    y = 75 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
    return (bases[layer_idx] + rng.randint(-4, 4), y)


def soft_layer_edge_path(source: CircuitNode, target: CircuitNode, rng: random.Random, difficulty: str) -> List[Point]:
    start = RENDERER.boundary_point_towards(source.box, target.center)
    end = RENDERER.boundary_point_towards(target.box, source.center)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy) or 1.0
    if dist < 150 and rng.random() < 0.35:
        return [start, end]
    nx, ny = -dy / dist, dx / dist
    bend = {"simple": 10, "medium": 16, "difficult": 20}.get(difficulty, 14)
    bend *= rng.choice([-1, 1])
    mid = (
        int(round((start[0] + end[0]) / 2 + nx * bend + rng.randint(-5, 5))),
        int(round((start[1] + end[1]) / 2 + ny * bend + rng.randint(-5, 5))),
    )
    return [start, mid, end]


def build_layered_circuit_case(style_mode: str, index: int, seed: int, difficulty: str, rng: random.Random, lookup):
    orientation = rng.choice(["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"])
    counts = layered_counts_for(difficulty, rng)
    layer_role_pools = [
        ["power", "sensor", "passive"],
        ["passive", "sensor", "control"],
        ["control", "active", "passive"],
        ["active", "load", "ground"],
        ["load", "ground", "active"],
    ]
    used: set[str] = set()
    nodes: List[CircuitNode] = []
    by_layer: List[List[CircuitNode]] = []
    idx = 1
    if difficulty == "simple":
        size = (86, 94)
    elif difficulty == "medium":
        size = (78, 88)
    else:
        size = (60, 70)

    for layer_idx, count in enumerate(counts):
        current = []
        for local_idx in range(count):
            role = rng.choice(layer_role_pools[layer_idx % len(layer_role_pools)])
            if layer_idx == len(counts) - 1 and rng.random() < 0.35:
                role = rng.choice(["load", "ground"])
            item = choose_label(role, rng, used)
            center = layered_center(layer_idx, local_idx, count, len(counts), orientation, difficulty, rng)
            x, y = center
            box = (x - size[0] // 2, y - size[1] // 2, x + size[0] // 2, y + size[1] // 2)
            node = make_foodweb_node(f"tmp{idx}", item, box, style_mode, rng, lookup)
            nodes.append(node)
            current.append(node)
            idx += 1
        by_layer.append(current)

    edges: List[CircuitEdge] = []
    seen: set[Tuple[str, str, str]] = set()

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    def add(src: CircuitNode, dst: CircuitNode, edge_type: str | None = None, allow_parallel: bool = False) -> None:
        if src.id == dst.id:
            return
        etype = edge_type or type_for(src, dst)
        if not allow_parallel and any(edge.source == src.id and edge.target == dst.id for edge in edges):
            return
        pair = tuple(sorted((src.id, dst.id)))
        key = (pair[0], pair[1], etype)
        if key in seen:
            return
        seen.add(key)
        edges.append(CircuitEdge(f"e{len(edges) + 1}", src.id, dst.id, soft_layer_edge_path(src, dst, rng, difficulty), etype))

    for layer_idx in range(len(by_layer) - 1):
        current = by_layer[layer_idx]
        next_layer = by_layer[layer_idx + 1]
        for src_idx, src in enumerate(current):
            mapped = round(src_idx * (len(next_layer) - 1) / max(1, len(current) - 1))
            dst_idx = max(0, min(len(next_layer) - 1, mapped + rng.choice([-1, 0, 0, 1])))
            add(src, next_layer[dst_idx])
            extra_prob = 0.25 if difficulty == "medium" else 0.30
            if difficulty != "simple" and len(next_layer) > 1 and rng.random() < extra_prob:
                adjacent_idx = max(0, min(len(next_layer) - 1, dst_idx + rng.choice([-1, 1])))
                if adjacent_idx != dst_idx:
                    add(src, next_layer[adjacent_idx])
        current_ids = {node.id for node in current}
        for dst_idx, dst in enumerate(next_layer):
            if not any(edge.target == dst.id and edge.source in current_ids for edge in edges):
                mapped = round(dst_idx * (len(current) - 1) / max(1, len(next_layer) - 1))
                add(current[max(0, min(len(current) - 1, mapped))], dst)

    extra_edges = {"simple": rng.randint(0, 1), "medium": rng.randint(2, 4), "difficult": rng.randint(4, 6)}[difficulty]
    for _ in range(extra_edges):
        if rng.random() < 0.72:
            layer_idx = rng.randrange(len(by_layer) - 1)
            src_layer, dst_layer = by_layer[layer_idx], by_layer[layer_idx + 1]
            src_idx = rng.randrange(len(src_layer))
            mapped = round(src_idx * (len(dst_layer) - 1) / max(1, len(src_layer) - 1))
            dst_idx = max(0, min(len(dst_layer) - 1, mapped + rng.choice([-1, 0, 1])))
            add(src_layer[src_idx], dst_layer[dst_idx])
        else:
            layer = rng.choice([items for items in by_layer if len(items) >= 2])
            idx = rng.randrange(len(layer) - 1)
            a, b = layer[idx], layer[idx + 1]
            if rng.random() < 0.5:
                a, b = b, a
            add(a, b)

    multi_roll = rng.random()
    if (difficulty == "medium" and multi_roll < 0.28) or (difficulty == "difficult" and multi_roll < 0.78):
        candidates = [edge for edge in edges if len(edge.path) >= 2]
        rng.shuffle(candidates)
        max_multi = 1 if difficulty == "medium" else 2
        for edge in candidates[:max_multi]:
            pair = {edge.source, edge.target}
            existing = {item.edge_type for item in edges if {item.source, item.target} == pair}
            choices = [etype for etype in ["power", "signal", "wire", "ground"] if etype not in existing]
            if choices:
                edges.append(CircuitEdge(f"e{len(edges) + 1}", edge.source, edge.target, list(edge.path), rng.choice(choices)))

    return nodes, edges, orientation


def apply_multilink_policy(edges: List[CircuitEdge], difficulty: str, rng: random.Random) -> None:
    if not edges:
        return
    roll = rng.random()
    if difficulty == "simple":
        mode = "light" if roll < 0.22 else "none"
    elif difficulty == "medium":
        mode = "heavy" if roll < 0.03 else ("light" if roll < 0.25 else "none")
    else:
        mode = "heavy" if roll < 0.18 else ("light" if roll < 0.55 else "none")
    if mode == "none":
        return
    candidates = [edge for edge in edges if len(edge.path) >= 2]
    rng.shuffle(candidates)
    selected = candidates[: 1 if mode == "light" else min(2, len(candidates))]
    for edge in selected:
        existing = {item.edge_type for item in edges if item.source == edge.source and item.target == edge.target}
        choices = [etype for etype in ["power", "signal", "wire"] if etype not in existing]
        rng.shuffle(choices)
        target_count = 2 if mode == "light" else 3
        while choices and sum(1 for item in edges if item.source == edge.source and item.target == edge.target) < target_count:
            edges.append(CircuitEdge(f"e{len(edges) + 1}", edge.source, edge.target, list(edge.path), choices.pop()))


def apply_standard_difficult_multilink_policy(edges: List[CircuitEdge], rng: random.Random) -> None:
    if not edges or rng.random() >= 0.22:
        return
    candidates = [edge for edge in edges if len(edge.path) >= 2]
    rng.shuffle(candidates)
    for edge in candidates:
        existing = {item.edge_type for item in edges if item.source == edge.source and item.target == edge.target}
        choices = [etype for etype in ["power", "signal", "wire"] if etype not in existing]
        if choices:
            edges.append(CircuitEdge(f"e{len(edges) + 1}", edge.source, edge.target, list(edge.path), rng.choice(choices)))
            return


def ensure_no_isolated_nodes(nodes: Sequence[CircuitNode], edges: List[CircuitEdge], rng: random.Random) -> None:
    linked = {edge.source for edge in edges} | {edge.target for edge in edges}
    isolated = [node for node in nodes if node.id not in linked]
    rng.shuffle(isolated)

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    for node in isolated:
        candidates = [other for other in nodes if other is not node]
        if not candidates:
            continue
        target = min(candidates, key=lambda other: center_distance(node, other))
        start = RENDERER.boundary_point_towards(node.box, target.center)
        end = RENDERER.boundary_point_towards(target.box, node.center)
        edges.append(CircuitEdge(f"e{len(edges) + 1}", node.id, target.id, [start, end], type_for(node, target)))
        linked.update([node.id, target.id])


def densify_edges_for_difficulty(nodes: Sequence[CircuitNode], edges: List[CircuitEdge], difficulty: str, rng: random.Random) -> None:
    if not nodes:
        return
    target_min = {"simple": 6, "medium": 8, "difficult": 15}.get(difficulty, 8)
    target_max = {"simple": 10, "medium": 14, "difficult": 21}.get(difficulty, 14)
    existing = {(edge.source, edge.target, edge.edge_type) for edge in edges}

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    attempts = 0
    while len(edges) < target_min and attempts < 600:
        attempts += 1
        src, dst = rng.sample(list(nodes), 2)
        if src.id == dst.id:
            continue
        min_center = {"simple": 150, "medium": 165, "difficult": 235}.get(difficulty, 165)
        if center_distance(src, dst) < min_center:
            continue
        same_pair = sum(1 for edge in edges if edge.source == src.id and edge.target == dst.id)
        reverse_pair = sum(1 for edge in edges if edge.source == dst.id and edge.target == src.id)
        if same_pair and difficulty != "difficult":
            continue
        if same_pair >= 2 or reverse_pair:
            continue
        edge_type = type_for(src, dst)
        key = (src.id, dst.id, edge_type)
        if key in existing:
            continue
        existing.add(key)
        bends = difficulty in {"medium", "difficult"} and rng.random() < (0.45 if difficulty == "difficult" else 0.25)
        edges.append(CircuitEdge(f"e{len(edges) + 1}", src.id, dst.id, edge_path(src, dst, bends=bends), edge_type))
    if len(edges) > target_max:
        del edges[target_max:]


def augment_standard_difficult_nodes(
    nodes: List[CircuitNode],
    edges: List[CircuitEdge],
    style_mode: str,
    rng: random.Random,
    lookup,
    used: set[str],
) -> None:
    target_nodes = rng.randint(14, 16)
    if len(nodes) >= target_nodes:
        return
    role_pool = ["sensor", "passive", "active", "load", "control"]
    candidate_centers = [
        (145, 105), (295, 115), (500, 105), (710, 115), (870, 145),
        (145, 595), (295, 580), (500, 595), (710, 580), (870, 555),
        (210, 245), (790, 250), (210, 455), (790, 460),
    ]
    rng.shuffle(candidate_centers)

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    while len(nodes) < target_nodes and candidate_centers:
        center = candidate_centers.pop()
        if any(math.hypot(center[0] - node.center[0], center[1] - node.center[1]) < 190 for node in nodes):
            continue
        role = rng.choice(role_pool)
        item = choose_label(role, rng, used)
        node = make_node(f"tmp_extra_{len(nodes) + 1}", item, center, style_mode, rng, lookup, size=(74, 84))
        nodes.append(node)
        candidates = [other for other in nodes if other is not node and other.role != "ground"]
        if not candidates:
            continue
        near = min(candidates, key=lambda other: center_distance(node, other))
        far_candidates = sorted(candidates, key=lambda other: center_distance(node, other), reverse=True)
        edges.append(CircuitEdge(f"e{len(edges) + 1}", near.id, node.id, edge_path(near, node, bends=rng.random() < 0.35), type_for(near, node)))
        if far_candidates and rng.random() < 0.35:
            other = far_candidates[0]
            edges.append(CircuitEdge(f"e{len(edges) + 1}", node.id, other.id, edge_path(node, other, bends=rng.random() < 0.45), type_for(node, other)))


def remove_too_short_edges(edges: List[CircuitEdge], min_length: float = 35.0) -> None:
    kept = []
    for edge in edges:
        path = visible_edge_path(edge)
        length = sum(math.hypot(path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1]) for idx in range(len(path) - 1))
        if length >= min_length:
            kept.append(edge)
    edges[:] = kept


def reroute_standard_difficult_edges(nodes: Sequence[CircuitNode], edges: List[CircuitEdge], difficulty: str) -> None:
    if difficulty != "difficult":
        return
    nodes_by_id = {node.id: node for node in nodes}
    grouped_target: Dict[str, int] = {}
    for idx, edge in enumerate(edges):
        src = nodes_by_id[edge.source]
        dst = nodes_by_id[edge.target]
        start, end = anchor_pair(src, dst)
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dx < 80 or dy < 80:
            edge.path = [start, end]
            continue

        
        
        lane_index = grouped_target.get(edge.target, 0)
        grouped_target[edge.target] = lane_index + 1
        lane_offset = (lane_index % 5 - 2) * 14
        if dx >= dy:
            mid_x = (start[0] + end[0]) // 2 + lane_offset
            edge.path = [start, (mid_x, start[1]), (mid_x, end[1]), end]
        else:
            mid_y = (start[1] + end[1]) // 2 + lane_offset
            edge.path = [start, (start[0], mid_y), (end[0], mid_y), end]


def densify_standard_edges_for_difficulty(nodes: Sequence[CircuitNode], edges: List[CircuitEdge], difficulty: str, rng: random.Random) -> None:
    if difficulty != "difficult" or not nodes:
        densify_edges_for_difficulty(nodes, edges, difficulty, rng)
        return
    target_min = 12
    target_max = 15
    existing_pairs = {(edge.source, edge.target) for edge in edges}
    existing_type_keys = {(edge.source, edge.target, edge.edge_type) for edge in edges}
    ordered = sorted(nodes, key=lambda node: node.center[0])

    def type_for(src: CircuitNode, dst: CircuitNode) -> str:
        if dst.role == "ground":
            return "ground"
        if src.role == "power":
            return rng.choice(["power", "signal"])
        if src.role in {"control", "sensor", "active"}:
            return rng.choice(["signal", "wire"])
        return rng.choice(["wire", "signal"])

    candidates: List[Tuple[float, CircuitNode, CircuitNode]] = []
    for src in ordered:
        for dst in ordered:
            if src.id == dst.id:
                continue
            dx = dst.center[0] - src.center[0]
            dy = abs(dst.center[1] - src.center[1])
            dist = center_distance(src, dst)
            if dx <= 80 or dist < 170 or dist > 470:
                continue
            if (src.id, dst.id) in existing_pairs or (dst.id, src.id) in existing_pairs:
                continue
            
            score = abs(dx - 250) + dy * 0.45 + rng.random() * 40
            candidates.append((score, src, dst))
    candidates.sort(key=lambda item: item[0])

    for _, src, dst in candidates:
        if len(edges) >= target_min:
            break
        edge_type = type_for(src, dst)
        key = (src.id, dst.id, edge_type)
        if key in existing_type_keys:
            continue
        existing_pairs.add((src.id, dst.id))
        existing_type_keys.add(key)
        edges.append(CircuitEdge(f"e{len(edges) + 1}", src.id, dst.id, edge_path(src, dst, bends=rng.random() < 0.30), edge_type))
    if len(edges) > target_max:
        removable = [idx for idx in range(len(edges) - 1, -1, -1) if len(visible_edge_path(edges[idx])) > 2]
        for idx in removable:
            if len(edges) <= target_max:
                break
            del edges[idx]


def build_case(style_mode: str, index: int, seed: int, difficulty: str = "preview", complexity: str = "normal"):
    rng = random.Random(f"{seed}:{style_mode}:{difficulty}:{index}")
    if difficulty in DIFFICULTY_TOPOLOGIES:
        topology_pool = DIFFICULTY_TOPOLOGIES[difficulty]
    else:
        topology_pool = COMPLEX_TOPOLOGIES if complexity == "complex" else TOPOLOGIES
    topology = choose_topology(topology_pool, index, rng)
    lookup = asset_lookup()
    used: set[str] = set()

    def node(role: str, center: Point, idx: int) -> CircuitNode:
        size = {"simple": (96, 104), "medium": (90, 98), "difficult": (72, 82)}.get(difficulty, (104, 112))
        return make_node(f"tmp{idx}", choose_label(role, rng, used), center, style_mode, rng, lookup, size=size)

    orientation = "left_to_right"
    if topology == LAYERED_CIRCUIT_TOPOLOGY:
        nodes, edges, orientation = build_layered_circuit_case(style_mode, index, seed, difficulty, rng, lookup)
        add_duplicate_suffixes(nodes, rng)
        offset_parallel_multilinks(edges)
        stagger_edge_endpoints(edges, nodes)
        offset_parallel_multilinks(edges)
        remove_too_short_edges(edges)
        reassign_ids(nodes, edges)
        return nodes, edges, topology, orientation
    if topology == FOODWEB_CODE_TOPOLOGY:
        nodes, edges, orientation = build_foodweb_code_case(style_mode, index, seed, difficulty, rng, lookup)
        add_duplicate_suffixes(nodes, rng)
        ensure_no_isolated_nodes(nodes, edges, rng)
        densify_edges_for_difficulty(nodes, edges, difficulty, rng)
        apply_multilink_policy(edges, difficulty, rng)
        offset_parallel_multilinks(edges)
        stagger_edge_endpoints(edges, nodes)
        offset_parallel_multilinks(edges)
        remove_too_short_edges(edges)
        reassign_ids(nodes, edges)
        return nodes, edges, topology, orientation
    elif topology in NETWORK_CODE_TOPOLOGIES:
        nodes, edges, orientation = build_network_code_case(topology, style_mode, index, seed, difficulty, rng, lookup)
        add_duplicate_suffixes(nodes, rng)
        ensure_no_isolated_nodes(nodes, edges, rng)
        densify_edges_for_difficulty(nodes, edges, difficulty, rng)
        apply_multilink_policy(edges, difficulty, rng)
        offset_parallel_multilinks(edges)
        stagger_edge_endpoints(edges, nodes)
        offset_parallel_multilinks(edges)
        remove_too_short_edges(edges)
        reassign_ids(nodes, edges)
        return nodes, edges, topology, orientation
    elif topology == "small_parallel_control":
        roles = ["power", "control", "passive", "active", "load", "ground"]
        centers = [(115, 350), (305, 350), (520, 235), (520, 465), (735, 350), (895, 350)]
        pairs = [(0, 1, "power"), (1, 2, "signal"), (1, 3, "wire"), (2, 4, "signal"), (3, 4, "wire"), (4, 5, "ground")]
    elif topology == "foodweb_tiered_circuit":
        if difficulty == "simple":
            roles = ["power", "sensor", "passive", "control", "load", "ground"]
            centers = [(135, 235), (135, 465), (385, 250), (385, 455), (650, 350), (875, 350)]
            pairs = [(0, 2, "power"), (1, 3, "signal"), (2, 4, "wire"), (3, 4, "signal"), (4, 5, "ground")]
        elif difficulty == "difficult" or complexity == "complex":
            roles = ["power", "sensor", "sensor", "passive", "control", "active", "active", "load", "load", "ground"]
            centers = [(110, 155), (110, 350), (110, 545), (345, 205), (345, 350), (345, 495), (585, 275), (585, 455), (780, 350), (930, 350)]
            pairs = [(0, 3, "power"), (0, 4, "power"), (1, 4, "signal"), (2, 5, "signal"), (3, 6, "wire"), (4, 6, "signal"), (4, 7, "signal"), (5, 7, "wire"), (6, 8, "signal"), (7, 8, "signal"), (8, 9, "ground")]
        else:
            roles = ["power", "sensor", "passive", "control", "active", "load", "load", "ground"]
            centers = [(120, 210), (120, 490), (355, 210), (355, 490), (585, 350), (770, 235), (770, 465), (925, 350)]
            pairs = [(0, 2, "power"), (0, 3, "power"), (1, 3, "signal"), (2, 4, "wire"), (3, 4, "signal"), (4, 5, "signal"), (4, 6, "signal"), (5, 7, "ground"), (6, 7, "ground")]
    elif topology == "multi_link_pair":
        roles = ["power", "sensor", "control", "load", "ground"]
        centers = [(115, 350), (355, 170), (355, 350), (650, 350), (875, 350)]
        pairs = [(0, 2, "power"), (1, 2, "signal"), (2, 3, "power"), (2, 3, "signal"), (2, 3, "wire"), (3, 4, "ground")]
    elif topology == "series_chain":
        roles = ["power", "control", "passive", "load", "ground"]
        centers = [(115, 350), (300, 350), (485, 350), (670, 350), (855, 350)]
        pairs = [(0, 1, "power"), (1, 2, "wire"), (2, 3, "wire"), (3, 4, "ground")]
    elif topology == "parallel_branches":
        roles = ["power", "control", "passive", "active", "load", "ground"]
        centers = [(115, 350), (300, 350), (500, 205), (500, 350), (500, 495), (815, 350)]
        pairs = [(0, 1, "power"), (1, 2, "wire"), (1, 3, "wire"), (1, 4, "wire"), (2, 5, "ground"), (3, 5, "ground"), (4, 5, "ground")]
    elif topology == "controller_io":
        roles = ["power", "sensor", "sensor", "control", "load", "load", "ground"]
        centers = [(120, 150), (120, 310), (120, 500), (455, 350), (760, 230), (760, 470), (910, 350)]
        pairs = [(0, 3, "power"), (1, 3, "signal"), (2, 3, "signal"), (3, 4, "signal"), (3, 5, "signal"), (4, 6, "ground"), (5, 6, "ground")]
    elif topology == "power_distribution":
        roles = ["power", "power", "control", "sensor", "load", "load", "ground"]
        centers = [(110, 350), (290, 350), (485, 210), (485, 350), (680, 210), (680, 490), (875, 350)]
        pairs = [(0, 1, "power"), (1, 2, "power"), (1, 3, "power"), (2, 4, "signal"), (3, 5, "signal"), (4, 6, "ground"), (5, 6, "ground")]
    elif topology == "complex_controller_io":
        roles = ["power", "sensor", "sensor", "sensor", "control", "active", "load", "load", "load", "ground"]
        centers = [(115, 90), (115, 230), (115, 350), (115, 510), (400, 350), (600, 350), (805, 170), (805, 350), (805, 530), (930, 350)]
        pairs = [(0, 4, "power"), (1, 4, "signal"), (2, 4, "signal"), (3, 4, "signal"), (4, 5, "signal"), (5, 6, "signal"), (5, 7, "signal"), (5, 8, "signal"), (6, 9, "ground"), (7, 9, "ground"), (8, 9, "ground")]
    elif topology == "complex_power_tree":
        roles = ["power", "power", "control", "passive", "active", "sensor", "load", "load", "ground"]
        centers = [(90, 350), (255, 350), (430, 190), (430, 350), (430, 510), (640, 190), (640, 350), (640, 510), (875, 350)]
        pairs = [(0, 1, "power"), (1, 2, "power"), (1, 3, "power"), (1, 4, "power"), (2, 5, "signal"), (3, 6, "wire"), (4, 7, "signal"), (5, 8, "ground"), (6, 8, "ground"), (7, 8, "ground")]
    elif topology == "complex_bridge_feedback":
        roles = ["power", "control", "passive", "active", "sensor", "active", "load", "load", "ground"]
        centers = [(90, 350), (245, 350), (420, 190), (420, 510), (595, 190), (595, 510), (760, 280), (760, 470), (910, 350)]
        pairs = [(0, 1, "power"), (1, 2, "wire"), (1, 3, "wire"), (2, 4, "signal"), (3, 5, "signal"), (4, 6, "signal"), (5, 7, "signal"), (4, 7, "signal"), (5, 6, "signal"), (6, 8, "ground"), (7, 8, "ground")]
    elif topology == "multi_link_bus":
        roles = ["power", "control", "sensor", "passive", "active", "load", "load", "ground"]
        centers = [(95, 350), (280, 350), (280, 170), (280, 530), (545, 350), (770, 245), (770, 455), (925, 350)]
        pairs = [(0, 1, "power"), (2, 1, "signal"), (3, 1, "wire"), (1, 4, "power"), (1, 4, "signal"), (1, 4, "wire"), (4, 5, "signal"), (4, 6, "signal"), (5, 7, "ground"), (6, 7, "ground")]
    else:
        roles = ["power", "control", "sensor", "passive", "active", "active", "load", "load", "ground", "ground"]
        centers = [(95, 350), (250, 350), (410, 160), (410, 350), (410, 540), (610, 350), (790, 230), (790, 470), (930, 260), (930, 500)]
        pairs = [(0, 1, "power"), (1, 2, "signal"), (1, 3, "wire"), (1, 4, "wire"), (2, 5, "signal"), (3, 5, "signal"), (4, 5, "signal"), (5, 6, "signal"), (5, 7, "signal"), (6, 8, "ground"), (7, 9, "ground")]

    nodes = [node(role, center, idx + 1) for idx, (role, center) in enumerate(zip(roles, centers))]
    if difficulty == "difficult":
        augment_standard_difficult_nodes(nodes, [], style_mode, rng, lookup, used)
    add_duplicate_suffixes(nodes, rng)
    edges = [CircuitEdge(f"e{i+1}", nodes[s].id, nodes[t].id, edge_path(nodes[s], nodes[t], bends=False), etype) for i, (s, t, etype) in enumerate(pairs)]
    if difficulty == "difficult":
        augment_standard_difficult_nodes(nodes, edges, style_mode, rng, lookup, used)
    ensure_no_isolated_nodes(nodes, edges, rng)
    densify_standard_edges_for_difficulty(nodes, edges, difficulty, rng)
    if difficulty == "difficult" and topology in STANDARD_COMPLEX_TOPOLOGIES:
        apply_standard_difficult_multilink_policy(edges, rng)
    else:
        apply_multilink_policy(edges, difficulty, rng)
    offset_parallel_multilinks(edges)
    reroute_standard_difficult_edges(nodes, edges, difficulty)
    stagger_edge_endpoints(edges, nodes)
    offset_parallel_multilinks(edges)
    remove_too_short_edges(edges)
    reassign_ids(nodes, edges)
    return nodes, edges, topology, orientation


def reading_order(nodes: Sequence[CircuitNode]) -> List[CircuitNode]:
    return sorted(nodes, key=lambda node: (node.box[1] // 30, node.box[0], node.box[1]))


def reassign_ids(nodes: List[CircuitNode], edges: List[CircuitEdge]) -> None:
    mapping = {node.id: f"n{idx}" for idx, node in enumerate(reading_order(nodes), 1)}
    for node in nodes:
        node.id = mapping[node.id]
    for edge in edges:
        edge.source = mapping[edge.source]
        edge.target = mapping[edge.target]


def node_names(nodes: Sequence[CircuitNode]) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def midpoint(path: Sequence[Point]) -> Point:
    return path[len(path) // 2] if len(path) >= 3 else ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


def edge_record(edge: CircuitEdge, names: Dict[str, str]) -> Dict:
    visible = visible_edge_path(edge)
    arrow = list(visible[-1]) if edge.directed else None
    target_head = RENDERER.arrowhead_points(visible, size=13) if arrow else []
    return {
        "id": edge.id,
        "source": edge.source,
        "source_name": names[edge.source],
        "target": edge.target,
        "target_name": names[edge.target],
        "type": edge.edge_type,
        "path": [list(point) for point in edge.path],
        "visible_path": [list(point) for point in visible],
        "start": list(visible[0]),
        "mid": list(midpoint(visible)),
        "end": list(visible[-1]),
        "arrow": arrow,
        "arrowheads": {"target": target_head, "source": []},
    }


def node_record(node: CircuitNode) -> Dict:
    asset = Path(node.asset_path).name if node.asset_path else None
    return {"id": node.id, "name": node.label, "box": list(node.box), "center": list(node.center), "image_box": list(node.icon_box) if node.icon_box else None, "text_box": list(node.text_box), "asset": asset}


def final_graph(nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge]) -> Dict:
    names = node_names(nodes)
    relation_map = {node.id: [] for node in nodes}
    for edge in edges:
        rec = {"source": edge.source, "source_name": names[edge.source], "target": edge.target, "target_name": names[edge.target], "type": edge.edge_type}
        relation_map[edge.source].append(rec)
        if not edge.directed:
            relation_map[edge.target].append({"source": edge.target, "source_name": names[edge.target], "target": edge.source, "target_name": names[edge.source], "type": edge.edge_type})
    return {"nodes": [{"id": node.id, "name": node.label, "box": list(node.box), "relations": relation_map[node.id]} for node in reading_order(nodes)]}


def think_graph(nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge], negatives) -> str:
    names = node_names(nodes)
    outgoing = {node.id: [] for node in nodes}
    for edge in edges:
        outgoing[edge.source].append(edge)
        if not edge.directed:
            outgoing[edge.target].append(CircuitEdge(edge.id + "_rev", edge.target, edge.source, list(reversed(edge.path)), edge.edge_type, directed=False))
    lines = ["<THINK_GRAPH>", "<NODES>"]
    for node in reading_order(nodes):
        lines.append(f'<NODE id="{node.id}" name="{node.label}" box="{list(node.box)}"/>')
    lines.extend(["</NODES>", ""])
    for node in reading_order(nodes):
        lines.append(f'<CHECK_NODE id="{node.id}" name="{node.label}">')
        for edge in outgoing[node.id]:
            rec = edge_record(edge, names)
            lines.append(f'<TRACE target="{rec["target"]}" target_name="{rec["target_name"]}" start="{rec["start"]}" mid="{rec["mid"]}" end="{rec["end"]}" arrow="{rec["arrow"]}"/>')
            lines.append(f'<EDGE source="{rec["source"]}" source_name="{rec["source_name"]}" target="{rec["target"]}" target_name="{rec["target_name"]}" type="{rec["type"]}"/>')
        lines.extend(["</CHECK_NODE>", ""])
    lines.append("</THINK_GRAPH>")
    return "\n".join(lines)


def training_text(nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge], negatives) -> str:
    return think_graph(nodes, edges, negatives) + "\n\n<FINAL_GRAPH>\n" + json.dumps(final_graph(nodes, edges), indent=2) + "\n</FINAL_GRAPH>"


def wrap_label(text: str, width: int) -> List[str]:
    return (textwrap.wrap(text, width=max(8, width // 8)) or [""])[:2]


def render_style(style_mode: str, sample_key: str, seed: int) -> Dict:
    rng = random.Random(f"{seed}:{sample_key}:style")
    backgrounds = [(255, 255, 255), (246, 250, 255), (250, 250, 247), (252, 248, 242)]
    return {
        "edge_color": list(EDGE_STYLES["wire"]["color"]),
        "edge_width": 3,
        "arrow_size": 13,
        "background": list(rng.choice(backgrounds)),
        "grid": rng.random() < 0.45,
        "halo": True,
    }


def draw_background(draw: ImageDraw.ImageDraw, style: Dict) -> None:
    pattern = style.get("background_pattern", "grid" if style.get("grid") else "none")
    color = (232, 237, 244)
    if pattern == "grid":
        for x in range(0, CANVAS_W + 1, 50):
            draw.line([(x, 0), (x, CANVAS_H)], fill=color, width=1)
        for y in range(0, CANVAS_H + 1, 50):
            draw.line([(0, y), (CANVAS_W, y)], fill=color, width=1)
    elif pattern == "dot":
        for x in range(25, CANVAS_W, 38):
            for y in range(25, CANVAS_H, 38):
                draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(225, 236, 226))
    elif pattern == "ruled":
        for y in range(42, CANVAS_H, 42):
            draw.line([(0, y), (CANVAS_W, y)], fill=(238, 231, 218), width=1)


def draw_node(image: Image.Image, draw: ImageDraw.ImageDraw, node: CircuitNode, font, foodweb_like: bool) -> None:
    if not foodweb_like:
        draw.rounded_rectangle(node.box, radius=6, fill=(255, 255, 255), outline=(62, 74, 92), width=2)
    if node.asset_path and node.icon_box:
        asset = Image.open(ROOT / node.asset_path).convert("RGBA")
        asset.thumbnail((node.icon_box[2] - node.icon_box[0], node.icon_box[3] - node.icon_box[1]))
        x = node.icon_box[0] + ((node.icon_box[2] - node.icon_box[0]) - asset.width) // 2
        y = node.icon_box[1] + ((node.icon_box[3] - node.icon_box[1]) - asset.height) // 2
        image.paste(asset, (x, y), asset if asset.mode == "RGBA" else None)
    else:
        x1, y1, x2, y2 = node.icon_box or (node.box[0] + 12, node.box[1] + 8, node.box[2] - 12, node.box[3] - 32)
        draw.rectangle((x1, y1, x2, y2), fill=(245, 247, 248), outline=None)
        draw.line((x1 + 10, y2 - 12, x2 - 10, y1 + 10), fill=(70, 74, 82), width=3)
    tx1, ty1, tx2, ty2 = node.text_box
    lines = wrap_label(node.label, tx2 - tx1)
    line_h = 13
    y = ty1 + max(0, (ty2 - ty1 - line_h * len(lines)) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((tx1 + max(0, (tx2 - tx1 - (bbox[2] - bbox[0])) // 2), y), line, fill=(24, 28, 35), font=font)
        y += line_h


def draw_edge(draw: ImageDraw.ImageDraw, edge: CircuitEdge, edge_styles: Dict[str, Dict]) -> None:
    style = edge_styles[edge.edge_type]
    path = visible_edge_path(edge)
    if True:
        draw.line(path, fill=(255, 255, 255), width=style["width"] + 4)
    draw.line(path, fill=style["color"], width=style["width"])


def draw_arrow(draw: ImageDraw.ImageDraw, edge: CircuitEdge, edge_styles: Dict[str, Dict]) -> None:
    style = edge_styles[edge.edge_type]
    if not style["arrow"]:
        return
    path = visible_edge_path(edge)
    draw.polygon(RENDERER.arrowhead_points(path, size=13), fill=style["color"])


def draw_legend(draw: ImageDraw.ImageDraw, font, edge_styles: Dict[str, Dict], position: str) -> None:
    labels = list(edge_styles)
    col_w = 210
    row_h = 32
    cols = min(2, max(1, len(labels)))
    rows = math.ceil(len(labels) / cols)
    box_w = cols * col_w + 26
    box_h = rows * row_h + 40
    if position == "bottom_right":
        x = CANVAS_W - box_w + 14
        y = CANVAS_H - box_h + 16
    else:
        x, y = 34, 30
    draw.rounded_rectangle((x - 14, y - 16, x + cols * col_w + 12, y - 16 + rows * row_h + 24), radius=7, fill=(255, 255, 255), outline=(190, 198, 210), width=2)
    for idx, etype in enumerate(labels):
        style = edge_styles[etype]
        x0 = x + (idx % cols) * col_w
        y0 = y + (idx // cols) * row_h
        legend_width = max(style["width"] + 1, 4)
        draw.line([(x0, y0 + 11), (x0 + 48, y0 + 11)], fill=style["color"], width=legend_width)
        if style["arrow"]:
            draw.polygon(RENDERER.arrowhead_points([(x0, y0 + 11), (x0 + 48, y0 + 11)], size=11), fill=style["color"])
        draw.text((x0 + 60, y0), style["label"], fill=(18, 24, 32), font=font)


def render_image(sample_id: str, style_mode: str, nodes: Sequence[CircuitNode], edges: Sequence[CircuitEdge], out_path: Path, seed: int, topology: str, edge_styles: Dict[str, Dict], legend_pos: str) -> Dict:
    style = render_style(style_mode, sample_id, seed)
    style["legend_position"] = legend_pos
    style["edge_styles"] = {
        etype: {
            "color": list(spec["color"]),
            "width": spec["width"],
            "arrow": spec["arrow"],
            "label": spec["label"],
        }
        for etype, spec in edge_styles.items()
    }
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), tuple(style["background"]))
    draw = ImageDraw.Draw(image)
    draw_background(draw, style)
    for edge in edges:
        draw_edge(draw, edge, edge_styles)
    font = RENDERER.load_font(13)
    foodweb_like = topology == LAYERED_CIRCUIT_TOPOLOGY or topology == FOODWEB_CODE_TOPOLOGY or topology in NETWORK_CODE_TOPOLOGIES
    for node in nodes:
        draw_node(image, draw, node, font, foodweb_like)
    for edge in edges:
        draw_arrow(draw, edge, edge_styles)
    draw_legend(draw, RENDERER.load_font(16), edge_styles, legend_pos)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return style


def annotation_for(sample_id: str, split: str, style_mode: str, index: int, seed: int, image_path: Path, out_dir: Path, complexity: str = "normal", difficulty: str = "preview") -> Dict:
    set_canvas(BASE_CANVAS_W, BASE_CANVAS_H)
    nodes, edges, topology, orientation = build_case(style_mode, index, seed, difficulty, complexity)
    target_w, target_h = target_canvas_for(difficulty, topology, orientation)
    scale_case(nodes, edges, target_w, target_h)
    set_canvas(target_w, target_h)
    edge_styles = semantic_circuit_edge_styles(edges, sample_id, seed) if topology == LAYERED_CIRCUIT_TOPOLOGY else randomize_link_types(edges, sample_id, seed)
    legend_pos = legend_position(sample_id, seed)
    translate_case(nodes, edges, 0, 32 if legend_pos == "top_left" else -28)
    style = render_image(sample_id, style_mode, nodes, edges, image_path, seed, topology, edge_styles, legend_pos)
    names = node_names(nodes)
    return {
        "sample_id": sample_id,
        "image": str(image_path.relative_to(out_dir)),
        "split": split,
        "difficulty": difficulty if difficulty != "preview" else ("difficult" if complexity == "complex" else "preview"),
        "orientation": orientation,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "nodes": [node_record(node) for node in reading_order(nodes)],
        "edges": [edge_record(edge, names) for edge in edges],
        "negative_edges": [],
        "final_graph": final_graph(nodes, edges),
        "think_graph": think_graph(nodes, edges, []),
        "training_text": training_text(nodes, edges, []),
        "render_style": style,
        "generation": {"seed": seed, "global_index": index, "attempt": 0, "topology": topology, "style_mode": style_mode},
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
    parser.add_argument("--out", default="knossos_circuit_v1/outputs/preview_samples")
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--per-style", type=int, default=2)
    parser.add_argument("--complexity", choices=["normal", "complex"], default="normal")
    parser.add_argument("--preview-kind", choices=["style", "difficulty"], default="style")
    parser.add_argument("--per-difficulty", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    image_paths: List[Path] = []
    summary = []
    if args.preview_kind == "difficulty":
        for difficulty in ["simple", "medium", "difficult"]:
            for idx in range(1, args.per_difficulty + 1):
                style_mode = STYLE_MODES[(idx - 1) % len(STYLE_MODES)]
                sample_id = f"circuit_{difficulty}_{idx:03d}"
                image_path = out / "images" / difficulty / f"{sample_id}.png"
                annotation_path = out / "annotations" / difficulty / f"{sample_id}.json"
                annotation = annotation_for(sample_id, "preview", style_mode, idx, args.seed, image_path, out, "complex" if difficulty == "difficult" else "normal", difficulty)
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
                image_paths.append(image_path)
                summary.append({"sample_id": sample_id, "difficulty": difficulty, "style_mode": style_mode, "nodes": len(annotation["nodes"]), "edges": len(annotation["edges"]), "image": annotation["image"], "annotation": str(annotation_path.relative_to(out))})
    else:
        for style_mode in STYLE_MODES:
            for idx in range(1, args.per_style + 1):
                sample_id = f"circuit_{style_mode}_{idx:03d}"
                image_path = out / "images" / style_mode / f"{sample_id}.png"
                annotation_path = out / "annotations" / style_mode / f"{sample_id}.json"
                annotation = annotation_for(sample_id, "preview", style_mode, idx, args.seed, image_path, out, args.complexity)
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
                image_paths.append(image_path)
                summary.append({"sample_id": sample_id, "style_mode": style_mode, "nodes": len(annotation["nodes"]), "edges": len(annotation["edges"]), "image": annotation["image"], "annotation": str(annotation_path.relative_to(out))})
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(image_paths, out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
