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
LABELS_PATH = ROOT / "config" / "map_labels.json"
ASSET_META_PATH = ROOT / "assets" / "renderer_asset_pool" / "metadata.json"
ASSET_POOL = ROOT / "assets" / "renderer_asset_pool"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDERER = load_module("knossos_map_renderer_base", RENDERER_PATH)

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

BASE_CANVAS_W = RENDERER.CANVAS_W
BASE_CANVAS_H = RENDERER.CANVAS_H
CANVAS_W = BASE_CANVAS_W
CANVAS_H = BASE_CANVAS_H
EDGE_CLEARANCE = -5

STYLE_MODES = ["flat_map_icon", "place_scene_icon", "marker_icon", "mixed"]
LAYOUT_FOODWEB = "spatial_free_map"
NETWORK_LAYOUTS = ["transit_star", "transit_layered", "route_cluster_bridge"]
TEMPLATE_LAYOUTS = ["subway_line", "floorplan_rooms", "warehouse_route"]
MAP_LAYOUTS = ["city_grid", "campus_grid", "indoor_corridor", "transit_grid"]
LAYOUTS = [LAYOUT_FOODWEB] + NETWORK_LAYOUTS + TEMPLATE_LAYOUTS + MAP_LAYOUTS
ROLE_COLORS = {
    "transit": (44, 105, 180),
    "indoor": (104, 88, 170),
    "campus": (54, 135, 86),
    "warehouse": (176, 103, 38),
    "landmark": (170, 76, 96),
    "utility": (72, 86, 99),
}


@dataclass
class MapNode:
    id: str
    label: str
    box: BBox
    role: str
    category: str
    asset_path: str | None
    text_only: bool = False

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def icon_box(self) -> BBox | None:
        if self.text_only or self.asset_path is None:
            return None
        x1, y1, x2, y2 = self.box
        return (x1 + 6, y1 + 4, x2 - 6, y2 - 28)

    @property
    def text_box(self) -> BBox:
        x1, y1, x2, y2 = self.box
        if self.text_only:
            return (x1 + 8, y1 + 8, x2 - 8, y2 - 8)
        return (x1, y2 - 26, x2, y2)


@dataclass
class MapEdge:
    id: str
    source: str
    target: str
    path: List[Point]
    edge_type: str
    directed: bool = False


def set_canvas(width: int, height: int) -> None:
    global CANVAS_W, CANVAS_H
    CANVAS_W = width
    CANVAS_H = height


def target_canvas(sample_id: str, seed: int) -> Tuple[int, int]:
    rng = random.Random(f"{seed}:{sample_id}:canvas")
    return rng.choice([(900, 630), (630, 900)])


def scale_point(point: Point, sx: float, sy: float) -> Point:
    return (int(round(point[0] * sx)), int(round(point[1] * sy)))


def scale_box(box: BBox, sx: float, sy: float) -> BBox:
    x1, y1, x2, y2 = box
    return (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))


def scale_case(nodes: Sequence[MapNode], edges: Sequence[MapEdge], width: int, height: int) -> None:
    sx = width / BASE_CANVAS_W
    sy = height / BASE_CANVAS_H
    for node in nodes:
        node.box = scale_box(node.box, sx, sy)
    for edge in edges:
        edge.path = [scale_point(point, sx, sy) for point in edge.path]


def translate_box(box: BBox, dx: int, dy: int) -> BBox:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def translate_path(path: Sequence[Point], dx: int, dy: int) -> List[Point]:
    return [(x + dx, y + dy) for x, y in path]


def translate_case(nodes: Sequence[MapNode], edges: Sequence[MapEdge], dx: int, dy: int) -> None:
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


def legend_position(sample_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:{sample_id}:legend")
    return rng.choice(["top_left", "bottom_right"])


def load_label_data() -> Dict:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def load_labels() -> List[Dict]:
    return load_label_data()["map"]


def labels_by_role() -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for item in load_labels():
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


def choose_label(role: str, rng: random.Random, used: set[str]) -> Dict:
    groups = labels_by_role()
    options = [item for item in groups[role] if item["label"] not in used] or groups[role]
    item = rng.choice(options)
    used.add(item["label"])
    return item


def display_name(item: Dict, rng: random.Random) -> str:
    name = item["display"]
    if rng.random() < 0.38:
        suffix = rng.choice(["A", "B", "C", "N", "S", "X", "Y", "01", "02", "03", "North", "East"])
        return f"{name} {suffix}"
    return name


def asset_for(label: str, style_mode: str, rng: random.Random, lookup: Dict[Tuple[str, str], List[Path]]) -> Path | None:
    if style_mode == "mixed":
        style = rng.choices(["flat_map_icon", "place_scene_icon", "marker_icon"], weights=[45, 40, 15], k=1)[0]
    else:
        style = style_mode
    options = lookup.get((label, style), [])
    return rng.choice(options) if options else None


def make_node(node_id: str, item: Dict, center: Point, size: Tuple[int, int], style_mode: str, rng: random.Random, lookup, text_only_rate: float) -> MapNode:
    x, y = center
    label = display_name(item, rng)
    text_only = rng.random() < text_only_rate
    if text_only:
        words = label.split()
        longest = max((len(word) for word in words), default=len(label))
        w = min(128, max(58, longest * 7 + 26, len(label) * 5 + 18))
        h = 30 if len(label) <= 14 else 38
    else:
        w, h = size
    box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    asset = None if text_only else asset_for(item["label"], style_mode, rng, lookup)
    return MapNode(f"tmp{node_id}", label, box, item["role"], item["category"], str(asset.relative_to(ROOT)) if asset else None, text_only)


def add_duplicate_suffixes(nodes: Sequence[MapNode], rng: random.Random) -> None:
    groups: Dict[str, List[MapNode]] = {}
    for node in nodes:
        groups.setdefault(node.label.lower(), []).append(node)
    suffixes = ["A", "B", "C", "1", "2", "3", "X", "Y"]
    for group in groups.values():
        if len(group) <= 1:
            continue
        rng.shuffle(suffixes)
        for idx, node in enumerate(group):
            node.label = f"{node.label} {suffixes[idx % len(suffixes)]}"


def layout_family(index: int) -> str:
    slot = (index - 1) % 10
    if slot < 4:
        return "foodweb"
    if slot < 7:
        return "network"
    return "template"


def layout_family_for_mode(index: int, layout_mode: str = "mixed") -> str:
    if layout_mode == "map_only":
        return "map_grid"
    if layout_mode == "map85":
        slot = (index - 1) % 20
        if slot < 17:
            return "map_grid"
        if slot == 17:
            return "network"
        if slot == 18:
            return "template"
        return "foodweb"
    if layout_mode == "map70":
        slot = (index - 1) % 10
        if slot < 7:
            return "map_grid"
        if slot == 7:
            return "foodweb"
        if slot == 8:
            return "network"
        return "template"
    return layout_family(index)


def choose_layout(difficulty: str, index: int, layout_mode: str = "mixed") -> str:
    if layout_mode == "map_only":
        return MAP_LAYOUTS[(index - 1) % len(MAP_LAYOUTS)]
    if layout_mode == "map85":
        family = layout_family_for_mode(index, layout_mode)
        if family == "map_grid":
            return MAP_LAYOUTS[(index - 1) % len(MAP_LAYOUTS)]
        if family == "network":
            return "transit_layered" if difficulty == "simple" else "route_cluster_bridge"
        if family == "template":
            return TEMPLATE_LAYOUTS[(index - 19) % len(TEMPLATE_LAYOUTS)]
        return LAYOUT_FOODWEB
    if layout_mode == "map70":
        family = layout_family_for_mode(index, layout_mode)
        if family == "map_grid":
            return MAP_LAYOUTS[(index - 1) % len(MAP_LAYOUTS)]
        if family == "foodweb":
            return LAYOUT_FOODWEB
        if family == "network":
            pool = ["transit_layered"] if difficulty == "simple" else ["transit_layered", "route_cluster_bridge"]
            return pool[(index - 9) % len(pool)]
        return TEMPLATE_LAYOUTS[(index - 10) % len(TEMPLATE_LAYOUTS)]
    family = layout_family(index)
    if family == "foodweb":
        return LAYOUT_FOODWEB
    if family == "network":
        pool = ["transit_star", "transit_layered"] if difficulty == "simple" else NETWORK_LAYOUTS
        return pool[(index - 5) % len(pool)]
    pool = TEMPLATE_LAYOUTS
    return pool[(index - 8) % len(pool)]


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
    return [(int(cx + rx * math.cos(start + 2 * math.pi * idx / count)), int(cy + ry * math.sin(start + 2 * math.pi * idx / count))) for idx in range(count)]


def spaced_centers(count: int, rng: random.Random, difficulty: str) -> List[Point]:
    min_dist = {"simple": 185, "medium": 150, "difficult": 118}[difficulty]
    centers: List[Point] = []
    attempts = 0
    while len(centers) < count and attempts < count * 90:
        attempts += 1
        point = (rng.randint(100, 900), rng.randint(115, 600))
        if all(math.hypot(point[0] - prev[0], point[1] - prev[1]) >= min_dist for prev in centers):
            centers.append(point)
    while len(centers) < count:
        cols = math.ceil(math.sqrt(count * CANVAS_W / CANVAS_H))
        rows = math.ceil(count / cols)
        idx = len(centers)
        col = idx % cols
        row = idx // cols
        jitter = {"simple": 18, "medium": 24, "difficult": 30}[difficulty]
        x = int(120 + col * (760 / max(1, cols - 1)) + rng.randint(-jitter, jitter))
        y = int(120 + row * (470 / max(1, rows - 1)) + rng.randint(-jitter, jitter))
        centers.append((x, y))
    rng.shuffle(centers)
    return centers


def node_size(difficulty: str, layout: str) -> Tuple[int, int]:
    if layout in TEMPLATE_LAYOUTS or layout in MAP_LAYOUTS:
        return (94, 68) if difficulty == "simple" else ((84, 62) if difficulty == "medium" else (74, 58))
    if difficulty == "simple":
        return (86, 94)
    if difficulty == "medium":
        return (78, 88)
    return (68, 80)


def text_only_rate(difficulty: str) -> float:
    return {"simple": 0.10, "medium": 0.16, "difficult": 0.22}[difficulty]


def center_distance(a: MapNode, b: MapNode) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def route_path(src: MapNode, dst: MapNode, rng: random.Random, bends: bool) -> List[Point]:
    start = RENDERER.boundary_point_towards(src.box, dst.center)
    end = RENDERER.boundary_point_towards(dst.box, src.center)
    if not bends:
        return [start, end]
    if rng.random() < 0.55:
        mid = ((start[0] + end[0]) // 2 + rng.randint(-24, 24), (start[1] + end[1]) // 2 + rng.randint(-20, 20))
        return [start, mid, end]
    if rng.random() < 0.5:
        return [start, ((start[0] + end[0]) // 2, start[1]), ((start[0] + end[0]) // 2, end[1]), end]
    return [start, (start[0], (start[1] + end[1]) // 2), (end[0], (start[1] + end[1]) // 2), end]


def orthogonal_route(src: MapNode, dst: MapNode, via: Sequence[Point] | None = None) -> List[Point]:
    inner = list(via or [])
    if not inner:
        sx, sy = src.center
        tx, ty = dst.center
        if abs(sx - tx) < 18 or abs(sy - ty) < 18:
            inner = []
        elif abs(sx - tx) > abs(sy - ty):
            inner = [((sx + tx) // 2, sy), ((sx + tx) // 2, ty)]
        else:
            inner = [(sx, (sy + ty) // 2), (tx, (sy + ty) // 2)]
    start_target = inner[0] if inner else dst.center
    end_target = inner[-1] if inner else src.center
    start = RENDERER.boundary_point_towards(src.box, start_target)
    end = RENDERER.boundary_point_towards(dst.box, end_target)
    path = [start] + inner + [end]
    compact: List[Point] = []
    for point in path:
        if not compact or compact[-1] != point:
            compact.append(point)
    return compact


def add_routed_edge(edges: List[MapEdge], seen: set[Tuple[str, str]], src: MapNode, dst: MapNode, rng: random.Random, directed_rate: float, path: List[Point], allow_repeat: bool = False) -> None:
    key = tuple(sorted([src.id, dst.id]))
    if src.id == dst.id or (key in seen and not allow_repeat):
        return
    seen.add(key)
    edges.append(MapEdge(f"e{len(edges) + 1}", src.id, dst.id, path, "route", rng.random() < directed_rate))


def add_edge(edges: List[MapEdge], seen: set[Tuple[str, str]], src: MapNode, dst: MapNode, rng: random.Random, directed_rate: float, bends: bool, allow_repeat: bool = False) -> None:
    key = tuple(sorted([src.id, dst.id]))
    if src.id == dst.id or (key in seen and not allow_repeat):
        return
    seen.add(key)
    directed = rng.random() < directed_rate
    edges.append(MapEdge(f"e{len(edges) + 1}", src.id, dst.id, route_path(src, dst, rng, bends), "route", directed))


def build_spatial_free_map(difficulty: str, style_mode: str, index: int, seed: int, rng: random.Random, lookup):
    orientation = rng.choice(["bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"])
    if difficulty == "simple":
        tier_counts = [2, rng.randint(3, 4), rng.randint(1, 2)]
        extra_edges = rng.randint(1, 2)
    elif difficulty == "medium":
        tier_counts = [rng.randint(2, 3), rng.randint(4, 5), rng.randint(3, 4)]
        extra_edges = rng.randint(2, 3)
    else:
        tier_counts = [3, rng.randint(5, 6), 4]
        extra_edges = rng.randint(2, 3)
    roles = ["transit", "indoor", "campus", "warehouse", "landmark", "utility"]
    size = node_size(difficulty, LAYOUT_FOODWEB)
    used: set[str] = set()
    nodes: List[MapNode] = []

    def tier_center(tier_idx: int, local_idx: int, count: int) -> Point:
        if orientation in {"bottom_to_top", "top_to_bottom"}:
            y_bases = [535, 350, 145] if orientation == "bottom_to_top" else [145, 350, 535]
            step = 820 // count
            jitter = 8 if difficulty == "difficult" else 16
            x = 90 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
            return (x, y_bases[tier_idx] + rng.randint(-8, 8))
        x_bases = [120, 500, 850] if orientation == "left_to_right" else [850, 500, 120]
        step = 560 // count
        jitter = 6 if difficulty == "difficult" else 14
        y = 80 + step // 2 + local_idx * step + rng.randint(-jitter, jitter)
        return (x_bases[tier_idx] + rng.randint(-8, 8), y)

    by_tier: List[List[MapNode]] = []
    node_idx = 1
    for tier_idx, tier_count in enumerate(tier_counts):
        tier_nodes: List[MapNode] = []
        for local_idx in range(tier_count):
            item = choose_label(rng.choice(roles), rng, used)
            node = make_node(str(node_idx), item, tier_center(tier_idx, local_idx, tier_count), size, style_mode, rng, lookup, text_only_rate(difficulty))
            nodes.append(node)
            tier_nodes.append(node)
            node_idx += 1
        by_tier.append(tier_nodes)

    def sort_axis(items: Sequence[MapNode]) -> List[MapNode]:
        if orientation in {"bottom_to_top", "top_to_bottom"}:
            return sorted(items, key=lambda n: n.center[0])
        return sorted(items, key=lambda n: n.center[1])

    lower, middle, upper = [sort_axis(tier) for tier in by_tier]
    edges: List[MapEdge] = []
    seen: set[Tuple[str, str]] = set()
    directed_rate = {"simple": 0.20, "medium": 0.32, "difficult": 0.38}[difficulty]

    def aligned_target(src_idx: int, src_count: int, dst_nodes: Sequence[MapNode]) -> MapNode:
        if len(dst_nodes) == 1:
            return dst_nodes[0]
        pos = round(src_idx * (len(dst_nodes) - 1) / max(1, src_count - 1))
        return dst_nodes[min(len(dst_nodes) - 1, max(0, pos))]

    for idx, src in enumerate(lower):
        add_edge(edges, seen, src, aligned_target(idx, len(lower), middle), rng, directed_rate, bends=False)
    for idx, src in enumerate(middle):
        add_edge(edges, seen, src, aligned_target(idx, len(middle), upper), rng, directed_rate, bends=False)
    adjacent_pairs = [(lower, middle), (middle, upper)]
    for _ in range(extra_edges):
        src_tier, dst_tier = rng.choice(adjacent_pairs)
        src = rng.choice(src_tier)
        near = sorted(dst_tier, key=lambda other: center_distance(src, other))[:2]
        add_edge(edges, seen, src, rng.choice(near), rng, directed_rate, bends=False)
    return nodes, edges, orientation


def build_network_style(layout: str, difficulty: str, style_mode: str, index: int, seed: int, rng: random.Random, lookup):
    orientation = rng.choice(["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"])
    count = {"simple": rng.randint(6, 8), "medium": rng.randint(9, 11), "difficult": rng.randint(12, 15)}[difficulty]
    size = node_size(difficulty, layout)
    if layout == "transit_star":
        centers = [(500, 350)] + radial_points(count - 1, 500, 350, 355, 235)
    elif layout == "transit_layered":
        top = 2 if difficulty != "difficult" else 3
        mid = max(3, count // 3) if difficulty != "difficult" else 5
        rows = [top, mid, count - top - mid]
        centers = []
        for row_count, y in zip(rows, [115, 350, 585]):
            step = 780 // max(1, row_count - 1) if row_count > 1 else 0
            start_x = 110 if row_count > 1 else 500
            centers.extend([(start_x + i * step, y) for i in range(row_count)])
    else:
        centers = []
        for ci, cc in enumerate([(260, 350), (740, 350)]):
            n = math.ceil(count / 2) if ci == 0 else count - len(centers)
            centers.append(cc)
            centers.extend(radial_points(max(0, n - 1), cc[0], cc[1], 220, 180, start=-math.pi / 2 + ci * 0.25))
    centers = [transform_center(c, orientation) for c in centers[:count]]
    role_pool = ["transit", "campus", "landmark", "utility"]
    used: set[str] = set()
    nodes = [make_node(str(i + 1), choose_label(rng.choice(role_pool), rng, used), c, size, style_mode, rng, lookup, text_only_rate(difficulty)) for i, c in enumerate(centers)]
    edges: List[MapEdge] = []
    seen: set[Tuple[str, str]] = set()
    directed_rate = {"simple": 0.18, "medium": 0.30, "difficult": 0.36}[difficulty]
    if layout == "transit_star":
        hub = min(nodes, key=lambda n: abs(n.center[0] - 500) + abs(n.center[1] - 350))
        for node in nodes:
            if node is not hub:
                add_edge(edges, seen, hub, node, rng, directed_rate, bends=False)
    elif layout == "transit_layered":
        ordered = sorted(nodes, key=lambda n: (n.center[1] if orientation in {"top_to_bottom", "bottom_to_top"} else n.center[0]))
        rows = [ordered[: max(2, count // 4)], ordered[max(2, count // 4) : max(2, count // 4) + max(3, count // 3)], ordered[max(2, count // 4) + max(3, count // 3) :]]
        for node in rows[1]:
            add_edge(edges, seen, min(rows[0], key=lambda p: center_distance(p, node)), node, rng, directed_rate, bends=False)
        for node in rows[2]:
            add_edge(edges, seen, min(rows[1], key=lambda p: center_distance(p, node)), node, rng, directed_rate, bends=False)
    else:
        split = len(nodes) // 2
        clusters = [nodes[:split], nodes[split:]]
        hubs = []
        for cluster in clusters:
            hub = min(cluster, key=lambda n: sum(center_distance(n, o) for o in cluster))
            hubs.append(hub)
            for node in cluster:
                if node is not hub:
                    add_edge(edges, seen, hub, node, rng, directed_rate, bends=False)
        add_edge(edges, seen, hubs[0], hubs[1], rng, directed_rate, bends=False)
    return nodes, edges, orientation


def build_template_style(layout: str, difficulty: str, style_mode: str, index: int, seed: int, rng: random.Random, lookup):
    size = node_size(difficulty, layout)
    used: set[str] = set()
    role_pool = {
        "subway_line": ["transit", "landmark", "utility"],
        "floorplan_rooms": ["indoor", "utility"],
        "warehouse_route": ["warehouse", "utility"],
    }[layout]
    if layout == "subway_line":
        centers = [(95, 350), (250, 350), (405, 250), (405, 450), (590, 350), (745, 250), (745, 450), (900, 350)]
        pairs = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
    elif layout == "floorplan_rooms":
        centers = [(150, 180), (350, 180), (550, 180), (750, 180), (250, 360), (500, 360), (750, 360), (250, 540), (500, 540), (750, 540)]
        pairs = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (4, 7), (5, 8), (6, 9)]
    else:
        centers = [(100, 350), (260, 350), (420, 210), (420, 350), (420, 490), (620, 210), (620, 350), (620, 490), (850, 350)]
        pairs = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8)]
    if difficulty == "simple":
        centers = centers[:6]
        pairs = [p for p in pairs if p[0] < 6 and p[1] < 6]
    elif difficulty == "medium":
        centers = centers[:8]
        pairs = [p for p in pairs if p[0] < 8 and p[1] < 8]
    nodes = [make_node(str(i + 1), choose_label(rng.choice(role_pool), rng, used), c, size, style_mode, rng, lookup, text_only_rate(difficulty)) for i, c in enumerate(centers)]
    directed_rate = {"simple": 0.18, "medium": 0.30, "difficult": 0.36}[difficulty]
    edges: List[MapEdge] = []
    seen: set[Tuple[str, str]] = set()
    for s, t in pairs:
        add_edge(edges, seen, nodes[s], nodes[t], rng, directed_rate, bends=False)
    return nodes, edges, "left_to_right"


def build_map_grid_style(layout: str, difficulty: str, style_mode: str, index: int, seed: int, rng: random.Random, lookup):
    size = node_size(difficulty, layout)
    used: set[str] = set()
    role_pool = {
        "city_grid": ["transit", "landmark", "utility", "campus"],
        "campus_grid": ["campus", "landmark", "utility"],
        "indoor_corridor": ["indoor", "utility"],
        "transit_grid": ["transit", "landmark", "utility"],
    }[layout]
    if layout == "city_grid":
        coords = [(180, 150), (380, 150), (600, 150), (820, 150), (180, 350), (380, 350), (600, 350), (820, 350), (180, 550), (380, 550), (600, 550), (820, 550)]
        pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (0, 4), (4, 8), (1, 5), (5, 9), (2, 6), (6, 10), (3, 7), (7, 11)]
        shortcut_pairs = [(1, 6), (5, 10), (4, 9)]
        keep = {"simple": 7, "medium": 10, "difficult": 12}[difficulty]
    elif layout == "campus_grid":
        coords = [(230, 160), (500, 160), (770, 160), (230, 350), (500, 350), (770, 350), (230, 540), (500, 540), (770, 540)]
        pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]
        shortcut_pairs = [(0, 4), (4, 8), (2, 4)]
        keep = {"simple": 6, "medium": 8, "difficult": 9}[difficulty]
    elif layout == "indoor_corridor":
        coords = [(230, 120), (500, 120), (770, 120), (230, 260), (500, 260), (770, 260), (230, 440), (500, 440), (770, 440), (500, 590)]
        pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (1, 4), (4, 7), (7, 9), (0, 3), (2, 5), (3, 6), (5, 8)]
        shortcut_pairs = [(0, 4), (4, 8), (3, 7)]
        keep = {"simple": 6, "medium": 8, "difficult": 10}[difficulty]
    else:
        coords = [(140, 360), (300, 360), (460, 240), (460, 480), (620, 240), (620, 480), (800, 360), (300, 170), (800, 170), (800, 540)]
        pairs = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (1, 7), (4, 8), (6, 9)]
        shortcut_pairs = [(0, 3), (2, 6), (3, 6)]
        keep = {"simple": 6, "medium": 8, "difficult": 10}[difficulty]
    coords = coords[:keep]
    pairs = [pair for pair in pairs if pair[0] < keep and pair[1] < keep]
    if difficulty == "simple":
        pairs = pairs[: max(keep - 1, 5)]
    elif difficulty == "medium":
        pairs = pairs[: min(len(pairs), keep + 2)]
    nodes = [
        make_node(str(i + 1), choose_label(rng.choice(role_pool), rng, used), center, size, style_mode, rng, lookup, text_only_rate(difficulty) * 0.75)
        for i, center in enumerate(coords)
    ]
    directed_rate = {"simple": 0.18, "medium": 0.28, "difficult": 0.34}[difficulty]
    edges: List[MapEdge] = []
    seen: set[Tuple[str, str]] = set()
    for s, t in pairs:
        add_routed_edge(edges, seen, nodes[s], nodes[t], rng, directed_rate, orthogonal_route(nodes[s], nodes[t]))
    if difficulty == "difficult":
        candidates = [pair for pair in shortcut_pairs if pair[0] < keep and pair[1] < keep]
        rng.shuffle(candidates)
        for s, t in candidates[: rng.randint(1, 2)]:
            add_routed_edge(edges, seen, nodes[s], nodes[t], rng, directed_rate, orthogonal_route(nodes[s], nodes[t]))
        for edge in rng.sample(edges, k=min(rng.randint(1, 2), len(edges))):
            edges.append(MapEdge(f"e{len(edges) + 1}", edge.source, edge.target, list(edge.path), "route", edge.directed))
    return nodes, edges, "grid_map"


def ensure_no_isolated(nodes: Sequence[MapNode], edges: List[MapEdge], rng: random.Random) -> None:
    linked = {e.source for e in edges} | {e.target for e in edges}
    for node in nodes:
        if node.id in linked:
            continue
        target = min([other for other in nodes if other is not node], key=lambda other: center_distance(node, other))
        edges.append(MapEdge(f"e{len(edges) + 1}", node.id, target.id, route_path(node, target, rng, False), "route", rng.random() < 0.25))
        linked.update([node.id, target.id])


def offset_parallel_multilinks(edges: List[MapEdge]) -> None:
    grouped: Dict[Tuple[str, str], List[MapEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)
    for pair_edges in grouped.values():
        if len(pair_edges) <= 1:
            continue
        center = (len(pair_edges) - 1) / 2
        for idx, edge in enumerate(pair_edges):
            start, end = edge.path[0], edge.path[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy) or 1
            nx, ny = -dy / length, dx / length
            offset = (idx - center) * 13
            edge.path = [(int(round(x + nx * offset)), int(round(y + ny * offset))) for x, y in edge.path]


def intervals_overlap(a: Tuple[int, int], b: Tuple[int, int], pad: int = 4) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1]) + pad


def offset_shared_axis_segments(edges: List[MapEdge], gap: int = 10) -> None:
    segment_groups: Dict[Tuple[str, int], List[Tuple[int, Tuple[int, int]]]] = {}
    for edge_idx, edge in enumerate(edges):
        for p1, p2 in zip(edge.path, edge.path[1:]):
            x1, y1 = p1
            x2, y2 = p2
            if abs(y1 - y2) <= 2:
                segment_groups.setdefault(("h", round((y1 + y2) / 4) * 4), []).append((edge_idx, tuple(sorted((x1, x2)))))
            elif abs(x1 - x2) <= 2:
                segment_groups.setdefault(("v", round((x1 + x2) / 4) * 4), []).append((edge_idx, tuple(sorted((y1, y2)))))

    shifts: Dict[int, Tuple[int, int]] = {}
    for (orient, _coord), records in segment_groups.items():
        if len(records) <= 1:
            continue
        records = sorted(records, key=lambda item: (item[1][0], item[1][1], item[0]))
        clusters: List[List[Tuple[int, Tuple[int, int]]]] = []
        for record in records:
            if not clusters or not intervals_overlap(clusters[-1][-1][1], record[1]):
                clusters.append([record])
            else:
                clusters[-1].append(record)
        for cluster in clusters:
            edge_ids = sorted({edge_idx for edge_idx, _interval in cluster})
            if len(edge_ids) <= 1:
                continue
            center = (len(edge_ids) - 1) / 2
            for pos, edge_idx in enumerate(edge_ids):
                if edge_idx in shifts:
                    continue
                offset = int(round((pos - center) * gap))
                shifts[edge_idx] = (0, offset) if orient == "h" else (offset, 0)

    for edge_idx, (dx, dy) in shifts.items():
        edges[edge_idx].path = [(x + dx, y + dy) for x, y in edges[edge_idx].path]


def side_for_point(box: BBox, point: Point) -> str:
    x1, y1, x2, y2 = box
    x, y = point
    distances = {
        "left": abs(x - x1),
        "right": abs(x - x2),
        "top": abs(y - y1),
        "bottom": abs(y - y2),
    }
    return min(distances, key=distances.get)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def shifted_anchor(box: BBox, side: str, point: Point, offset: int) -> Point:
    x1, y1, x2, y2 = box
    x, y = point
    margin = 8
    if side in {"left", "right"}:
        return (x1 if side == "left" else x2, clamp(y + offset, y1 + margin, y2 - margin))
    return (clamp(x + offset, x1 + margin, x2 - margin), y1 if side == "top" else y2)


def adjust_neighbor_for_anchor(path: List[Point], endpoint_idx: int, old_point: Point, new_point: Point) -> None:
    if len(path) < 2 or old_point == new_point:
        return
    neighbor_idx = 1 if endpoint_idx == 0 else len(path) - 2
    nx, ny = path[neighbor_idx]
    ox, oy = old_point
    px, py = new_point
    if abs(nx - ox) <= 2:
        path[neighbor_idx] = (px, ny)
    elif abs(ny - oy) <= 2:
        path[neighbor_idx] = (nx, py)


def spread_incident_node_anchors(nodes: Sequence[MapNode], edges: List[MapEdge], gap: int = 12) -> None:
    node_by_id = {node.id: node for node in nodes}
    groups: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
    for edge_idx, edge in enumerate(edges):
        src = node_by_id[edge.source]
        dst = node_by_id[edge.target]
        groups.setdefault((edge.source, "source", side_for_point(src.box, edge.path[0])), []).append((edge_idx, 0))
        groups.setdefault((edge.target, "target", side_for_point(dst.box, edge.path[-1])), []).append((edge_idx, -1))

    for (node_id, _kind, side), refs in groups.items():
        if len(refs) <= 1:
            continue
        node = node_by_id[node_id]
        refs.sort(key=lambda ref: edges[ref[0]].path[1 if ref[1] == 0 else -2])
        center = (len(refs) - 1) / 2
        for pos, (edge_idx, endpoint_idx) in enumerate(refs):
            path_idx = 0 if endpoint_idx == 0 else len(edges[edge_idx].path) - 1
            old_point = edges[edge_idx].path[path_idx]
            offset = int(round((pos - center) * gap))
            new_point = shifted_anchor(node.box, side, old_point, offset)
            adjust_neighbor_for_anchor(edges[edge_idx].path, path_idx, old_point, new_point)
            edges[edge_idx].path[path_idx] = new_point


def add_multilinks(edges: List[MapEdge], difficulty: str, rng: random.Random) -> None:
    roll = rng.random()
    mode = "none"
    if difficulty == "simple" and roll < 0.10:
        mode = "light"
    elif difficulty == "medium":
        mode = "heavy" if roll < 0.08 else ("light" if roll < 0.35 else "none")
    elif difficulty == "difficult":
        mode = "heavy" if roll < 0.10 else ("light" if roll < 0.42 else "none")
    if mode == "none" or not edges:
        return
    rng.shuffle(edges)
    for edge in edges[: 1 if mode == "light" else 2]:
        target_count = 2 if mode == "light" else 3
        while sum(1 for e in edges if e.source == edge.source and e.target == edge.target) < target_count:
            edges.append(MapEdge(f"e{len(edges) + 1}", edge.source, edge.target, list(edge.path), "route", edge.directed))


def reading_order(nodes: Sequence[MapNode]) -> List[MapNode]:
    return sorted(nodes, key=lambda n: (n.box[1] // 30, n.box[0], n.box[1]))


def reassign_ids(nodes: List[MapNode], edges: List[MapEdge]) -> None:
    mapping = {node.id: f"n{idx}" for idx, node in enumerate(reading_order(nodes), 1)}
    for node in nodes:
        node.id = mapping[node.id]
    for edge in edges:
        edge.source = mapping[edge.source]
        edge.target = mapping[edge.target]


def build_case(style_mode: str, index: int, seed: int, difficulty: str, layout_mode: str = "mixed"):
    rng = random.Random(f"{seed}:{difficulty}:{index}:{style_mode}")
    layout = choose_layout(difficulty, index, layout_mode)
    lookup = asset_lookup()
    if layout == LAYOUT_FOODWEB:
        nodes, edges, orientation = build_spatial_free_map(difficulty, style_mode, index, seed, rng, lookup)
    elif layout in NETWORK_LAYOUTS:
        nodes, edges, orientation = build_network_style(layout, difficulty, style_mode, index, seed, rng, lookup)
    elif layout in MAP_LAYOUTS:
        nodes, edges, orientation = build_map_grid_style(layout, difficulty, style_mode, index, seed, rng, lookup)
    else:
        nodes, edges, orientation = build_template_style(layout, difficulty, style_mode, index, seed, rng, lookup)
    add_duplicate_suffixes(nodes, rng)
    ensure_no_isolated(nodes, edges, rng)
    add_multilinks(edges, difficulty, rng)
    offset_parallel_multilinks(edges)
    if layout in MAP_LAYOUTS:
        offset_shared_axis_segments(edges, gap=12)
        spread_incident_node_anchors(nodes, edges, gap=12)
        offset_parallel_multilinks(edges)
    reassign_ids(nodes, edges)
    for idx, edge in enumerate(edges, 1):
        edge.id = f"e{idx}"
    return nodes, edges, layout, orientation


def distinct_colors(count: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    start = rng.random()
    hues = [((start + idx / max(1, count)) % 1.0) for idx in range(count)]
    rng.shuffle(hues)
    colors = []
    for idx, hue in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(hue, 0.68 + 0.18 * ((idx % 3) / 2), 0.48 + 0.24 * (((idx + 1) % 3) / 2))
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def direction_mode(sample_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:{sample_id}:direction-mode")
    roll = rng.random()
    if roll < 0.60:
        return "directed"
    if roll < 0.85:
        return "undirected"
    return "mixed"


def route_type_labels(edges: List[MapEdge], sample_id: str, seed: int, mode: str) -> Dict[str, Dict]:
    rng = random.Random(f"{seed}:{sample_id}:route-types")
    stems = load_label_data()["route_type_stems"]
    grouped: Dict[Tuple[str, str], List[MapEdge]] = {}
    for edge in edges:
        grouped.setdefault(tuple(sorted([edge.source, edge.target])), []).append(edge)
    max_parallel = max((len(group) for group in grouped.values()), default=1)
    total_types = rng.randint(max(3, max_parallel), min(5, max(3, len(edges))))
    labels = []
    used = set()
    while len(labels) < total_types:
        label = f"{rng.choice(stems)} {rng.choice(['A','B','C','1','2','3','X','Y','N','S'])}"
        if label.lower() not in used:
            used.add(label.lower())
            labels.append(label)
    colors = distinct_colors(total_types, rng)
    styles = {}
    for idx, label in enumerate(labels):
        if mode == "directed":
            arrow = True
        elif mode == "undirected":
            arrow = False
        else:
            
            
            
            arrow = idx < max(1, math.ceil(total_types * 0.62))
        styles[label] = {"color": colors.pop(), "width": rng.choice([2, 2, 3, 3, 4]), "arrow": arrow, "label": label}
    if mode == "mixed" and len(styles) > 1:
        if all(style["arrow"] for style in styles.values()):
            styles[labels[-1]]["arrow"] = False
        if not any(style["arrow"] for style in styles.values()):
            styles[labels[0]]["arrow"] = True
    all_labels = list(labels)
    for group in grouped.values():
        if len(group) > 1:
            assigned = rng.sample(all_labels, k=min(len(group), len(all_labels)))
            if len(assigned) < len(group):
                assigned.extend(rng.choices(all_labels, k=len(group) - len(assigned)))
            for edge, label in zip(group, assigned):
                edge.edge_type = label
                edge.directed = styles[label]["arrow"]
        else:
            edge = group[0]
            preferred = [label for label in all_labels if styles[label]["arrow"] == edge.directed]
            label = rng.choice(preferred or all_labels)
            edge.edge_type = label
            edge.directed = styles[label]["arrow"]
    return styles


def visible_edge_path(edge: MapEdge) -> List[Point]:
    path = list(edge.path)
    path[0] = RENDERER.offset_point(path[0], path[1], EDGE_CLEARANCE)
    path[-1] = RENDERER.offset_point(path[-1], path[-2], EDGE_CLEARANCE)
    return path


def midpoint(path: Sequence[Point]) -> Point:
    return path[len(path) // 2] if len(path) >= 3 else ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


def node_names(nodes: Sequence[MapNode]) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def edge_record(edge: MapEdge, names: Dict[str, str]) -> Dict:
    visible = visible_edge_path(edge)
    arrow = list(visible[-1]) if edge.directed else None
    return {
        "id": edge.id,
        "source": edge.source,
        "source_name": names[edge.source],
        "target": edge.target,
        "target_name": names[edge.target],
        "type": edge.edge_type,
        "path": [list(p) for p in edge.path],
        "visible_path": [list(p) for p in visible],
        "start": list(visible[0]),
        "mid": list(midpoint(visible)),
        "end": list(visible[-1]),
        "arrow": arrow,
        "arrowheads": {"target": RENDERER.arrowhead_points(visible, size=13) if arrow else [], "source": []},
    }


def node_record(node: MapNode) -> Dict:
    return {"id": node.id, "name": node.label, "box": list(node.box), "center": list(node.center), "image_box": list(node.icon_box) if node.icon_box else None, "text_box": list(node.text_box), "asset": Path(node.asset_path).name if node.asset_path else None}


def final_graph(nodes: Sequence[MapNode], edges: Sequence[MapEdge]) -> Dict:
    names = node_names(nodes)
    relation_map = {node.id: [] for node in nodes}
    for edge in edges:
        relation_map[edge.source].append({"source": edge.source, "source_name": names[edge.source], "target": edge.target, "target_name": names[edge.target], "type": edge.edge_type})
        if not edge.directed:
            relation_map[edge.target].append({"source": edge.target, "source_name": names[edge.target], "target": edge.source, "target_name": names[edge.source], "type": edge.edge_type})
    return {"nodes": [{"id": node.id, "name": node.label, "box": list(node.box), "relations": relation_map[node.id]} for node in reading_order(nodes)]}


def think_graph(nodes: Sequence[MapNode], edges: Sequence[MapEdge], negatives) -> str:
    names = node_names(nodes)
    outgoing = {node.id: [] for node in nodes}
    for edge in edges:
        outgoing[edge.source].append(edge)
        if not edge.directed:
            outgoing[edge.target].append(MapEdge(edge.id + "_rev", edge.target, edge.source, list(reversed(edge.path)), edge.edge_type, directed=False))
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


def training_text(nodes: Sequence[MapNode], edges: Sequence[MapEdge], negatives) -> str:
    return think_graph(nodes, edges, negatives) + "\n\n<FINAL_GRAPH>\n" + json.dumps(final_graph(nodes, edges), indent=2) + "\n</FINAL_GRAPH>"


def render_style(sample_key: str, seed: int) -> Dict:
    rng = random.Random(f"{seed}:{sample_key}:style")
    return {
        "edge_color": [40, 80, 120],
        "edge_width": 3,
        "arrow_size": 13,
        "background": list(rng.choice([(255, 255, 255), (246, 250, 255), (250, 250, 247), (252, 248, 242)])),
        "grid": rng.random() < 0.55,
        "halo": True,
    }


def draw_background(draw: ImageDraw.ImageDraw, style: Dict) -> None:
    if not style["grid"]:
        return
    color = (232, 237, 244)
    for x in range(0, CANVAS_W + 1, 50):
        draw.line([(x, 0), (x, CANVAS_H)], fill=color, width=1)
    for y in range(0, CANVAS_H + 1, 50):
        draw.line([(0, y), (CANVAS_W, y)], fill=color, width=1)


def wrap_label(text: str, width: int) -> List[str]:
    return (textwrap.wrap(text, width=max(8, width // 8)) or [""])[:2]


def draw_node(image: Image.Image, draw: ImageDraw.ImageDraw, node: MapNode, font) -> None:
    if node.text_only:
        role_color = ROLE_COLORS.get(node.role, (72, 86, 99))
        draw.rounded_rectangle(node.box, radius=9, fill=(255, 255, 255), outline=(role_color[0], role_color[1], role_color[2]), width=1)
        cx = node.box[0] + 10
        cy = (node.box[1] + node.box[3]) // 2
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=role_color)
    if node.asset_path and node.icon_box:
        asset = Image.open(ROOT / node.asset_path).convert("RGBA")
        asset.thumbnail((node.icon_box[2] - node.icon_box[0], node.icon_box[3] - node.icon_box[1]))
        x = node.icon_box[0] + ((node.icon_box[2] - node.icon_box[0]) - asset.width) // 2
        y = node.icon_box[1] + ((node.icon_box[3] - node.icon_box[1]) - asset.height) // 2
        image.paste(asset, (x, y), asset if asset.mode == "RGBA" else None)
    tx1, ty1, tx2, ty2 = node.text_box
    lines = wrap_label(node.label, tx2 - tx1)
    line_h = 12
    y = ty1 + max(0, (ty2 - ty1 - line_h * len(lines)) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        if node.text_only:
            x = tx1 + 16
        else:
            x = tx1 + max(0, (tx2 - tx1 - (bbox[2] - bbox[0])) // 2)
        draw.text((x, y), line, fill=(24, 28, 35), font=font)
        y += line_h


def draw_edge(draw: ImageDraw.ImageDraw, edge: MapEdge, styles: Dict[str, Dict]) -> None:
    style = styles[edge.edge_type]
    path = visible_edge_path(edge)
    draw.line(path, fill=(255, 255, 255), width=style["width"] + 4)
    draw.line(path, fill=style["color"], width=style["width"], joint="curve")


def draw_arrow(draw: ImageDraw.ImageDraw, edge: MapEdge, styles: Dict[str, Dict]) -> None:
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


def render_image(sample_id: str, nodes: Sequence[MapNode], edges: Sequence[MapEdge], image_path: Path, seed: int, styles: Dict[str, Dict], legend_pos: str) -> Dict:
    style = render_style(sample_id, seed)
    style["legend_position"] = legend_pos
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), tuple(style["background"]))
    draw = ImageDraw.Draw(image)
    draw_background(draw, style)
    for edge in edges:
        draw_edge(draw, edge, styles)
    font = RENDERER.load_font(12)
    for node in nodes:
        draw_node(image, draw, node, font)
    for edge in edges:
        draw_arrow(draw, edge, styles)
    draw_legend(draw, RENDERER.load_font(16), styles, legend_pos)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return style


def annotation_for(sample_id: str, split: str, style_mode: str, index: int, seed: int, image_path: Path, out_dir: Path, difficulty: str, layout_mode: str = "mixed") -> Dict:
    set_canvas(BASE_CANVAS_W, BASE_CANVAS_H)
    nodes, edges, layout, orientation = build_case(style_mode, index, seed, difficulty, layout_mode)
    canvas_w, canvas_h = target_canvas(sample_id, seed)
    scale_case(nodes, edges, canvas_w, canvas_h)
    set_canvas(canvas_w, canvas_h)
    mode = direction_mode(sample_id, seed)
    styles = route_type_labels(edges, sample_id, seed, mode)
    legend_pos = legend_position(sample_id, seed)
    translate_case(nodes, edges, 0, 32 if legend_pos == "top_left" else -28)
    style = render_image(sample_id, nodes, edges, image_path, seed, styles, legend_pos)
    names = node_names(nodes)
    return {
        "sample_id": sample_id,
        "image": str(image_path.relative_to(out_dir)),
        "split": split,
        "difficulty": difficulty,
        "orientation": orientation,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "nodes": [node_record(node) for node in reading_order(nodes)],
        "edges": [edge_record(edge, names) for edge in edges],
        "negative_edges": [],
        "final_graph": final_graph(nodes, edges),
        "think_graph": think_graph(nodes, edges, []),
        "training_text": training_text(nodes, edges, []),
        "render_style": style,
        "generation": {"seed": seed, "global_index": index, "attempt": 0, "layout": layout, "layout_mode": layout_mode, "direction_mode": mode},
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
    parser.add_argument("--out", default="knossos_map_v1/outputs/preview_samples")
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--per-difficulty", type=int, default=10)
    parser.add_argument("--layout-mode", choices=["mixed", "map_only", "map70", "map85"], default="mixed")
    args = parser.parse_args()

    out = Path(args.out)
    image_paths: List[Path] = []
    summary = []
    for difficulty in ["simple", "medium", "difficult"]:
        for idx in range(1, args.per_difficulty + 1):
            style_mode = STYLE_MODES[(idx - 1) % len(STYLE_MODES)]
            sample_id = f"map_{difficulty}_{idx:03d}"
            image_path = out / "images" / difficulty / f"{sample_id}.png"
            ann_path = out / "annotations" / difficulty / f"{sample_id}.json"
            ann = annotation_for(sample_id, "preview", style_mode, idx, args.seed, image_path, out, difficulty, args.layout_mode)
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")
            image_paths.append(image_path)
            family = layout_family_for_mode(idx, args.layout_mode)
            summary.append({"sample_id": sample_id, "difficulty": difficulty, "nodes": len(ann["nodes"]), "edges": len(ann["edges"]), "layout_family": family, "layout": ann["generation"]["layout"], "image": ann["image"], "annotation": str(ann_path.relative_to(out)), "edge_types": sorted({e["type"] for e in ann["edges"]})})
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(image_paths, out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
