#!/usr/bin/env python3


from __future__ import annotations

import argparse
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
LABELS_PATH = ROOT / "config" / "workflow_labels.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDERER = load_module("knossos_workflow_renderer_base", RENDERER_PATH)

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

BASE_CANVAS_W = RENDERER.CANVAS_W
BASE_CANVAS_H = RENDERER.CANVAS_H
CANVAS_W = BASE_CANVAS_W
CANVAS_H = BASE_CANVAS_H
EDGE_CLEARANCE = -5
REVERSE_EDGE_PROB = 0.0
REVERSE_EDGE_PROBS = {
    "simple": 0.0,
    "medium": 0.07,
    "difficult": 0.20,
}

TOPOLOGIES = {
    "simple": ["swimlane_workflow", "staged_grid_workflow", "ladder_feedback_workflow"],
    "medium": ["swimlane_workflow", "staged_grid_workflow", "ladder_feedback_workflow"],
    "difficult": ["swimlane_workflow", "staged_grid_workflow", "ladder_feedback_workflow"],
}

ORIENTATIONS = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]

PALETTE = {
    "terminal": ((232, 244, 255), (55, 92, 135)),
    "process": ((246, 248, 251), (66, 78, 96)),
    "decision": ((255, 248, 226), (133, 96, 33)),
    "database": ((236, 248, 244), (45, 112, 93)),
    "document": ((245, 241, 255), (88, 72, 132)),
    "input_output": ((237, 247, 255), (54, 105, 143)),
    "queue": ((251, 242, 235), (132, 82, 46)),
    "manual_operation": ((247, 244, 239), (105, 87, 61)),
    "message": ((239, 249, 241), (54, 119, 75)),
    "merge": ((242, 246, 250), (67, 84, 105)),
    "parallel": ((242, 246, 250), (67, 84, 105)),
    "delay": ((250, 245, 236), (125, 91, 51)),
}

DIFFICULT_EDGE_TYPES = [
    "approve",
    "reject",
    "retry",
    "notify",
    "escalate",
    "sync",
    "archive",
    "handoff",
    "review",
    "fallback",
]

EDGE_TYPE_PALETTES = [
    [(26, 84, 180), (196, 55, 61), (34, 139, 82), (133, 74, 170)],
    [(31, 119, 180), (214, 97, 38), (117, 112, 179), (44, 160, 44)],
    [(0, 105, 130), (162, 62, 72), (105, 81, 170), (82, 120, 42)],
]

SCALE_BY_DIFFICULTY = {
    "simple": (0.90, 0.96, 1.00),
    "medium": (0.96, 1.00, 1.06),
    "difficult": (1.00, 1.06, 1.12),
}


@dataclass
class WorkflowNode:
    id: str
    label: str
    box: BBox
    shape: str
    role: str
    category: str
    asset_path: str | None = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def text_box(self) -> BBox:
        x1, y1, x2, y2 = self.box
        if self.shape == "decision":
            return (x1 + 18, y1 + 17, x2 - 18, y2 - 17)
        return (x1 + 10, y1 + 8, x2 - 10, y2 - 8)

    @property
    def icon_box(self):
        return None


@dataclass
class WorkflowEdge:
    id: str
    source: str
    target: str
    path: List[Point]
    edge_type: str = "directed"


def load_labels() -> List[Dict[str, str]]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))["workflow"]


def labels_by_role() -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for item in load_labels():
        groups.setdefault(item["role"], []).append(item)
    return groups


def choose_label(role: str, rng: random.Random, used: set[str]) -> Dict[str, str]:
    groups = labels_by_role()
    options = [item for item in groups[role] if item["label"] not in used]
    if not options:
        options = groups[role]
    item = rng.choice(options)
    used.add(item["label"])
    return item


def display_name(item: Dict[str, str], rng: random.Random) -> str:
    name = item["display"]
    if item["role"] in {"start", "end"}:
        return name
    if rng.random() < 0.22:
        suffix = rng.choice(["A", "B", "C", "1", "2", "3"])
        return f"{name} {suffix}"
    return name


def size_for_shape(shape: str, difficulty: str) -> Tuple[int, int]:
    if difficulty == "difficult":
        if shape == "decision":
            return (98, 64)
        if shape == "terminator":
            return (76, 40)
        if shape in {"merge", "parallel"}:
            return (92, 42)
        if shape in {"database", "document"}:
            return (100, 46)
        return (96, 42)
    if difficulty == "medium":
        if shape == "decision":
            return (100, 66)
        if shape == "terminator":
            return (76, 40)
        if shape in {"database", "document"}:
            return (100, 46)
        if shape in {"merge", "parallel"}:
            return (96, 44)
        return (96, 44)
    if shape == "decision":
        return (92, 62)
    if shape == "terminator":
        return (70, 38)
    if shape in {"database", "document"}:
        return (92, 42)
    if shape in {"merge", "parallel"}:
        return (88, 40)
    return (90, 40)


def make_node(node_id: str, item: Dict[str, str], center: Point, difficulty: str, rng: random.Random) -> WorkflowNode:
    width, height = size_for_shape(item["shape"], difficulty)
    x, y = center
    box = (x - width // 2, y - height // 2, x + width // 2, y + height // 2)
    return WorkflowNode(node_id, display_name(item, rng), box, item["shape"], item["role"], item["category"])


def transform_point(point: Point, orientation: str) -> Point:
    x, y = point
    if orientation == "left_to_right":
        return (int(x / BASE_CANVAS_W * CANVAS_W), int(y / BASE_CANVAS_H * CANVAS_H))
    if orientation == "right_to_left":
        return (int((BASE_CANVAS_W - x) / BASE_CANVAS_W * CANVAS_W), int(y / BASE_CANVAS_H * CANVAS_H))
    if orientation == "top_to_bottom":
        return (int(y / BASE_CANVAS_H * CANVAS_W), int(x / BASE_CANVAS_W * CANVAS_H))
    if orientation == "bottom_to_top":
        return (int((BASE_CANVAS_H - y) / BASE_CANVAS_H * CANVAS_W), int((BASE_CANVAS_W - x) / BASE_CANVAS_W * CANVAS_H))
    return point


def set_canvas_for_case(difficulty: str, orientation: str) -> None:
    global CANVAS_W, CANVAS_H
    horizontal, vertical = (900, 630), (630, 900)
    if orientation in {"top_to_bottom", "bottom_to_top"}:
        CANVAS_W, CANVAS_H = vertical
    else:
        CANVAS_W, CANVAS_H = horizontal


def clamp_node(node: WorkflowNode) -> None:
    x1, y1, x2, y2 = node.box
    dx = 0
    dy = 0
    if x1 < 35:
        dx = 35 - x1
    elif x2 > CANVAS_W - 35:
        dx = CANVAS_W - 35 - x2
    if y1 < 35:
        dy = 35 - y1
    elif y2 > CANVAS_H - 35:
        dy = CANVAS_H - 35 - y2
    if dx or dy:
        node.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def build_nodes_from_layout(layout: List[Tuple[str, Point]], difficulty: str, orientation: str, rng: random.Random) -> List[WorkflowNode]:
    used: set[str] = set()
    nodes: List[WorkflowNode] = []
    for idx, (role, center) in enumerate(layout, 1):
        item = choose_label(role, rng, used)
        transformed = transform_point(center, orientation)
        node = make_node(f"tmp{idx}", item, transformed, difficulty, rng)
        clamp_node(node)
        nodes.append(node)
    return nodes


def reading_order(nodes: Sequence[WorkflowNode]) -> List[WorkflowNode]:
    return sorted(nodes, key=lambda node: (node.box[1] // 30, node.box[0], node.box[1]))


def reassign_ids(nodes: List[WorkflowNode], edges: List[WorkflowEdge]) -> None:
    mapping = {node.id: f"n{idx}" for idx, node in enumerate(reading_order(nodes), 1)}
    for node in nodes:
        node.id = mapping[node.id]
    for edge in edges:
        edge.source = mapping[edge.source]
        edge.target = mapping[edge.target]


def rect_boundary_point(box: BBox, target: Point) -> Point:
    return RENDERER.boundary_point_towards(box, target)


def side_center_anchor(box: BBox, side: str) -> Point:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if side == "left":
        return (x1, cy)
    if side == "right":
        return (x2, cy)
    if side == "top":
        return (cx, y1)
    if side == "bottom":
        return (cx, y2)
    return (cx, cy)


def anchor_pair(source: WorkflowNode, target: WorkflowNode) -> Tuple[Point, Point]:
    sx, sy = source.center
    tx, ty = target.center
    if abs(tx - sx) >= abs(ty - sy):
        if tx >= sx:
            return side_center_anchor(source.box, "right"), side_center_anchor(target.box, "left")
        return side_center_anchor(source.box, "left"), side_center_anchor(target.box, "right")
    if ty >= sy:
        return side_center_anchor(source.box, "bottom"), side_center_anchor(target.box, "top")
    return side_center_anchor(source.box, "top"), side_center_anchor(target.box, "bottom")


def visible_edge_path(edge: WorkflowEdge, clearance: int = EDGE_CLEARANCE) -> List[Point]:
    path = list(edge.path)
    if len(path) < 2:
        return path
    path[0] = RENDERER.offset_point(path[0], path[1], clearance)
    path[-1] = RENDERER.offset_point(path[-1], path[-2], clearance)
    return path


def edge_path(source: WorkflowNode, target: WorkflowNode, orientation: str, bends: bool = True) -> List[Point]:
    start, end = anchor_pair(source, target)
    if not bends:
        return [start, end]
    if orientation == "left_to_right":
        mid_x = (start[0] + end[0]) // 2
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]
    mid_y = (start[1] + end[1]) // 2
    return [start, (start[0], mid_y), (end[0], mid_y), end]


def add_edge(edges: List[WorkflowEdge], source: WorkflowNode, target: WorkflowNode, orientation: str, bends: bool = True) -> None:
    edges.append(WorkflowEdge(f"e{len(edges) + 1}", source.id, target.id, edge_path(source, target, orientation, bends)))


def reverse_random_edges(edges: List[WorkflowEdge], nodes: Sequence[WorkflowNode], rng: random.Random, probability: float) -> None:
    if probability <= 0:
        return
    nodes_by_id = {node.id: node for node in nodes}
    for edge in edges:
        source_role = nodes_by_id[edge.source].role
        target_role = nodes_by_id[edge.target].role
        if source_role in {"start", "split"} or target_role in {"end", "merge"}:
            continue
        if rng.random() < probability:
            edge.source, edge.target = edge.target, edge.source
            edge.path = list(reversed(edge.path))


def assign_edge_types(edges: List[WorkflowEdge], difficulty: str, rng: random.Random) -> None:
    
    
    return


def add_difficult_skip_edges(edges: List[WorkflowEdge], nodes: Sequence[WorkflowNode], orientation: str, rng: random.Random, target_count: int) -> None:
    existing = {(edge.source, edge.target) for edge in edges}
    candidates = []
    for source in nodes:
        for target in nodes:
            if source.id == target.id or (source.id, target.id) in existing:
                continue
            dx = target.center[0] - source.center[0]
            dy = target.center[1] - source.center[1]
            if orientation == "left_to_right":
                if dx < 160:
                    continue
                score = abs(dy) + dx * 0.15
            else:
                if dy < 110:
                    continue
                score = abs(dx) + dy * 0.15
            candidates.append((score, source, target))
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, source, target in candidates:
        if len(edges) >= target_count:
            break
        if (source.id, target.id) in existing:
            continue
        existing.add((source.id, target.id))
        add_edge(edges, source, target, orientation, bends=True)


def target_side(node: WorkflowNode, point: Point) -> str:
    x1, y1, x2, y2 = node.box
    x, y = point
    distances = {
        "left": abs(x - x1),
        "right": abs(x - x2),
        "top": abs(y - y1),
        "bottom": abs(y - y2),
    }
    return min(distances, key=distances.get)


def stagger_incoming_arrowheads(edges: List[WorkflowEdge], nodes: Sequence[WorkflowNode]) -> None:
    nodes_by_id = {node.id: node for node in nodes}
    grouped: Dict[Tuple[str, str], List[WorkflowEdge]] = {}
    for edge in edges:
        target = nodes_by_id[edge.target]
        side = target_side(target, edge.path[-1])
        grouped.setdefault((edge.target, side), []).append(edge)

    for (target_id, side), side_edges in grouped.items():
        if len(side_edges) <= 1:
            continue
        target = nodes_by_id[target_id]
        x1, y1, x2, y2 = target.box
        ordered = sorted(side_edges, key=lambda item: (item.path[0][1], item.path[0][0], item.id))
        step = 12
        center = (len(ordered) - 1) / 2
        for idx, edge in enumerate(ordered):
            offset = int(round((idx - center) * step))
            x, y = edge.path[-1]
            if side in {"left", "right"}:
                edge.path[-1] = (x, max(y1 + 12, min(y2 - 12, y + offset)))
            else:
                edge.path[-1] = (max(x1 + 12, min(x2 - 12, x + offset)), y)


def source_side(node: WorkflowNode, point: Point) -> str:
    return target_side(node, point)


def stagger_outgoing_tails(edges: List[WorkflowEdge], nodes: Sequence[WorkflowNode]) -> None:
    nodes_by_id = {node.id: node for node in nodes}
    grouped: Dict[Tuple[str, str], List[WorkflowEdge]] = {}
    for edge in edges:
        source = nodes_by_id[edge.source]
        side = source_side(source, edge.path[0])
        grouped.setdefault((edge.source, side), []).append(edge)

    for (source_id, side), side_edges in grouped.items():
        if len(side_edges) <= 1:
            continue
        source = nodes_by_id[source_id]
        x1, y1, x2, y2 = source.box
        ordered = sorted(side_edges, key=lambda item: (item.path[-1][1], item.path[-1][0], item.id))
        step = 12
        center = (len(ordered) - 1) / 2
        for idx, edge in enumerate(ordered):
            offset = int(round((idx - center) * step))
            x, y = edge.path[0]
            if side in {"left", "right"}:
                edge.path[0] = (x, max(y1 + 12, min(y2 - 12, y + offset)))
            else:
                edge.path[0] = (max(x1 + 12, min(x2 - 12, x + offset)), y)


def build_case(difficulty: str, index: int, seed: int):
    rng = random.Random(f"{seed}:{difficulty}:{index}")
    topology = TOPOLOGIES[difficulty][(index - 1) % len(TOPOLOGIES[difficulty])]
    orientation = ORIENTATIONS[(index - 1) % len(ORIENTATIONS)] if difficulty == "difficult" else rng.choice(ORIENTATIONS)
    set_canvas_for_case(difficulty, orientation)

    if topology == "linear_pipeline":
        count = rng.randint(5, 6) if difficulty == "simple" else rng.randint(7, 8)
        xs = [95 + i * (810 // (count - 1)) for i in range(count)]
        layout = [("start", (xs[0], 350))]
        layout.extend(("process", (x, 350 + rng.choice([-20, 0, 20]))) for x in xs[1:-1])
        layout.append(("end", (xs[-1], 350)))
        edge_pairs = [(i, i + 1, False) for i in range(count - 1)]
    elif topology == "branching_decision":
        if difficulty == "simple":
            layout = [
                ("start", (60, 350)),
                ("process", (230, 350)),
                ("decision", (420, 350)),
                ("process", (595, 185)),
                ("process", (595, 515)),
                ("process", (730, 350)),
                ("merge", (850, 270)),
                ("end", (960, 350)),
            ]
            edge_pairs = [(0, 1, False), (1, 2, False), (2, 3, True), (2, 4, True), (3, 5, True), (4, 5, True), (5, 6, False), (6, 7, False)]
        else:
            layout = [
                ("start", (60, 350)),
                ("process", (225, 350)),
                ("decision", (390, 350)),
                ("process", (555, 150)),
                ("process", (555, 330)),
                ("process", (555, 540)),
                ("decision", (705, 245)),
                ("process", (705, 455)),
                ("merge", (850, 350)),
                ("end", (960, 350)),
            ]
            edge_pairs = [
                (0, 1, False), (1, 2, False),
                (2, 3, True), (2, 4, True), (2, 5, True),
                (3, 6, False), (4, 6, False), (4, 7, False), (5, 7, False),
                (6, 8, True), (7, 8, True), (8, 9, False),
            ]
    elif topology == "parallel_merge":
        layout = [
            ("start", (60, 350)),
            ("split", (230, 350)),
            ("process", (390, 130)),
            ("process", (390, 300)),
            ("process", (390, 470)),
            ("process", (565, 210)),
            ("process", (565, 390)),
            ("merge", (725, 350)),
            ("process", (855, 430)),
            ("end", (960, 350)),
        ]
        edge_pairs = [
            (0, 1, False),
            (1, 2, False), (1, 3, False), (1, 4, False),
            (2, 5, False), (3, 5, False), (3, 6, False), (4, 6, False),
            (5, 7, False), (6, 7, False), (7, 8, False), (8, 9, False),
        ]
    elif topology == "staged_grid_workflow":
        if difficulty == "simple":
            mid_y = 350 + rng.randint(-8, 8)
            layout = [
                ("start", (70, mid_y - 60)),
                ("split", (210, mid_y)),
                ("process", (390, 190 + rng.randint(-6, 6))),
                ("process", (390, 350 + rng.randint(-6, 6))),
                ("process", (390, 510 + rng.randint(-6, 6))),
                ("process", (610, 260 + rng.randint(-6, 6))),
                ("merge", (610, 455 + rng.randint(-6, 6))),
                ("end", (900, mid_y + 60)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False),
                (2, 5, False), (3, 5, False), (3, 6, False), (4, 6, False),
                (5, 7, False), (6, 7, False),
            ]
        elif difficulty == "medium":
            mid_y = 350 + rng.randint(-8, 8)
            layout = [
                ("start", (60, mid_y - 65)),
                ("split", (180, mid_y)),
                ("process", (330, 130 + rng.randint(-6, 6))),
                ("process", (330, 275 + rng.randint(-6, 6))),
                ("process", (330, 430 + rng.randint(-6, 6))),
                ("process", (330, 565 + rng.randint(-6, 6))),
                ("decision", (510, 175 + rng.randint(-6, 6))),
                ("process", (510, 350 + rng.randint(-6, 6))),
                ("merge", (510, 525 + rng.randint(-6, 6))),
                ("process", (690, 240 + rng.randint(-6, 6))),
                ("process", (690, 465 + rng.randint(-6, 6))),
                ("merge", (845, mid_y)),
                ("end", (960, mid_y + 70)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, False),
                (2, 6, False), (3, 6, False), (3, 7, False),
                (4, 7, False), (4, 8, False), (5, 8, False),
                (6, 9, False), (7, 9, False), (7, 10, False), (8, 10, False),
                (9, 11, False), (10, 11, False), (11, 12, False),
            ]
        else:
            mid_y = 350 + rng.randint(-10, 10)
            end_y = mid_y + rng.choice([-76, 76])
            layout = [
                ("start", (60, mid_y - rng.choice([-70, 70]))),
                ("split", (175, mid_y)),
                ("process", (310, 100 + rng.randint(-6, 6))),
                ("process", (310, 250 + rng.randint(-6, 6))),
                ("process", (310, 410 + rng.randint(-6, 6))),
                ("process", (310, 560 + rng.randint(-6, 6))),
                ("decision", (465, 90 + rng.randint(-5, 5))),
                ("process", (465, 210 + rng.randint(-5, 5))),
                ("process", (465, 330 + rng.randint(-5, 5))),
                ("process", (465, 455 + rng.randint(-5, 5))),
                ("merge", (465, 575 + rng.randint(-5, 5))),
                ("process", (625, 140 + rng.randint(-7, 7))),
                ("decision", (625, 285 + rng.randint(-7, 7))),
                ("process", (625, 430 + rng.randint(-7, 7))),
                ("process", (625, 575 + rng.randint(-7, 7))),
                ("process", (740, 190 + rng.randint(-8, 8))),
                ("process", (740, 350 + rng.randint(-8, 8))),
                ("process", (740, 510 + rng.randint(-8, 8))),
                ("merge", (890, mid_y)),
                ("end", (960, end_y)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, False),
                (2, 6, False), (2, 7, False), (3, 7, False), (3, 8, False),
                (4, 8, False), (4, 9, False), (5, 9, False), (5, 10, False),
                (6, 11, False), (7, 11, False), (7, 12, False), (8, 12, False),
                (8, 13, False), (9, 13, False), (9, 14, False), (10, 14, False),
                (11, 15, False), (12, 15, False), (12, 16, False), (13, 16, False),
                (13, 17, False), (14, 17, False),
                (15, 18, False), (16, 18, False), (17, 18, False), (18, 19, False),
            ]
            if rng.random() < 0.5:
                edge_pairs.append(rng.choice([(6, 12, False), (8, 14, False), (11, 16, False)]))
    elif topology == "ladder_feedback_workflow":
        if difficulty == "simple":
            top = 205 + rng.randint(-6, 6)
            mid = 350 + rng.randint(-6, 6)
            bot = 495 + rng.randint(-6, 6)
            xs = [65, 215, 390, 560, 735, 925]
            layout = [
                ("start", (xs[0], mid - 60)),
                ("split", (xs[1], mid)),
                ("process", (xs[2], top)),
                ("process", (xs[2], mid)),
                ("process", (xs[2], bot)),
                ("decision", (xs[3], top + 35)),
                ("process", (xs[3], bot - 35)),
                ("merge", (xs[4], mid)),
                ("end", (xs[5], mid + 60)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False),
                (2, 5, False), (3, 5, False), (3, 6, False), (4, 6, False),
                (5, 7, False), (6, 7, False), (7, 8, False),
            ]
        elif difficulty == "medium":
            top = 160 + rng.randint(-8, 8)
            mid = 350 + rng.randint(-8, 8)
            bot = 540 + rng.randint(-8, 8)
            xs = [60, 185, 330, 485, 640, 800, 960]
            layout = [
                ("start", (xs[0], mid - 70)),
                ("split", (xs[1], mid)),
                ("process", (xs[2], top)),
                ("process", (xs[2], mid)),
                ("process", (xs[2], bot)),
                ("decision", (xs[3], top + 35)),
                ("process", (xs[3], mid - 25)),
                ("merge", (xs[3], bot - 35)),
                ("process", (xs[4], top + 45)),
                ("decision", (xs[4], mid + 25)),
                ("process", (xs[4], bot - 45)),
                ("merge", (xs[5], mid)),
                ("end", (xs[6], mid + 75)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False),
                (2, 5, False), (2, 6, False), (3, 6, False), (3, 7, False), (4, 7, False),
                (5, 8, False), (6, 8, False), (6, 9, False), (7, 9, False), (7, 10, False),
                (8, 11, False), (9, 11, False), (10, 11, False), (11, 12, False),
            ]
        else:
            top = 150 + rng.randint(-8, 8)
            mid = 350 + rng.randint(-8, 8)
            bot = 550 + rng.randint(-8, 8)
            end_y = mid + rng.choice([-80, 80])
            xs = [60, 175, 305, 430, 555, 675, 790, 960]
            layout = [
                ("start", (xs[0], mid - rng.choice([-72, 72]))),
                ("split", (xs[1], mid)),
                ("process", (xs[2], top)),
                ("process", (xs[2], mid)),
                ("process", (xs[2], bot)),
                ("decision", (xs[3], top + 25)),
                ("process", (xs[3], mid - 25)),
                ("merge", (xs[3], bot - 25)),
                ("process", (xs[4], top - 15)),
                ("decision", (xs[4], mid)),
                ("process", (xs[4], bot + 15)),
                ("process", (xs[5], top + 35)),
                ("process", (xs[5], mid - 35)),
                ("process", (xs[5], bot - 20)),
                ("decision", (xs[6] - 35, top + 60)),
                ("process", (xs[6] - 35, mid)),
                ("merge", (xs[6] - 35, bot - 60)),
                ("process", (xs[6] + 10, mid - 70)),
                ("merge", (xs[6] + 10, mid + 70)),
                ("end", (xs[7], end_y)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False),
                (2, 5, False), (2, 6, False), (3, 6, False), (3, 7, False), (4, 7, False),
                (5, 8, False), (6, 8, False), (6, 9, False), (7, 9, False), (7, 10, False),
                (8, 11, False), (9, 11, False), (9, 12, False), (9, 13, False), (10, 13, False),
                (11, 14, False), (12, 14, False), (12, 15, False), (13, 16, False),
                (14, 17, False), (15, 17, False), (15, 18, False), (16, 18, False),
                (17, 19, False), (18, 19, False),
            ]
            if rng.random() < 0.55:
                edge_pairs.append(rng.choice([(5, 9, False), (8, 12, False), (10, 16, False)]))
    elif topology == "swimlane_workflow":
        if difficulty == "simple":
            lanes = [210 + rng.randint(-8, 8), 350 + rng.randint(-8, 8), 490 + rng.randint(-8, 8)]
            mid_y = 350 + rng.randint(-8, 8)
            layout = [
                ("start", (65, mid_y - 60)),
                ("split", (210, mid_y)),
                ("process", (390, lanes[0])),
                ("process", (390, lanes[1])),
                ("process", (390, lanes[2])),
                ("process", (610, lanes[0] + 35)),
                ("merge", (610, lanes[2] - 35)),
                ("end", (925, mid_y + 60)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False),
                (2, 5, False), (3, 5, False), (3, 6, False), (4, 6, False),
                (5, 7, False), (6, 7, False),
            ]
        elif difficulty == "medium":
            lanes = [130 + rng.randint(-8, 8), 275 + rng.randint(-8, 8), 430 + rng.randint(-8, 8), 570 + rng.randint(-8, 8)]
            mid_y = 350 + rng.randint(-8, 8)
            layout = [
                ("start", (60, mid_y - 70)),
                ("split", (190, mid_y)),
                ("process", (350, lanes[0])),
                ("process", (350, lanes[1])),
                ("process", (350, lanes[2])),
                ("process", (350, lanes[3])),
                ("decision", (520, lanes[0] + 25)),
                ("process", (520, lanes[1] + 25)),
                ("process", (520, lanes[2] + 25)),
                ("merge", (520, lanes[3] - 25)),
                ("process", (700, 260 + rng.randint(-8, 8))),
                ("process", (700, 455 + rng.randint(-8, 8))),
                ("merge", (850, mid_y)),
                ("end", (960, mid_y + 75)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, False),
                (2, 6, False), (3, 6, False), (3, 7, False), (4, 7, False),
                (4, 8, False), (5, 9, False),
                (6, 10, False), (7, 10, False), (8, 11, False), (9, 11, False),
                (10, 12, False), (11, 12, False), (12, 13, False),
            ]
        else:
            lanes = [base + rng.randint(-10, 10) for base in [90, 220, 350, 480, 610]]
            stage_x = [60, 190, 330, 500, 650, 790, 870, 960]
            mid_y = 350 + rng.randint(-12, 12)
            upper_mid = 180 + rng.randint(-12, 12)
            lower_mid = 520 + rng.randint(-12, 12)
            variant = (index - 1) % 4
            if variant == 1:
                lanes[1], lanes[2] = lanes[2] - 12, lanes[1] + 12
            elif variant == 2:
                upper_mid -= 22
                lower_mid += 18
            elif variant == 3:
                upper_mid -= 8
                lower_mid += 8
            start_y = mid_y + rng.choice([-82, 82])
            end_y = mid_y + rng.choice([-82, 82])
            layout = [
                ("start", (stage_x[0], start_y)),
                ("split", (stage_x[1], mid_y)),
                ("process", (stage_x[2], lanes[0])),
                ("process", (stage_x[2], lanes[1])),
                ("process", (stage_x[2], lanes[2])),
                ("process", (stage_x[2], lanes[3])),
                ("process", (stage_x[2], lanes[4])),
                ("decision", (stage_x[3], lanes[0])),
                ("process", (stage_x[3], lanes[1])),
                ("process", (stage_x[3], lanes[2])),
                ("process", (stage_x[3], lanes[3])),
                ("merge", (stage_x[3], lanes[4])),
                ("process", (stage_x[4], upper_mid - 30)),
                ("decision", (stage_x[4], mid_y - 65)),
                ("process", (stage_x[4], mid_y + 75)),
                ("process", (stage_x[4], lower_mid + 35)),
                ("process", (stage_x[5], upper_mid + 58)),
                ("process", (stage_x[5], mid_y + 138)),
                ("merge", (stage_x[6], mid_y)),
                ("end", (stage_x[7], end_y)),
            ]
            edge_pairs = [
                (0, 1, False),
                (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, False), (1, 6, False),
                (2, 7, False), (2, 8, False),
                (3, 8, False), (3, 9, False),
                (4, 9, False), (4, 10, False),
                (5, 10, False), (5, 11, False),
                (6, 11, False),
                (7, 12, False), (8, 12, False), (8, 13, False),
                (9, 13, False), (9, 14, False),
                (10, 14, False), (10, 15, False), (11, 15, False),
                (12, 16, False), (13, 16, False), (13, 17, False),
                (14, 17, False),
                (16, 18, False), (17, 18, False),
                (18, 19, False),
            ]
            extra_pool = [(7, 13), (8, 14), (9, 15), (12, 17)]
            if variant in {1, 3}:
                extra_pool.extend([(3, 13), (10, 17)])
            for extra in rng.sample(extra_pool, rng.randint(0, 1)):
                edge_pairs.append((extra[0], extra[1], False))
    else:
        variant = ["three_two_two", "two_three_two", "three_two_three"][(index - 1) % 3]
        if variant == "three_two_two":
            layout = [
                ("start", (65, 350)),
                ("process", (210, 120)),
                ("process", (210, 280)),
                ("process", (210, 440)),
                ("process", (210, 585)),
                ("decision", (390, 205)),
                ("merge", (390, 395)),
                ("process", (555, 130)),
                ("process", (555, 300)),
                ("process", (555, 490)),
                ("decision", (720, 250)),
                ("merge", (720, 455)),
                ("process", (855, 350)),
                ("end", (940, 350)),
            ]
            edge_pairs = [
                (0, 1, False), (0, 2, False), (0, 3, False), (0, 4, False),
                (1, 5, False), (2, 5, False), (2, 6, False), (3, 6, False), (4, 6, False),
                (5, 7, False), (5, 8, False), (6, 8, False), (6, 9, False),
                (7, 10, False), (8, 10, False), (8, 11, False), (9, 11, False),
                (10, 12, False), (11, 12, False), (12, 13, False),
            ]
        elif variant == "two_three_two":
            layout = [
                ("start", (65, 350)),
                ("process", (210, 210)),
                ("process", (210, 490)),
                ("decision", (365, 130)),
                ("process", (365, 300)),
                ("merge", (365, 500)),
                ("process", (520, 180)),
                ("process", (520, 350)),
                ("process", (520, 535)),
                ("decision", (695, 245)),
                ("merge", (695, 455)),
                ("process", (850, 270)),
                ("process", (850, 460)),
                ("end", (940, 350)),
            ]
            edge_pairs = [
                (0, 1, False), (0, 2, False),
                (1, 3, False), (1, 4, False), (2, 4, False), (2, 5, False),
                (3, 6, False), (4, 6, False), (4, 7, False), (5, 7, False), (5, 8, False),
                (6, 9, False), (7, 9, False), (7, 10, False), (8, 10, False),
                (9, 11, False), (9, 12, False), (10, 12, False),
                (11, 13, False), (12, 13, False),
            ]
        else:
            layout = [
                ("start", (65, 350)),
                ("process", (200, 125)),
                ("process", (200, 285)),
                ("process", (200, 445)),
                ("process", (200, 585)),
                ("decision", (365, 220)),
                ("merge", (365, 430)),
                ("process", (530, 140)),
                ("process", (530, 305)),
                ("process", (530, 500)),
                ("decision", (700, 210)),
                ("process", (700, 390)),
                ("merge", (830, 300)),
                ("end", (940, 350)),
            ]
            edge_pairs = [
                (0, 1, False), (0, 2, False), (0, 3, False), (0, 4, False),
                (1, 5, False), (2, 5, False), (2, 6, False), (3, 6, False), (4, 6, False),
                (5, 7, False), (5, 8, False), (6, 8, False), (6, 9, False),
                (7, 10, False), (8, 10, False), (8, 11, False), (9, 11, False),
                (10, 12, False), (11, 12, False), (12, 13, False),
            ]

    nodes = build_nodes_from_layout(layout, difficulty, orientation, rng)
    edges: List[WorkflowEdge] = []
    for source_idx, target_idx, bends in edge_pairs:
        add_edge(edges, nodes[source_idx], nodes[target_idx], orientation, bends=bends)
    if difficulty == "difficult" and topology == "multi_stage_workflow":
        node_by_role = {}
        for idx, (role, _) in enumerate(layout):
            node_by_role.setdefault(role, []).append(idx)
        extra_pairs = []
        process_idxs = node_by_role.get("process", [])
        decision_idxs = node_by_role.get("decision", [])
        merge_idxs = node_by_role.get("merge", [])
        if len(process_idxs) >= 6:
            extra_pairs.extend([(process_idxs[0], process_idxs[-2]), (process_idxs[1], process_idxs[-1])])
        if decision_idxs and merge_idxs:
            extra_pairs.extend([(decision_idxs[0], merge_idxs[-1])])
        if len(merge_idxs) >= 2:
            extra_pairs.extend([(merge_idxs[0], merge_idxs[-1])])
        existing = {(a, b) for a, b, _ in edge_pairs}
        rng.shuffle(extra_pairs)
        for source_idx, target_idx in extra_pairs[:2]:
            if source_idx != target_idx and (source_idx, target_idx) not in existing:
                add_edge(edges, nodes[source_idx], nodes[target_idx], orientation, bends=True)
        add_difficult_skip_edges(edges, nodes, orientation, rng, target_count=rng.randint(22, 24))
    reverse_random_edges(edges, nodes, rng, REVERSE_EDGE_PROBS[difficulty])
    stagger_outgoing_tails(edges, nodes)
    stagger_incoming_arrowheads(edges, nodes)
    assign_edge_types(edges, difficulty, rng)
    reassign_ids(nodes, edges)
    return nodes, edges, [], topology, orientation


def node_names(nodes: Sequence[WorkflowNode]) -> Dict[str, str]:
    return {node.id: node.label for node in nodes}


def midpoint(path: Sequence[Point]) -> Point:
    if len(path) >= 3:
        return path[len(path) // 2]
    return ((path[0][0] + path[-1][0]) // 2, (path[0][1] + path[-1][1]) // 2)


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


def choose_image_scale(difficulty: str, nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge], seed: int, sample_id: str) -> float:
    return 1.0


def edge_record(edge: WorkflowEdge, names: Dict[str, str], scale: float = 1.0) -> Dict:
    visible = visible_edge_path(edge)
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
        "arrow": scale_point(visible[-1], scale),
        "arrowheads": {"target": [scale_point(point, scale) for point in RENDERER.arrowhead_points(visible)], "source": []},
    }


def node_record(node: WorkflowNode, scale: float = 1.0) -> Dict:
    return {
        "id": node.id,
        "name": node.label,
        "box": scale_box(node.box, scale),
        "center": scale_point(node.center, scale),
        "image_box": None,
        "text_box": scale_box(node.text_box, scale),
        "asset": None,
    }


def final_graph(nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge], scale: float = 1.0) -> Dict:
    names = node_names(nodes)
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
    return {
        "nodes": [
            {"id": node.id, "name": node.label, "box": scale_box(node.box, scale), "relations": relation_map[node.id]}
            for node in reading_order(nodes)
        ]
    }


def think_graph(nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge], negatives, scale: float = 1.0) -> str:
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
            lines.append(
                f'<TRACE target="{rec["target"]}" target_name="{rec["target_name"]}" '
                f'start="{rec["start"]}" mid="{rec["mid"]}" end="{rec["end"]}" arrow="{rec["arrow"]}"/>'
            )
            lines.append(
                f'<EDGE source="{rec["source"]}" source_name="{rec["source_name"]}" '
                f'target="{rec["target"]}" target_name="{rec["target_name"]}" type="{rec["type"]}"/>'
            )
        lines.extend(["</CHECK_NODE>", ""])
    lines.append("</THINK_GRAPH>")
    return "\n".join(lines)


def training_text(nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge], negatives, scale: float = 1.0) -> str:
    return think_graph(nodes, edges, negatives, scale) + "\n\n<FINAL_GRAPH>\n" + json.dumps(final_graph(nodes, edges, scale), indent=2) + "\n</FINAL_GRAPH>"


def wrap_label(text: str, width: int) -> List[str]:
    max_chars = max(8, width // 7)
    lines: List[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=max_chars) or [""])
    return lines[:3]


def node_palette(node: WorkflowNode, style: Dict) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    fill, stroke = PALETTE.get(node.shape, PALETTE["process"])
    fill_mode = style.get("node_fill_mode", "role_palette")
    if fill_mode == "white_blocks":
        fill = tuple(style.get("node_fill", [255, 255, 255]))
    elif fill_mode == "paper_blocks":
        fill = tuple(style.get("node_fill", [252, 251, 247]))
    elif fill_mode == "soft_role_palette":
        fill = tuple(min(255, int(channel * 0.55 + 255 * 0.45)) for channel in fill)
    stroke = tuple(style.get("node_stroke", stroke))
    return fill, stroke


def draw_shape(draw: ImageDraw.ImageDraw, node: WorkflowNode, font, style: Dict) -> None:
    x1, y1, x2, y2 = node.box
    fill, stroke = node_palette(node, style)
    stroke_width = style.get("node_stroke_width", 2)
    radius = style.get("node_radius", 5)
    if node.shape == "terminator":
        draw.rounded_rectangle(node.box, radius=max(18, radius + 16), fill=fill, outline=stroke, width=stroke_width)
    elif node.shape == "decision":
        points = [(node.center[0], y1), (x2, node.center[1]), (node.center[0], y2), (x1, node.center[1])]
        draw.polygon(points, fill=fill, outline=stroke)
        draw.line(points + [points[0]], fill=stroke, width=stroke_width)
    elif node.shape in {"database", "document", "queue", "message", "manual_operation", "merge", "parallel"}:
        draw.rounded_rectangle(node.box, radius=radius, fill=fill, outline=stroke, width=stroke_width)
    elif node.shape == "input_output":
        slant = 16
        points = [(x1 + slant, y1), (x2, y1), (x2 - slant, y2), (x1, y2)]
        draw.polygon(points, fill=fill, outline=stroke)
        draw.line(points + [points[0]], fill=stroke, width=stroke_width)
    elif node.shape == "delay":
        draw.rounded_rectangle(node.box, radius=max(24, radius + 20), fill=fill, outline=stroke, width=stroke_width)
    else:
        if style.get("process_variant") == "square":
            draw.rectangle(node.box, fill=fill, outline=stroke, width=stroke_width)
        else:
            draw.rounded_rectangle(node.box, radius=radius, fill=fill, outline=stroke, width=stroke_width)

    tx1, ty1, tx2, ty2 = node.text_box
    lines = wrap_label(node.label, tx2 - tx1)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max((box[3] - box[1] for box in line_boxes), default=12)
    total_h = line_h * len(lines) + 2 * max(0, len(lines) - 1)
    y = ty1 + max(0, (ty2 - ty1 - total_h) // 2)
    for line, bbox in zip(lines, line_boxes):
        text_w = bbox[2] - bbox[0]
        x = tx1 + max(0, (tx2 - tx1 - text_w) // 2)
        draw.text((x, y), line, fill=tuple(style.get("label_color", [28, 35, 45])), font=font)
        y += line_h + 2


def draw_edge(draw: ImageDraw.ImageDraw, edge: WorkflowEdge, style: Dict) -> None:
    path = visible_edge_path(edge)
    width = style["edge_width"]
    color = tuple(style.get("edge_type_colors", {}).get(edge.edge_type, style["edge_color"]))
    if style["halo"]:
        draw.line(path, fill=(255, 255, 255), width=width + 4, joint="curve")
    draw.line(path, fill=color, width=width, joint="curve")


def draw_edge_arrowhead(draw: ImageDraw.ImageDraw, edge: WorkflowEdge, style: Dict) -> None:
    path = visible_edge_path(edge)
    color = tuple(style.get("edge_type_colors", {}).get(edge.edge_type, style["edge_color"]))
    if style["halo"]:
        outline = RENDERER.arrowhead_points(path, size=style["arrow_size"] + 5)
        draw.polygon(outline, fill=(255, 255, 255))
    draw.polygon(RENDERER.arrowhead_points(path, size=style["arrow_size"]), fill=color)


def render_style(sample_key: str, difficulty: str, seed: int) -> Dict:
    rng = random.Random(f"{seed}:{sample_key}:style")
    if difficulty == "simple":
        edge_palettes = [(38, 48, 64), (25, 25, 25), (16, 28, 140)]
        backgrounds = [
            ((255, 255, 255), "none", "white"),
            ((250, 250, 247), "none", "paper"),
            ((248, 252, 249), "dot", "soft_green_dots"),
        ]
        width = rng.choice([2, 2, 3])
    elif difficulty == "medium":
        edge_palettes = [(38, 48, 64), (25, 25, 25), (16, 28, 140), (0, 105, 130), (92, 58, 150)]
        backgrounds = [
            ((255, 255, 255), "none", "white"),
            ((250, 250, 247), "ruled", "paper_ruled"),
            ((246, 250, 255), "grid", "blue_grid"),
            ((248, 252, 249), "dot", "green_dots"),
            ((252, 248, 242), "grid", "warm_grid"),
        ]
        width = rng.choice([2, 3, 3])
    else:
        edge_palettes = [(38, 48, 64), (25, 25, 25), (16, 28, 140), (0, 105, 130), (92, 58, 150), (125, 30, 55)]
        backgrounds = [
            ((255, 255, 255), "none", "white"),
            ((250, 250, 247), "grid", "paper_grid"),
            ((246, 250, 255), "grid", "blue_grid"),
            ((248, 252, 249), "dot", "green_dots"),
            ((252, 248, 242), "ruled", "warm_ruled"),
        ]
        width = rng.choice([2, 2, 3])
    background, background_pattern, background_name = rng.choice(backgrounds)
    block_styles = [
        ("role_palette", [66, 78, 96], 2, 5, "rounded"),
        ("soft_role_palette", [54, 68, 86], 2, 6, "rounded"),
        ("white_blocks", [54, 68, 86], 2, 2, "square"),
        ("paper_blocks", [78, 76, 68], 2, 4, "rounded"),
    ]
    fill_mode, node_stroke, node_stroke_width, node_radius, process_variant = rng.choice(block_styles)
    return {
        "edge_color": list(rng.choice(edge_palettes)),
        "edge_width": width,
        "arrow_size": max(11, width * 4 + rng.randint(3, 5)),
        "label_color": [24, 28, 35],
        "background": list(background),
        "background_name": background_name,
        "background_pattern": background_pattern,
        "grid": background_pattern in {"grid", "dot", "ruled"},
        "halo": True,
        "edge_clearance": EDGE_CLEARANCE,
        "node_fill_mode": fill_mode,
        "node_fill": [255, 255, 255] if fill_mode == "white_blocks" else [252, 251, 247],
        "node_stroke": node_stroke,
        "node_stroke_width": node_stroke_width,
        "node_radius": node_radius,
        "process_variant": process_variant,
    }


def add_edge_type_style(style: Dict, edges: Sequence[WorkflowEdge], sample_key: str, seed: int) -> None:
    typed = sorted({edge.edge_type for edge in edges if edge.edge_type != "directed"})
    if not typed:
        style["edge_type_colors"] = {}
        style["legend"] = None
        return
    rng = random.Random(f"{seed}:{sample_key}:edge_type_style")
    palette = rng.choice(EDGE_TYPE_PALETTES)
    style["edge_type_colors"] = {edge_type: list(palette[idx % len(palette)]) for idx, edge_type in enumerate(typed)}
    style["legend"] = {
        "position": rng.choice(["top_right", "bottom_left"]),
        "items": [{"type": edge_type, "color": style["edge_type_colors"][edge_type]} for edge_type in typed],
    }


def draw_legend(draw: ImageDraw.ImageDraw, style: Dict) -> None:
    legend = style.get("legend")
    if not legend:
        return
    items = legend["items"]
    font = RENDERER.load_font(14)
    width = 142
    height = 24 + 24 * len(items)
    if legend["position"] == "bottom_left":
        x, y = 34, CANVAS_H - height - 30
    else:
        x, y = CANVAS_W - width - 34, 30
    draw.rounded_rectangle([x, y, x + width, y + height], radius=5, fill=(255, 255, 255), outline=(190, 198, 210), width=1)
    draw.text((x + 10, y + 6), "Legend", fill=(24, 28, 35), font=font)
    yy = y + 28
    for item in items:
        color = tuple(item["color"])
        draw.line([(x + 12, yy + 9), (x + 48, yy + 9)], fill=color, width=3)
        draw.polygon(RENDERER.arrowhead_points([(x + 12, yy + 9), (x + 48, yy + 9)], size=9), fill=color)
        draw.text((x + 56, yy), item["type"], fill=(24, 28, 35), font=font)
        yy += 24


def draw_background_pattern(draw: ImageDraw.ImageDraw, style: Dict) -> None:
    pattern = style.get("background_pattern", "none")
    base = tuple(style.get("background", [255, 255, 255]))
    if base[2] > base[0] + 4:
        color = (225, 233, 244)
    elif base[1] > base[0] + 4:
        color = (226, 238, 226)
    elif base[0] > base[2] + 4:
        color = (238, 231, 218)
    else:
        color = (235, 238, 242)
    if pattern == "grid":
        for x in range(0, CANVAS_W + 1, 50):
            draw.line([(x, 0), (x, CANVAS_H)], fill=color, width=1)
        for y in range(0, CANVAS_H + 1, 50):
            draw.line([(0, y), (CANVAS_W, y)], fill=color, width=1)
    elif pattern == "dot":
        for x in range(25, CANVAS_W, 38):
            for y in range(25, CANVAS_H, 38):
                draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)
    elif pattern == "ruled":
        for y in range(45, CANVAS_H, 42):
            draw.line([(0, y), (CANVAS_W, y)], fill=color, width=1)


def render_image(sample_key: str, difficulty: str, nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge], out_path: Path, seed: int, scale: float = 1.0) -> Dict:
    style = render_style(sample_key, difficulty, seed)
    add_edge_type_style(style, edges, sample_key, seed)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), tuple(style["background"]))
    draw = ImageDraw.Draw(image)
    if style["grid"]:
        draw_background_pattern(draw, style)
    for edge in edges:
        draw_edge(draw, edge, style)
    font = RENDERER.load_font(15 if difficulty == "simple" else (13 if difficulty == "medium" else 12))
    for node in nodes:
        draw_shape(draw, node, font, style)
    for edge in edges:
        draw_edge_arrowhead(draw, edge, style)
    draw_legend(draw, style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = output_canvas(scale)
    if scale != 1.0:
        image = image.resize((canvas["width"], canvas["height"]), Image.Resampling.LANCZOS)
    image.save(out_path)
    style["image_scale"] = scale
    style["canonical_canvas"] = {"width": CANVAS_W, "height": CANVAS_H}
    style["output_canvas"] = canvas
    return style


def annotation_for(sample_id: str, split: str, difficulty: str, index: int, seed: int, image_path: Path, out_dir: Path) -> Dict:
    nodes, edges, negatives, topology, orientation = build_case(difficulty, index, seed)
    image_scale = choose_image_scale(difficulty, nodes, edges, seed, sample_id)
    canvas = output_canvas(image_scale)
    style = render_image(f"{difficulty}_{sample_id}", difficulty, nodes, edges, image_path, seed, image_scale)
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
        "think_graph": think_graph(nodes, edges, negatives, image_scale),
        "training_text": training_text(nodes, edges, negatives, image_scale),
        "render_style": style,
        "generation": {"seed": seed, "index": index, "domain": "workflow", "topology": topology, "image_scale": image_scale},
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
    global REVERSE_EDGE_PROB
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_workflow_v1/outputs/preview_samples")
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--per-difficulty", type=int, default=4)
    parser.add_argument("--difficulty", choices=["all", "simple", "medium", "difficult"], default="all")
    parser.add_argument("--reverse-edge-prob", type=float, default=0.0)
    parser.add_argument("--medium-reverse-edge-prob", type=float, default=0.07)
    parser.add_argument("--difficult-reverse-edge-prob", type=float, default=0.20)
    args = parser.parse_args()
    REVERSE_EDGE_PROB = args.reverse_edge_prob
    REVERSE_EDGE_PROBS["simple"] = 0.0
    REVERSE_EDGE_PROBS["medium"] = args.medium_reverse_edge_prob
    REVERSE_EDGE_PROBS["difficult"] = args.difficult_reverse_edge_prob if args.reverse_edge_prob == 0.0 else args.reverse_edge_prob

    out = Path(args.out)
    image_paths: List[Path] = []
    summary = []
    difficulties = ["simple", "medium", "difficult"] if args.difficulty == "all" else [args.difficulty]
    for difficulty in difficulties:
        for idx in range(1, args.per_difficulty + 1):
            sample_id = f"workflow_{difficulty}_{idx:03d}"
            image_path = out / "images" / difficulty / f"{sample_id}.png"
            annotation_path = out / "annotations" / difficulty / f"{sample_id}.json"
            annotation = annotation_for(sample_id, "preview", difficulty, idx, args.seed, image_path, out)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            image_paths.append(image_path)
            summary.append(
                {
                    "sample_id": sample_id,
                    "difficulty": difficulty,
                    "topology": annotation["generation"]["topology"],
                    "orientation": annotation["orientation"],
                    "nodes": len(annotation["nodes"]),
                    "edges": len(annotation["edges"]),
                    "image": annotation["image"],
                    "annotation": str(annotation_path.relative_to(out)),
                }
            )
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(image_paths, out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
