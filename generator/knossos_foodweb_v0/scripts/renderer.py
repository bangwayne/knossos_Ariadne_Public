#!/usr/bin/env python3








from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


CANVAS_W = 1000
CANVAS_H = 700


@dataclass
class Node:
    id: str
    label: str
    box: BBox
    shape: str
    icon: Optional[str] = None
    asset_path: Optional[str] = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def text_box(self) -> BBox:
        x1, y1, x2, y2 = self.box
        if self.shape == "image_label":
            return (x1, y2 - 24, x2, y2)
        if self.icon:
            return (x1 + 62, y1 + 22, x2 - 16, y2 - 18)
        return (x1 + 16, y1 + 22, x2 - 16, y2 - 18)

    @property
    def icon_box(self) -> Optional[BBox]:
        if self.shape == "image_label":
            x1, y1, x2, y2 = self.box
            return (x1, y1, x2, y2 - 28)
        if not self.icon:
            return None
        x1, y1, _, _ = self.box
        return (x1 + 18, y1 + 25, x1 + 48, y1 + 55)


@dataclass
class Edge:
    id: str
    source: str
    target: str
    path: List[Point]
    edge_type: str = "directed"


@dataclass
class NegativeEdge:
    source: str
    target: str
    reason: str
    evidence_point: Point


@dataclass
class RenderStyle:
    edge_color: Tuple[int, int, int]
    edge_width: int
    arrow_size: int
    label_color: Tuple[int, int, int]
    background: Tuple[int, int, int]
    grid: bool = False
    halo: bool = True


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def rect_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def anchor_for_side(box: BBox, side: str) -> Point:
    x1, y1, x2, y2 = box
    cx, cy = rect_center(box)
    if side == "left":
        return (x1, cy)
    if side == "right":
        return (x2, cy)
    if side == "top":
        return (cx, y1)
    if side == "bottom":
        return (cx, y2)
    raise ValueError(f"unknown side: {side}")


def side_between(source: Node, target: Node) -> Tuple[str, str]:
    sx, sy = source.center
    tx, ty = target.center
    if abs(tx - sx) >= abs(ty - sy):
        return ("right", "left") if tx >= sx else ("left", "right")
    return ("bottom", "top") if ty >= sy else ("top", "bottom")


def side_from_anchor_point(box: BBox, point: Point) -> str:
    x1, y1, x2, y2 = box
    px, py = point
    distances = {
        "left": abs(px - x1),
        "right": abs(px - x2),
        "top": abs(py - y1),
        "bottom": abs(py - y2),
    }
    return min(distances, key=distances.get)


def boundary_point_towards(box: BBox, target: Point) -> Point:
    pass
    x1, y1, x2, y2 = box
    cx, cy = rect_center(box)
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    candidates: List[Tuple[float, Point]] = []
    if dx:
        for x in (x1, x2):
            t = (x - cx) / dx
            y = cy + t * dy
            if t > 0 and y1 <= y <= y2:
                candidates.append((t, (int(round(x)), int(round(y)))))
    if dy:
        for y in (y1, y2):
            t = (y - cy) / dy
            x = cx + t * dx
            if t > 0 and x1 <= x <= x2:
                candidates.append((t, (int(round(x)), int(round(y)))))
    if not candidates:
        return rect_center(box)
    return min(candidates, key=lambda item: item[0])[1]


def arrowhead_points(path: Sequence[Point], size: int = 16) -> List[Point]:
    if len(path) < 2:
        return []
    x2, y2 = path[-1]
    x1, y1 = path[-2]
    angle = math.atan2(y2 - y1, x2 - x1)
    left = angle + math.pi * 0.82
    right = angle - math.pi * 0.82
    return [
        (x2, y2),
        (int(x2 + size * math.cos(left)), int(y2 + size * math.sin(left))),
        (int(x2 + size * math.cos(right)), int(y2 + size * math.sin(right))),
    ]


def reversed_path(path: Sequence[Point]) -> List[Point]:
    return list(reversed(path))


def offset_point(point: Point, toward: Point, distance: int) -> Point:
    px, py = point
    tx, ty = toward
    dx = tx - px
    dy = ty - py
    length = math.hypot(dx, dy)
    if length == 0:
        return point
    return (int(round(px + dx / length * distance)), int(round(py + dy / length * distance)))


def visible_edge_path(edge: Edge, clearance: int = 12) -> List[Point]:
    pass
    path = list(edge.path)
    if len(path) < 2:
        return path
    path[0] = offset_point(path[0], path[1], clearance)
    path[-1] = offset_point(path[-1], path[-2], clearance)
    return path


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    for x in range(0, CANVAS_W + 1, 50):
        draw.line([(x, 0), (x, CANVAS_H)], fill=(235, 238, 242), width=1)
    for y in range(0, CANVAS_H + 1, 50):
        draw.line([(0, y), (CANVAS_W, y)], fill=(235, 238, 242), width=1)


def draw_icon(draw: ImageDraw.ImageDraw, icon: str, box: BBox) -> None:
    x1, y1, x2, y2 = box
    stroke = (64, 78, 96)
    fill = (232, 237, 245)
    if icon == "database":
        draw.ellipse([x1, y1, x2, y1 + 10], outline=stroke, fill=fill, width=2)
        draw.rectangle([x1, y1 + 5, x2, y2 - 5], outline=stroke, fill=fill, width=2)
        draw.ellipse([x1, y2 - 10, x2, y2], outline=stroke, fill=fill, width=2)
    elif icon == "cloud":
        draw.ellipse([x1, y1 + 10, x1 + 18, y2], outline=stroke, fill=fill, width=2)
        draw.ellipse([x1 + 10, y1, x1 + 28, y2 - 4], outline=stroke, fill=fill, width=2)
        draw.ellipse([x1 + 22, y1 + 11, x2, y2], outline=stroke, fill=fill, width=2)
        draw.line([(x1 + 8, y2), (x2 - 7, y2)], fill=stroke, width=2)
    elif icon == "gear":
        cx, cy = rect_center(box)
        draw.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], outline=stroke, fill=fill, width=2)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=stroke, width=2)
        for a in range(0, 360, 45):
            r = math.radians(a)
            draw.line([(cx, cy), (int(cx + 17 * math.cos(r)), int(cy + 17 * math.sin(r)))], fill=stroke, width=2)
    else:
        draw.rectangle(box, outline=stroke, fill=fill, width=2)


def color_from_label(label: str) -> Tuple[int, int, int]:
    value = sum((i + 1) * ord(ch) for i, ch in enumerate(label))
    return (
        80 + value % 120,
        90 + (value // 7) % 120,
        80 + (value // 13) % 120,
    )


def draw_image_patch(draw: ImageDraw.ImageDraw, label: str, box: BBox) -> None:
    pass
    x1, y1, x2, y2 = box
    base = color_from_label(label)
    draw.rectangle(box, fill=(235, 240, 230), outline=None)
    for i in range(y1, y2, 4):
        t = (i - y1) / max(1, y2 - y1)
        fill = (
            int(base[0] * (0.65 + 0.35 * t)),
            int(base[1] * (0.75 + 0.25 * (1 - t))),
            int(base[2] * (0.70 + 0.30 * t)),
        )
        draw.rectangle([x1, i, x2, min(i + 3, y2)], fill=fill)
    rng = random.Random(label)
    for _ in range(55):
        px = rng.randint(x1, x2 - 1)
        py = rng.randint(y1, y2 - 1)
        shade = rng.randint(-35, 45)
        fill = tuple(max(0, min(255, c + shade)) for c in base)
        draw.point((px, py), fill=fill)

    cx, cy = rect_center(box)
    if any(word in label.upper() for word in ["BEAR", "WOLF", "BOBCAT", "MOOSE", "CARIBOU", "SQUIRREL"]):
        draw.ellipse([cx - 22, cy - 12, cx + 22, cy + 12], fill=(96, 72, 54), outline=(40, 35, 30), width=2)
        draw.ellipse([cx + 12, cy - 17, cx + 30, cy + 3], fill=(96, 72, 54), outline=(40, 35, 30), width=2)
        draw.line([(cx - 18, cy + 12), (cx - 26, cy + 24)], fill=(40, 35, 30), width=3)
        draw.line([(cx + 8, cy + 12), (cx + 1, cy + 24)], fill=(40, 35, 30), width=3)
    elif any(word in label.upper() for word in ["OWL", "HAWK"]):
        draw.ellipse([cx - 22, cy - 20, cx + 22, cy + 20], fill=(82, 76, 70), outline=(35, 35, 35), width=2)
        draw.polygon([(cx - 34, cy - 2), (cx - 5, cy - 12), (cx - 8, cy + 8)], fill=(64, 62, 58))
        draw.polygon([(cx + 34, cy - 2), (cx + 5, cy - 12), (cx + 8, cy + 8)], fill=(64, 62, 58))
        draw.ellipse([cx - 10, cy - 5, cx - 4, cy + 1], fill=(245, 238, 190))
        draw.ellipse([cx + 4, cy - 5, cx + 10, cy + 1], fill=(245, 238, 190))
    else:
        draw.line([(x1 + 8, y2 - 12), (x2 - 10, y1 + 8)], fill=(46, 116, 54), width=4)
        draw.line([(x1 + 16, y2 - 8), (x1 + 34, y1 + 16)], fill=(88, 150, 62), width=3)
        draw.ellipse([cx - 17, cy - 10, cx + 17, cy + 10], fill=(86, 135, 58), outline=(32, 95, 36), width=2)


def paste_asset_patch(base: Image.Image, asset_path: str, box: BBox) -> None:
    asset = Image.open(asset_path).convert("RGB")
    x1, y1, x2, y2 = box
    patch = ImageOps.fit(asset, (x2 - x1, y2 - y1), method=Image.Resampling.LANCZOS)
    base.paste(patch, (x1, y1))


def draw_node(base: Image.Image, draw: ImageDraw.ImageDraw, node: Node, font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = node.box
    fill = (255, 255, 255)
    outline = (48, 64, 82)
    if node.shape == "image_label":
        if node.icon_box:
            if node.asset_path and Path(node.asset_path).exists():
                paste_asset_patch(base, node.asset_path, node.icon_box)
            else:
                draw_image_patch(draw, node.label, node.icon_box)
    elif node.shape == "rounded":
        draw.rounded_rectangle(node.box, radius=14, fill=fill, outline=outline, width=3)
    elif node.shape == "circle":
        draw.ellipse(node.box, fill=fill, outline=outline, width=3)
    elif node.shape == "diamond":
        cx, cy = node.center
        pts = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        draw.polygon(pts, fill=fill, outline=outline)
        draw.line(pts + [pts[0]], fill=outline, width=3)
    else:
        draw.rectangle(node.box, fill=fill, outline=outline, width=3)

    if node.shape != "image_label" and node.icon and node.icon_box:
        draw_icon(draw, node.icon, node.icon_box)

    tx1, ty1, tx2, ty2 = node.text_box
    bbox = draw.textbbox((0, 0), node.label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((tx1 + max(0, (tx2 - tx1 - tw) // 2), ty1 + max(0, (ty2 - ty1 - th) // 2)), node.label, fill=(28, 35, 45), font=font)


def draw_arrowhead(draw: ImageDraw.ImageDraw, path: Sequence[Point], color: Tuple[int, int, int], size: int, halo: bool) -> None:
    pts = arrowhead_points(path, size=size)
    if not pts:
        return
    if halo:
        outline = arrowhead_points(path, size=size + 6)
        draw.polygon(outline, fill=(255, 255, 255))
    draw.polygon(pts, fill=color)


def draw_edge(draw: ImageDraw.ImageDraw, edge: Edge, style: RenderStyle) -> None:
    color = style.edge_color
    width = style.edge_width
    clearance = max(12, style.arrow_size + 8)
    path = visible_edge_path(edge, clearance=clearance)
    if style.halo:
        draw.line(path, fill=(255, 255, 255), width=width + 3, joint="curve")
    draw.line(path, fill=color, width=width, joint="curve")


def draw_edge_arrowheads(draw: ImageDraw.ImageDraw, edge: Edge, style: RenderStyle) -> None:
    color = style.edge_color
    clearance = max(12, style.arrow_size // 2)
    path = visible_edge_path(edge, clearance=clearance)
    if edge.edge_type in {"directed", "bidirectional"}:
        draw_arrowhead(draw, path, color, style.arrow_size, style.halo)
    if edge.edge_type == "bidirectional":
        draw_arrowhead(draw, reversed_path(path), color, style.arrow_size, style.halo)


def draw_near_miss(draw: ImageDraw.ImageDraw, path: Sequence[Point]) -> None:
    draw.line(path, fill=(151, 88, 42), width=3)


def style_for_case(name: str, seed: int, ai2d_like: bool) -> RenderStyle:
    rng = random.Random(f"{seed}:{name}")
    if not ai2d_like:
        return RenderStyle(
            edge_color=(38, 48, 64),
            edge_width=4,
            arrow_size=16,
            label_color=(28, 35, 45),
            background=(248, 250, 252),
            grid=True,
            halo=False,
        )
    if name.startswith("simple"):
        palettes = [
            (16, 28, 140),
            (25, 25, 25),
            (40, 105, 52),
        ]
        backgrounds = [
            ((255, 255, 255), False),
            ((250, 250, 247), False),
            ((248, 252, 249), False),
        ]
        color = rng.choice(palettes)
        background, grid = rng.choice(backgrounds)
        width = rng.choice([2, 3])
    elif name.startswith("medium"):
        palettes = [
            (16, 28, 140),
            (25, 25, 25),
            (40, 105, 52),
            (92, 58, 150),
            (0, 105, 130),
        ]
        backgrounds = [
            ((255, 255, 255), False),
            ((250, 250, 247), False),
            ((246, 250, 255), False),
            ((248, 252, 249), True),
        ]
        color = rng.choice(palettes)
        background, grid = rng.choice(backgrounds)
        width = rng.choice([2, 3, 3, 4])
    elif name.startswith("difficult"):
        palettes = [
            (16, 28, 140),
            (25, 25, 25),
            (40, 105, 52),
            (92, 58, 150),
            (136, 66, 35),
            (0, 105, 130),
            (125, 30, 55),
        ]
        backgrounds = [
            ((255, 255, 255), False),
            ((250, 250, 247), True),
            ((246, 250, 255), True),
            ((248, 252, 249), True),
            ((252, 248, 242), False),
        ]
        color = rng.choice(palettes)
        background, grid = rng.choice(backgrounds)
        width = rng.choice([2, 2, 3])
    else:
        palettes = [
            ((16, 28, 140), "classic blue"),
            ((25, 25, 25), "textbook black"),
            ((40, 105, 52), "ecology green"),
            ((92, 58, 150), "muted purple"),
            ((136, 66, 35), "brown ink"),
            ((0, 105, 130), "teal"),
        ]
        color, _ = rng.choice(palettes)
        background = (255, 255, 255)
        grid = False
        width = rng.choice([2, 2, 3, 3, 4])
    return RenderStyle(
        edge_color=color,
        edge_width=width,
        arrow_size=max(11, width * 4 + rng.randint(2, 4)) if name.startswith("difficult") else max(15, width * 5 + rng.randint(4, 7)),
        label_color=(24, 24, 24),
        background=background,
        grid=grid,
        halo=True,
    )


def node_to_json(node: Node) -> Dict[str, object]:
    return {
        "id": node.id,
        "label": node.label,
        "box": list(node.box),
        "center": list(node.center),
        "shape": node.shape,
        "icon": node.icon,
        "asset": Path(node.asset_path).name if node.asset_path else None,
        "text_box": list(node.text_box),
        "icon_box": list(node.icon_box) if node.icon_box else None,
    }


def edge_to_json(edge: Edge, nodes: Dict[str, Node]) -> Dict[str, object]:
    source = nodes[edge.source]
    target = nodes[edge.target]
    source_side = side_from_anchor_point(source.box, edge.path[0])
    target_side = side_from_anchor_point(target.box, edge.path[-1])
    visible_path = visible_edge_path(edge, clearance=12)
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "type": edge.edge_type,
        "path": [list(p) for p in edge.path],
        "visible_path": [list(p) for p in visible_path],
        "source_anchor": {"side": source_side, "point": list(edge.path[0])},
        "target_anchor": {"side": target_side, "point": list(edge.path[-1])},
        "arrowheads": {
            "target": arrowhead_points(visible_path) if edge.edge_type in {"directed", "bidirectional"} else [],
            "source": arrowhead_points(reversed_path(visible_path)) if edge.edge_type == "bidirectional" else [],
        },
    }


def final_graph_json(nodes: Iterable[Node], edges: Iterable[Edge]) -> Dict[str, object]:
    nodes = list(nodes)
    edge_map: Dict[str, List[Dict[str, str]]] = {node.id: [] for node in nodes}
    for edge in edges:
        edge_map[edge.source].append({"target": edge.target, "type": edge.edge_type})
        if edge.edge_type in {"bidirectional", "undirected"}:
            edge_map[edge.target].append({"target": edge.source, "type": edge.edge_type})
    return {
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "box": list(node.box),
                "relations": edge_map[node.id],
            }
            for node in nodes
        ]
    }


def think_graph(nodes: List[Node], edges: List[Edge], negatives: List[NegativeEdge]) -> str:
    node_by_id = {node.id: node for node in nodes}
    lines = ["<THINK_GRAPH>"]
    for node in nodes:
        lines.append(
            f'<NODE id="{node.id}" label="{node.label}" '
            f'box="{list(node.box)}" center="{list(node.center)}"/>'
        )
    lines.append("")
    for edge in edges:
        source = node_by_id[edge.source]
        target = node_by_id[edge.target]
        source_side = side_from_anchor_point(source.box, edge.path[0])
        target_side = side_from_anchor_point(target.box, edge.path[-1])
        lines.append(f'<ANCHOR node="{source.id}" side="{source_side}" point="{list(edge.path[0])}"/>')
        lines.append(f'<ANCHOR node="{target.id}" side="{target_side}" point="{list(edge.path[-1])}"/>')
        lines.append(f'<CONNECTOR_TRACE source="{edge.source}" target="{edge.target}">')
        lines.append(json.dumps([list(p) for p in edge.path]))
        lines.append("</CONNECTOR_TRACE>")
        lines.append(f'<EDGE source="{edge.source}" target="{edge.target}" type="{edge.edge_type}"/>')
        lines.append("")
    for neg in negatives:
        lines.append(
            f'<NO_EDGE source="{neg.source}" target="{neg.target}" '
            f'reason="{neg.reason}" evidence_point="{list(neg.evidence_point)}"/>'
        )
    lines.append("</THINK_GRAPH>")
    return "\n".join(lines)


def load_asset_paths(asset_dir: Optional[str]) -> List[Path]:
    if not asset_dir:
        return []
    root = Path(asset_dir)
    if not root.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in exts)


def asset_label_map(asset_paths: Sequence[Path]) -> Dict[str, List[str]]:
    if not asset_paths:
        return {}
    root = asset_paths[0].parent
    metadata_path = root / "metadata.json"
    mapping: Dict[str, List[str]] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for filename, info in metadata.items():
            label = info.get("label")
            path = root / filename
            if label and path.exists():
                mapping.setdefault(slugify(label), []).append(str(path))
    for path in asset_paths:
        stem = re.sub(r"_\\d+$", "", path.stem)
        mapping.setdefault(stem, []).append(str(path))
    for paths in mapping.values():
        paths.sort()
    return mapping


def assign_assets(nodes: List[Node], asset_paths: Sequence[Path]) -> None:
    mapping = asset_label_map(asset_paths)
    if not mapping:
        return
    aliases = {
        "hawk_owl": ["hawk", "owl"],
        "black_spruce": ["spruce_tree", "evergreen_tree"],
        "evergreen": ["evergreen_tree", "spruce_tree"],
        "black_bear": ["black_bear", "bear"],
        "bear": ["black_bear"],
        "gray_wolf": ["gray_wolf", "wolf"],
        "wolf": ["gray_wolf"],
    }
    for node in nodes:
        if node.shape == "image_label":
            key = slugify(node.label)
            candidates = [key] + aliases.get(key, [])
            for candidate in candidates:
                if candidate in mapping:
                    node.asset_path = random.choice(mapping[candidate])
                    break


def make_random_ai2d_case(case_idx: int, asset_paths: Sequence[Path]) -> Tuple[str, List[Node], List[Edge], List[NegativeEdge], List[List[Point]]]:
    rng = random.Random(1000 + case_idx + random.randint(0, 100000))
    
    
    tier_pools = [
        ["GRASS", "BERRIES", "ALGAE", "LEAVES", "ACORN", "FLOWERING PLANT", "PLANKTON", "SEAWEED", "CORAL", "MUSHROOM"],
        ["RABBIT", "MOUSE", "SQUIRREL", "DEER", "CARIBOU", "MOOSE", "CATERPILLAR", "GRASSHOPPER", "DRAGONFLY", "BUTTERFLY", "FROG", "FISH", "SHRIMP", "CLAM", "SNAIL"],
        ["BOBCAT", "FOX", "HAWK", "OWL", "GRAY WOLF", "BLACK BEAR", "SNAKE", "EAGLE", "SHARK", "SEAL", "KILLER WHALE"],
    ]
    tier_counts = [
        rng.randint(2, 3),
        rng.randint(3, 4),
        rng.randint(2, 3),
    ]
    tiers = [rng.sample(pool, count) for pool, count in zip(tier_pools, tier_counts)]

    nodes: List[Node] = []
    idx = 1
    for tier_idx, tier in enumerate(tiers):
        y_base = [520, 330, 115][tier_idx]
        available_w = 850
        step = available_w // max(1, len(tier))
        for local_idx, label in enumerate(tier):
            w = rng.randint(96, 112)
            h = rng.randint(108, 120)
            x_center = 70 + step // 2 + local_idx * step + rng.randint(-18, 18)
            x = min(900 - w, max(45, x_center - w // 2))
            y = y_base + rng.randint(-12, 12)
            box = (x, y, x + w, y + h)
            nodes.append(Node(f"n{idx}", label, box, "image_label"))
            idx += 1

    assign_assets(nodes, asset_paths)
    by_tier = {
        0: nodes[: len(tiers[0])],
        1: nodes[len(tiers[0]) : len(tiers[0]) + len(tiers[1])],
        2: nodes[len(tiers[0]) + len(tiers[1]) :],
    }
    edges: List[Edge] = []

    def add_edge(src: Node, dst: Node, edge_type: str = "directed") -> None:
        if src.id == dst.id:
            return
        if any(e.source == src.id and e.target == dst.id for e in edges):
            return
        start = boundary_point_towards(src.box, dst.center)
        end = boundary_point_towards(dst.box, src.center)
        if rng.random() < 0.55:
            mid = ((start[0] + end[0]) // 2 + rng.randint(-18, 18), (start[1] + end[1]) // 2 + rng.randint(-10, 10))
            path = [start, mid, end]
        else:
            path = [start, end]
        edges.append(Edge(f"e{len(edges) + 1}", src.id, dst.id, path, edge_type))

    for producer in by_tier[0]:
        targets = rng.sample(by_tier[1], 1)
        for target in targets:
            add_edge(producer, target)
    for consumer in by_tier[1]:
        targets = rng.sample(by_tier[2], 1)
        for target in targets:
            add_edge(consumer, target)
    for _ in range(rng.randint(1, 2)):
        add_edge(rng.choice(by_tier[0]), rng.choice(by_tier[1]))
        add_edge(rng.choice(by_tier[1]), rng.choice(by_tier[2]))
    if case_idx == 1 and len(by_tier[2]) >= 2:
        a, b = rng.sample(by_tier[2], 2)
        add_edge(a, b, "bidirectional")

    negatives: List[NegativeEdge] = []
    if case_idx == 3 and len(nodes) >= 4:
        for src, dst in rng.sample([(a, b) for a in nodes for b in nodes if a.id != b.id], 1):
            px = (src.center[0] + dst.center[0]) // 2 + rng.randint(-20, 20)
            py = (src.center[1] + dst.center[1]) // 2 + rng.randint(-20, 20)
            negatives.append(NegativeEdge(src.id, dst.id, rng.choice(["near_miss", "crossing_not_connected"]), (px, py)))

    return (f"ai2d_rule_generated_{case_idx:02d}", nodes, edges, negatives, [])


def sample_cases(asset_paths: Sequence[Path]) -> List[Tuple[str, List[Node], List[Edge], List[NegativeEdge], List[List[Point]]]]:
    cases = []

    nodes1 = [
        Node("n1", "Input", (90, 285, 230, 365), "rounded", "cloud"),
        Node("n2", "Process", (410, 285, 570, 365), "rounded", "gear"),
        Node("n3", "Output", (740, 285, 880, 365), "rounded", None),
    ]
    edges1 = [
        Edge("e1", "n1", "n2", [(230, 325), (320, 325), (410, 325)]),
        Edge("e2", "n2", "n3", [(570, 325), (655, 325), (740, 325)]),
    ]
    cases.append(("straight_pipeline", nodes1, edges1, [], []))

    nodes2 = [
        Node("n1", "Client", (90, 80, 250, 160), "rounded", None),
        Node("n2", "API", (420, 80, 560, 160), "rounded", "gear"),
        Node("n3", "DB", (420, 430, 560, 510), "rounded", "database"),
        Node("n4", "Cache", (760, 250, 900, 330), "rounded", "database"),
    ]
    edges2 = [
        Edge("e1", "n1", "n2", [(250, 120), (335, 120), (420, 120)]),
        Edge("e2", "n2", "n3", [(490, 160), (490, 265), (490, 430)]),
        Edge("e3", "n1", "n4", [(250, 120), (310, 120), (310, 290), (760, 290)]),
    ]
    negatives2 = [NegativeEdge("n2", "n4", "crossing_not_connected", (490, 290))]
    cases.append(("crossing_not_connected", nodes2, edges2, negatives2, []))

    nodes3 = [
        Node("n1", "Parser", (90, 110, 250, 190), "rounded", "gear"),
        Node("n2", "Router", (420, 110, 580, 190), "rounded", None),
        Node("n3", "Store", (420, 390, 580, 470), "rounded", "database"),
    ]
    edges3 = [
        Edge("e1", "n1", "n2", [(250, 150), (335, 150), (420, 150)]),
    ]
    near_miss_path = [(250, 150), (335, 150), (335, 382), (418, 382)]
    negatives3 = [NegativeEdge("n1", "n3", "near_miss", (418, 382))]
    cases.append(("near_miss_gap", nodes3, edges3, negatives3, [near_miss_path]))

    nodes4 = [
        Node("n1", "Start", (90, 290, 230, 370), "circle", None),
        Node("n2", "Decision", (410, 260, 590, 400), "diamond", None),
        Node("n3", "Yes", (760, 120, 900, 200), "rounded", None),
        Node("n4", "No", (760, 460, 900, 540), "rounded", None),
    ]
    edges4 = [
        Edge("e1", "n1", "n2", [(230, 330), (320, 330), (410, 330)]),
        Edge("e2", "n2", "n3", [(590, 330), (670, 330), (670, 160), (760, 160)]),
        Edge("e3", "n2", "n4", [(590, 330), (670, 330), (670, 500), (760, 500)]),
    ]
    cases.append(("branching_decision", nodes4, edges4, [], []))

    nodes5 = [
        Node("n1", "GRASS", (150, 505, 260, 610), "image_label"),
        Node("n2", "SQUIRREL", (145, 315, 250, 425), "image_label"),
        Node("n3", "HAWK OWL", (18, 74, 105, 184), "image_label"),
        Node("n4", "BOBCAT", (302, 116, 412, 226), "image_label"),
        Node("n5", "CARIBOU", (505, 286, 620, 396), "image_label"),
        Node("n6", "MOOSE", (704, 340, 824, 450), "image_label"),
        Node("n7", "GRAY WOLF", (872, 205, 980, 315), "image_label"),
        Node("n8", "BLACK BEAR", (610, 45, 730, 160), "image_label"),
        Node("n9", "EVERGREEN", (515, 535, 625, 645), "image_label"),
        Node("n10", "BLACK SPRUCE", (872, 510, 978, 632), "image_label"),
    ]
    assign_assets(nodes5, asset_paths)
    edges5 = [
        Edge("e1", "n1", "n2", [(205, 505), (198, 468), (198, 425)], "directed"),
        Edge("e2", "n1", "n5", [(260, 555), (372, 472), (505, 346)], "directed"),
        Edge("e3", "n2", "n3", [(170, 315), (105, 184)], "directed"),
        Edge("e4", "n2", "n4", [(250, 340), (302, 180)], "directed"),
        Edge("e5", "n2", "n8", [(250, 350), (420, 250), (610, 100)], "directed"),
        Edge("e6", "n2", "n7", [(250, 360), (560, 282), (872, 250)], "directed"),
        Edge("e7", "n4", "n8", [(412, 170), (505, 132), (610, 92)], "directed"),
        Edge("e8", "n5", "n8", [(560, 286), (620, 205), (670, 160)], "directed"),
        Edge("e9", "n5", "n7", [(620, 336), (735, 276), (872, 250)], "directed"),
        Edge("e10", "n9", "n5", [(570, 535), (560, 462), (560, 396)], "directed"),
        Edge("e11", "n9", "n6", [(625, 575), (675, 496), (704, 395)], "directed"),
        Edge("e12", "n10", "n6", [(872, 560), (825, 490), (795, 450)], "directed"),
        Edge("e13", "n6", "n8", [(760, 340), (710, 248), (690, 160)], "directed"),
        Edge("e14", "n6", "n7", [(824, 390), (848, 322), (872, 270)], "bidirectional"),
        Edge("e15", "n3", "n8", [(105, 130), (350, 100), (610, 85)], "undirected"),
    ]
    negatives5 = [
        NegativeEdge("n5", "n6", "crossing_not_connected", (620, 336)),
        NegativeEdge("n9", "n7", "near_miss", (675, 496)),
    ]
    cases.append(("ai2d_food_web_mixed", nodes5, edges5, negatives5, []))

    for idx in range(1, 4):
        cases.append(make_random_ai2d_case(idx, asset_paths))

    return cases


def render_case(
    name: str,
    nodes: List[Node],
    edges: List[Edge],
    negatives: List[NegativeEdge],
    extra_paths: List[List[Point]],
    out_dir: Path,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ai2d_like = any(node.shape == "image_label" for node in nodes)
    style = style_for_case(name, seed, ai2d_like)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), style.background)
    draw = ImageDraw.Draw(image)
    if style.grid:
        draw_grid(draw)
    if ai2d_like and name.startswith("difficult"):
        font = load_font(12)
    elif ai2d_like and name.startswith("medium"):
        font = load_font(14)
    else:
        font = load_font(16 if ai2d_like else 22)

    for path in extra_paths:
        draw_near_miss(draw, path)
    for edge in edges:
        draw_edge(draw, edge, style)
    for node in nodes:
        draw_node(image, draw, node, font)
    for edge in edges:
        draw_edge_arrowheads(draw, edge, style)

    image_path = out_dir / f"{name}.png"
    image.save(image_path)

    node_by_id = {node.id: node for node in nodes}
    gt = {
        "image": image_path.name,
        "nodes": [node_to_json(node) for node in nodes],
        "edges": [edge_to_json(edge, node_by_id) for edge in edges],
        "negative_edges": [
            {
                "source": neg.source,
                "target": neg.target,
                "reason": neg.reason,
                "evidence_point": list(neg.evidence_point),
            }
            for neg in negatives
        ],
        "final_graph": final_graph_json(nodes, edges),
        "think_graph": think_graph(nodes, edges, negatives),
        "render_style": {
            "edge_color": list(style.edge_color),
            "edge_width": style.edge_width,
            "arrow_size": style.arrow_size,
            "halo": style.halo,
        },
    }
    (out_dir / f"{name}.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")


def make_contact_sheet(image_paths: List[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 224))
        thumbs.append((path.stem, img.copy()))

    font = load_font(18)
    cols = 2
    cell_w = 500
    cell_h = 285
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    positions = [((idx % cols) * cell_w + 10, (idx // cols) * cell_h + 35) for idx in range(len(thumbs))]
    for (title, img), (x, y) in zip(thumbs, positions):
        draw.text((x, y - 25), title, fill=(28, 35, 45), font=font)
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width, y + img.height], outline=(210, 215, 222), width=1)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/knossos_preview", help="Output directory")
    parser.add_argument("--asset-dir", default="assets/node_images", help="Optional directory of node image assets")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    asset_paths = load_asset_paths(args.asset_dir)
    image_paths = []
    for name, nodes, edges, negatives, extra_paths in sample_cases(asset_paths):
        render_case(name, nodes, edges, negatives, extra_paths, out_dir, args.seed)
        image_paths.append(out_dir / f"{name}.png")

    make_contact_sheet(image_paths, out_dir / "contact_sheet.png")
    print(f"Generated {len(image_paths)} diagrams in {out_dir.resolve()}")
    print(f"Loaded {len(asset_paths)} node image assets from {Path(args.asset_dir).resolve()}")
    print(f"Contact sheet: {(out_dir / 'contact_sheet.png').resolve()}")


if __name__ == "__main__":
    main()
