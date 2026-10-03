#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_renderer():
    path = Path(__file__).with_name("renderer.py")
    spec = importlib.util.spec_from_file_location("knossos_renderer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def build_asset_index(asset_dir: Path) -> Dict[str, List[str]]:
    metadata = json.loads((asset_dir / "metadata.json").read_text())
    index: Dict[str, List[str]] = {}
    for filename, info in metadata.items():
        index.setdefault(info["slug"], []).append(str(asset_dir / filename))
    for paths in index.values():
        paths.sort()
    return index


TIER_POOLS = [
    ["GRASS", "BERRIES", "ALGAE", "LEAVES", "ACORN", "FLOWERING PLANT", "PLANKTON", "SEAWEED"],
    ["RABBIT", "MOUSE", "SQUIRREL", "DEER", "CARIBOU", "MOOSE", "CATERPILLAR", "GRASSHOPPER", "DRAGONFLY", "BUTTERFLY", "FROG", "FISH"],
    ["BOBCAT", "FOX", "HAWK", "OWL", "GRAY WOLF", "BLACK BEAR", "SNAKE", "EAGLE", "SHARK", "SEAL", "KILLER WHALE"],
]


def label_slug(label: str) -> str:
    aliases = {
        "BEAR": "black_bear",
        "WOLF": "gray_wolf",
    }
    return aliases.get(label, slug(label))


def display_label(label: str, rng: random.Random) -> str:
    modes = [
        "upper",
        "lower",
        "title",
        "source",
    ]
    mode = rng.choice(modes)
    text = label.replace("_", " ")
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "title":
        return text.title()
    return text


def route_edge(renderer, edge_id: int, nodes, src, dst, edge_type: str, rng: random.Random, bends: bool):
    start = renderer.boundary_point_towards(src.box, dst.center)
    end = renderer.boundary_point_towards(dst.box, src.center)
    if bends:
        mid = (
            (start[0] + end[0]) // 2 + rng.randint(-32, 32),
            (start[1] + end[1]) // 2 + rng.randint(-16, 16),
        )
        path = [start, mid, end]
    else:
        path = [start, end]
    return renderer.Edge(f"e{edge_id}", src.id, dst.id, path, edge_type)


def make_case(renderer, asset_index, difficulty: str, sample_idx: int, seed: int):
    rng = random.Random(f"{seed}:{difficulty}:{sample_idx}")
    orientation = rng.choice(["bottom_to_top", "top_to_bottom", "left_to_right", "right_to_left"])
    if difficulty == "simple":
        tier_counts = [2, rng.randint(3, 4), rng.randint(1, 2)]
        extra_edges = rng.randint(2, 3)
        bidirectional = 0
        negatives = 0
        bends = rng.random() < 0.35
    elif difficulty == "medium":
        tier_counts = [rng.randint(2, 3), rng.randint(4, 6), rng.randint(3, 4)]
        extra_edges = rng.randint(2, 4)
        bidirectional = 1
        negatives = 1
        bends = True
    else:
        tier_counts = [4, rng.randint(7, 8), 5]
        extra_edges = rng.randint(4, 7)
        bidirectional = 2
        negatives = rng.randint(2, 3)
        bends = True

    tiers = [rng.sample(pool, count) for pool, count in zip(TIER_POOLS, tier_counts)]
    nodes = []
    idx = 1
    def node_size() -> Tuple[int, int]:
        if difficulty == "difficult":
            return rng.randint(66, 78), rng.randint(74, 86)
        if difficulty == "medium":
            return rng.randint(90, 104), rng.randint(102, 114)
        return rng.randint(96, 112), rng.randint(108, 120)

    def box_for(tier_idx: int, local_idx: int, count: int, width: int, height: int) -> Tuple[int, int, int, int]:
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

    for tier_idx, labels in enumerate(tiers):
        for local_idx, label in enumerate(labels):
            width, height = node_size()
            if orientation in {"left_to_right", "right_to_left"}:
                max_height = max(46, (600 // len(labels)) - 30)
                height = min(height, max_height)
            node = renderer.Node(f"n{idx}", display_label(label, rng), box_for(tier_idx, local_idx, len(labels), width, height), "image_label")
            slug_key = label_slug(label)
            if slug_key in asset_index:
                node.asset_path = rng.choice(asset_index[slug_key])
            nodes.append(node)
            idx += 1

    by_tier = {
        0: nodes[: len(tiers[0])],
        1: nodes[len(tiers[0]) : len(tiers[0]) + len(tiers[1])],
        2: nodes[len(tiers[0]) + len(tiers[1]) :],
    }

    edges = []
    seen = set()

    def add(src, dst, edge_type="directed"):
        if src.id == dst.id or (src.id, dst.id) in seen:
            return
        seen.add((src.id, dst.id))
        edges.append(route_edge(renderer, len(edges) + 1, nodes, src, dst, edge_type, rng, bends))

    for producer in by_tier[0]:
        add(producer, rng.choice(by_tier[1]))
    for consumer in by_tier[1]:
        add(consumer, rng.choice(by_tier[2]))
    for _ in range(extra_edges):
        if rng.random() < 0.45:
            add(rng.choice(by_tier[0]), rng.choice(by_tier[1]))
        else:
            add(rng.choice(by_tier[1]), rng.choice(by_tier[2]))
    def center_distance(a, b) -> float:
        return ((a.center[0] - b.center[0]) ** 2 + (a.center[1] - b.center[1]) ** 2) ** 0.5

    def bidirectional_candidates():
        top_nodes = by_tier[2]
        min_distance = 220 if difficulty == "difficult" else 180
        candidates = []
        for i, a in enumerate(top_nodes):
            for j, b in enumerate(top_nodes[i + 1 :], start=i + 1):
                if (a.id, b.id) in seen or (b.id, a.id) in seen:
                    continue
                if center_distance(a, b) < min_distance:
                    continue
                if difficulty == "difficult" and orientation in {"left_to_right", "right_to_left"} and abs(i - j) < 2:
                    continue
                candidates.append((a, b))
        rng.shuffle(candidates)
        candidates.sort(key=lambda pair: center_distance(pair[0], pair[1]), reverse=True)
        return candidates

    for pair in bidirectional_candidates()[:bidirectional]:
        add(pair[0], pair[1], "bidirectional")

    negative_edges = []
    for _ in range(negatives):
        src = rng.choice(by_tier[1])
        dst = rng.choice(by_tier[2])
        point = ((src.center[0] + dst.center[0]) // 2 + rng.randint(-24, 24), (src.center[1] + dst.center[1]) // 2)
        negative_edges.append(renderer.NegativeEdge(src.id, dst.id, rng.choice(["near_miss", "crossing_not_connected"]), point))

    name = f"{difficulty}_{sample_idx:02d}"
    return name, nodes, edges, negative_edges, [], orientation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="assets/generated_node_images_style_validation/renderer_asset_pool")
    parser.add_argument("--out", default="outputs/foodweb_difficulty_preview")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--counts", default="simple=2,medium=2,difficult=2")
    args = parser.parse_args()

    renderer = load_renderer()
    asset_index = build_asset_index(Path(args.asset_dir))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    summary = []
    counts = {}
    for item in args.counts.split(","):
        key, value = item.split("=", 1)
        counts[key.strip()] = int(value)

    for difficulty in ["simple", "medium", "difficult"]:
        difficulty_dir = out_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(1, counts.get(difficulty, 0) + 1):
            name, nodes, edges, negatives, extra, orientation = make_case(renderer, asset_index, difficulty, sample_idx, args.seed)
            renderer.render_case(name, nodes, edges, negatives, extra, difficulty_dir, args.seed)
            image_paths.append(difficulty_dir / f"{name}.png")
            summary.append(
                {
                    "name": name,
                    "difficulty": difficulty,
                    "folder": str(difficulty_dir.relative_to(out_dir)),
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "negative_edges": len(negatives),
                    "orientation": orientation,
                    "edge_types": sorted({edge.edge_type for edge in edges}),
                }
            )

    renderer.make_contact_sheet(image_paths, out_dir / "contact_sheet.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {len(image_paths)} difficulty preview diagrams in {out_dir.resolve()}")
    print(f"Contact sheet: {(out_dir / 'contact_sheet.png').resolve()}")


if __name__ == "__main__":
    main()
