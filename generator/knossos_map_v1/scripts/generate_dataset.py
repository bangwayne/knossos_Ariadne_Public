#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO = ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))

import generate_preview_samples as PREVIEW  


DEFAULT_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"simple": 810, "medium": 2700, "difficult": 1890},
    "test": {"simple": 90, "medium": 300, "difficult": 210},
}


def parse_counts(text: str | None) -> Dict[str, Dict[str, int]]:
    if not text:
        return DEFAULT_COUNTS
    result: Dict[str, Dict[str, int]] = {}
    for split_spec in text.split(";"):
        split, values = split_spec.split(":", 1)
        result[split.strip()] = {}
        for item in values.split(","):
            key, value = item.split("=", 1)
            result[split.strip()][key.strip()] = int(value)
    return result


def make_contact_sheet(image_paths: List[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((320, 224))
        thumbs.append((path.stem, image.copy()))
    cols = 3
    cell_w = 370
    cell_h = 268
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = PREVIEW.RENDERER.load_font(14)
    for idx, (title, image) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 16
        y = (idx // cols) * cell_h + 36
        draw.text((x, y - 22), title, fill=(28, 35, 45), font=font)
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + image.width, y + image.height], outline=(210, 215, 222), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_map_v1/outputs/Knossos-MapRoute-v1-raw6000")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--counts", default=None, help="Optional: train:simple=1,medium=1,difficult=1;test:simple=...")
    parser.add_argument("--layout-mode", choices=["map70", "map85"], default="map70")
    args = parser.parse_args()

    out = Path(args.out)
    counts = parse_counts(args.counts)
    summary = []
    contact_candidates: List[Path] = []
    global_index = 1

    for split, split_counts in counts.items():
        for difficulty in ["simple", "medium", "difficult"]:
            count = split_counts.get(difficulty, 0)
            for local_idx in range(1, count + 1):
                style_mode = PREVIEW.STYLE_MODES[(global_index - 1) % len(PREVIEW.STYLE_MODES)]
                sample_id = f"{split}_{difficulty}_{local_idx:06d}"
                image_path = out / "images" / split / difficulty / f"{sample_id}.png"
                ann_path = out / "annotations" / split / difficulty / f"{sample_id}.json"
                ann = PREVIEW.annotation_for(
                    sample_id,
                    split,
                    style_mode,
                    global_index,
                    args.seed,
                    image_path,
                    out,
                    difficulty,
                    args.layout_mode,
                )
                ann_path.parent.mkdir(parents=True, exist_ok=True)
                ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")
                family = PREVIEW.layout_family_for_mode(global_index, args.layout_mode)
                summary.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "difficulty": difficulty,
                        "nodes": len(ann["nodes"]),
                        "edges": len(ann["edges"]),
                        "negative_edges": len(ann["negative_edges"]),
                        "layout_family": family,
                        "layout": ann["generation"]["layout"],
                        "image": ann["image"],
                        "annotation": str(ann_path.relative_to(out)),
                        "edge_types": sorted({edge["type"] for edge in ann["edges"]}),
                    }
                )
                if local_idx <= 3 or (split == "test" and local_idx <= 5):
                    contact_candidates.append(image_path)
                global_index += 1
            print(f"generated {split}/{difficulty}: {count}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    family_counts = Counter(item["layout_family"] for item in summary)
    split_counts = Counter(item["split"] for item in summary)
    difficulty_counts = Counter(item["difficulty"] for item in summary)
    (out / "generation_report.json").write_text(
        json.dumps(
            {
                "total": len(summary),
                "splits": dict(split_counts),
                "difficulties": dict(difficulty_counts),
                "layout_families": dict(family_counts),
                "seed": args.seed,
                "layout_mode": args.layout_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_contact_sheet(contact_candidates[:36], out / "previews" / "contact_sheet.png")
    print(json.dumps({"samples": len(summary), "out": str(out), "contact_sheet": str(out / "previews" / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
