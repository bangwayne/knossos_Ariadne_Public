#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "assets" / "prompts" / "map_node_asset_jobs.jsonl"
PILOT_PROMPTS = ROOT / "assets" / "prompts" / "map_node_asset_pilot_24.jsonl"
POOL = ROOT / "assets" / "renderer_asset_pool"
PILOT_POOL = ROOT / "assets" / "pilot_raw"
METADATA = POOL / "metadata.json"
PREVIEW_DIR = ROOT / "assets" / "previews"


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


def load_jobs(path: Path = PROMPTS) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_metadata(jobs: List[Dict]) -> None:
    metadata = {}
    for job in jobs:
        path = POOL / job["out"]
        if not path.exists():
            continue
        with Image.open(path) as image:
            width, height = image.size
        metadata[job["out"]] = {
            "label": job["label"],
            "display": job["display"],
            "category": job["category"],
            "role": job["role"],
            "symbol": job["symbol"],
            "slug": job["out"].rsplit("_", 3)[0],
            "style": job["style_name"],
            "variant": int(Path(job["out"]).stem.rsplit("_", 1)[1]),
            "source": "generated_openai_image_api",
            "prompt": job["prompt"],
            "constraints": job["constraints"],
            "width": width,
            "height": height,
        }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_contact_sheet(jobs: List[Dict], out_path: Path, pool: Path, *, variant: int | None = None) -> None:
    filtered = []
    for job in jobs:
        stem = Path(job["out"]).stem
        item_variant = int(stem.rsplit("_", 1)[1])
        if variant is not None and item_variant != variant:
            continue
        path = pool / job["out"]
        if path.exists():
            filtered.append((job, path))

    if not filtered:
        return

    labels = []
    for job, _ in filtered:
        if job["label"] not in labels:
            labels.append(job["label"])
    styles = ["flat_map_icon", "marker_icon", "place_scene_icon"]
    variants = [1] if variant is not None else [1, 2]
    columns = [(style, item_variant) for style in styles for item_variant in variants]

    cell_w = 190
    cell_h = 220
    thumb_size = 150
    left_margin = 140
    top_margin = 42
    font = load_font(12)
    header_font = load_font(13)
    label_font = load_font(13)

    sheet = Image.new("RGB", (left_margin + len(columns) * cell_w, top_margin + len(labels) * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for col, (style, item_variant) in enumerate(columns):
        title = f"{style.replace('_', ' ')} {item_variant}"
        draw.text((left_margin + col * cell_w + 10, 12), title, fill=(20, 30, 40), font=header_font)

    lookup = {(job["label"], job["style_name"], int(Path(job["out"]).stem.rsplit("_", 1)[1])): path for job, path in filtered}
    display_by_label = {job["label"]: job["display"] for job, _ in filtered}

    for row, label in enumerate(labels):
        y = top_margin + row * cell_h
        draw.text((8, y + 70), display_by_label[label], fill=(20, 30, 40), font=label_font)
        for col, (style, item_variant) in enumerate(columns):
            path = lookup.get((label, style, item_variant))
            if not path:
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            x = left_margin + col * cell_w + (cell_w - thumb_size) // 2
            sheet.paste(image, (x + (thumb_size - image.width) // 2, y + 28))
            draw.rectangle([x, y + 28, x + thumb_size, y + 28 + thumb_size], outline=(220, 225, 232), width=1)
            draw.text((left_margin + col * cell_w + 10, y + 184), path.name[:24], fill=(70, 80, 90), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    jobs = load_jobs()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    if POOL.exists():
        write_metadata(jobs)
        make_contact_sheet(jobs, PREVIEW_DIR / "map_assets_contact_sheet_variant01.png", POOL, variant=1)
        make_contact_sheet(jobs, PREVIEW_DIR / "map_assets_contact_sheet_all.png", POOL, variant=None)
    if PILOT_PROMPTS.exists() and PILOT_POOL.exists():
        pilot_jobs = load_jobs(PILOT_PROMPTS)
        make_contact_sheet(pilot_jobs, PREVIEW_DIR / "map_assets_pilot_contact_sheet.png", PILOT_POOL, variant=1)
    print(json.dumps({"pool_assets": len(list(POOL.glob("*.png"))), "pilot_assets": len(list(PILOT_POOL.glob("*.png"))), "previews": str(PREVIEW_DIR)}, indent=2))


if __name__ == "__main__":
    main()
