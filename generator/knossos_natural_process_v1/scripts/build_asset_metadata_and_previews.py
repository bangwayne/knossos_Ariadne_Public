#!/usr/bin/env python3


from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "config" / "natural_process_labels.json"
PROMPTS = ROOT / "assets" / "prompts" / "natural_node_asset_jobs.jsonl"
PILOT_PROMPTS = ROOT / "assets" / "prompts" / "natural_node_asset_pilot_24.jsonl"
POOL = ROOT / "assets" / "renderer_asset_pool"
PILOT_POOL = ROOT / "assets" / "pilot_raw"
METADATA = POOL / "metadata.json"
PREVIEW_DIR = ROOT / "assets" / "previews"

CATEGORY_COLORS = {
    "energy": ((255, 248, 223), (177, 126, 32)),
    "climate": ((235, 247, 255), (57, 113, 155)),
    "water": ((230, 245, 255), (34, 106, 170)),
    "earth": ((246, 239, 225), (126, 94, 54)),
    "organic": ((240, 247, 230), (84, 128, 55)),
    "organism": ((234, 248, 235), (43, 128, 71)),
    "role": ((242, 238, 250), (100, 83, 156)),
    "chemical": ((244, 246, 250), (84, 95, 111)),
    "process": ((255, 246, 235), (172, 98, 38)),
    "human_nature": ((255, 238, 238), (164, 65, 65)),
}


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_labels() -> List[Dict]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))["labels"]


def load_jobs(path: Path = PROMPTS) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_text_card(item: Dict, variant: int, path: Path) -> None:
    fill, outline = CATEGORY_COLORS.get(item["category"], ((248, 248, 248), (90, 95, 105)))
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    card = (104, 246, 920, 778) if variant == 1 else (128, 224, 896, 800)
    radius = 44 if variant == 1 else 30
    draw.rounded_rectangle(card, radius=radius, fill=fill, outline=outline, width=12)
    symbol_font = load_font(86)
    title_size = 104 if len(item["display"]) <= 8 else 92 if len(item["display"]) <= 13 else 76
    title_font = load_font(title_size)
    sub_font = load_font(46)
    symbol = item["symbol"] if item["category"] == "chemical" else item["category"].replace("_", " ").title()
    lines = textwrap.wrap(item["display"], width=13)[:2]
    line_h = title_size + 12
    text_h = len(lines) * line_h
    y = card[1] + 86 if len(lines) == 1 else card[1] + 68
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        draw.text(((1024 - (bbox[2] - bbox[0])) // 2, y), line, fill=(27, 33, 42), font=title_font)
        y += line_h
    bbox = draw.textbbox((0, 0), symbol, font=sub_font if len(symbol) > 6 else symbol_font)
    font = sub_font if len(symbol) > 6 else symbol_font
    draw.text(((1024 - (bbox[2] - bbox[0])) // 2, card[3] - 118), symbol, fill=outline, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def ensure_text_assets() -> List[Dict]:
    jobs = []
    for item in load_labels():
        if item["render_mode"] != "text_only":
            continue
        for variant in [1, 2]:
            out = f"{item['label']}_text_card_{variant:02d}.png"
            path = POOL / out
            draw_text_card(item, variant, path)
            jobs.append(
                {
                    "label": item["label"],
                    "display": item["display"],
                    "category": item["category"],
                    "role": item["role"],
                    "symbol": item["symbol"],
                    "render_mode": item["render_mode"],
                    "style_name": "text_card",
                    "out": out,
                    "prompt": "deterministic local text card",
                    "constraints": "text-only scientific label card",
                }
            )
    return jobs


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
            "render_mode": job.get("render_mode", "icon"),
            "style": job["style_name"],
            "variant": int(Path(job["out"]).stem.rsplit("_", 1)[1]),
            "source": "local_text_card" if job["style_name"] == "text_card" else "generated_openai_image_api",
            "prompt": job["prompt"],
            "constraints": job["constraints"],
            "width": width,
            "height": height,
        }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_contact_sheet(jobs: List[Dict], out_path: Path, pool: Path, *, variant: int | None = 1) -> None:
    filtered = []
    for job in jobs:
        path = pool / job["out"]
        if not path.exists():
            continue
        item_variant = int(Path(job["out"]).stem.rsplit("_", 1)[1])
        if variant is not None and item_variant != variant:
            continue
        filtered.append((job, path))
    if not filtered:
        return

    labels = []
    for job, _ in filtered:
        if job["label"] not in labels:
            labels.append(job["label"])
    styles = []
    for job, _ in filtered:
        if job["style_name"] not in styles:
            styles.append(job["style_name"])

    cell_w = 170
    cell_h = 202
    thumb = 132
    left = 150
    top = 42
    font = load_font(11)
    header_font = load_font(12)
    label_font = load_font(12)
    sheet = Image.new("RGB", (left + len(styles) * cell_w, top + len(labels) * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for col, style in enumerate(styles):
        draw.text((left + col * cell_w + 8, 12), style.replace("_", " "), fill=(20, 30, 40), font=header_font)
    lookup = {(job["label"], job["style_name"]): path for job, path in filtered}
    display = {job["label"]: job["display"] for job, _ in filtered}
    for row, label in enumerate(labels):
        y = top + row * cell_h
        draw.text((8, y + 62), display[label], fill=(20, 30, 40), font=label_font)
        for col, style in enumerate(styles):
            path = lookup.get((label, style))
            if not path:
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb, thumb))
            x = left + col * cell_w + (cell_w - thumb) // 2
            sheet.paste(image, (x + (thumb - image.width) // 2, y + 22))
            draw.rectangle([x, y + 22, x + thumb, y + 22 + thumb], outline=(220, 225, 232), width=1)
            draw.text((left + col * cell_w + 8, y + 160), path.name[:22], fill=(70, 80, 90), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    generated_jobs = load_jobs()
    text_jobs = ensure_text_assets()
    all_jobs = generated_jobs + text_jobs
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata(all_jobs)
    make_contact_sheet(all_jobs, PREVIEW_DIR / "natural_assets_contact_sheet_variant01.png", POOL, variant=1)
    make_contact_sheet(all_jobs, PREVIEW_DIR / "natural_assets_contact_sheet_all.png", POOL, variant=None)
    if PILOT_PROMPTS.exists() and PILOT_POOL.exists():
        make_contact_sheet(load_jobs(PILOT_PROMPTS), PREVIEW_DIR / "natural_assets_pilot_contact_sheet.png", PILOT_POOL, variant=1)
    print(json.dumps({"pool_assets": len(list(POOL.glob("*.png"))), "text_assets": len(text_jobs), "metadata": str(METADATA), "previews": str(PREVIEW_DIR)}, indent=2))


if __name__ == "__main__":
    main()
