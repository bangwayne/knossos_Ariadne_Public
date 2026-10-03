#!/usr/bin/env python3


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "config" / "natural_process_labels.json"
OUT_PATH = ROOT / "assets" / "prompts" / "natural_node_asset_jobs.jsonl"
PILOT_OUT_PATH = ROOT / "assets" / "prompts" / "natural_node_asset_pilot_24.jsonl"

VARIANTS_PER_STYLE = 2

STYLE_SPECS: Dict[str, Dict[str, str]] = {
    "eco_flat_icon": {
        "style": "clean flat ecological science icon, vector-like educational look",
        "composition": "single centered natural science icon, square crop, plain white background",
        "prompt": (
            "Create a small clean ecological science icon representing {display}. "
            "Use a flat vector-like educational style, centered in a square crop on a plain white background. "
            "The subject should be immediately recognizable at small size as a node in a natural process network. "
            "Do not include text, letters, numbers, chemical formulas, arrows, route lines, brand marks, watermarks, or border."
        ),
    },
    "natural_scene_icon": {
        "style": "minimal natural object pictogram with slight scene context, clean and readable",
        "composition": "single centered natural object or process pictogram, light neutral background",
        "prompt": (
            "Create a small minimal pictogram of {display} for a biogeochemical network diagram node. "
            "Use one clear main natural object or process symbol, with very subtle environmental context only if helpful. "
            "Keep the image clean, centered, and readable at small size on a white or very light neutral background. "
            "Do not include text, letters, numbers, chemical formulas, arrows, route lines, brand marks, watermarks, or border."
        ),
    },
}

CONSTRAINTS = (
    "no text, no letters, no numbers, no chemical formulas, no arrows, no route lines, "
    "no brand marks, no watermarks, no border, clean centered single subject"
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_labels() -> List[Dict[str, str]]:
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = data["labels"]
    if len(labels) != 60:
        raise ValueError(f"expected 60 natural process labels, found {len(labels)}")
    return labels


def iter_jobs(labels: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    for item in labels:
        if item["render_mode"] == "text_only":
            continue
        slug = slugify(item["label"])
        for style_name, spec in STYLE_SPECS.items():
            for variant in range(1, VARIANTS_PER_STYLE + 1):
                yield {
                    "prompt": spec["prompt"].format(display=item["display"]),
                    "use_case": "natural-process-topology-educational",
                    "label": item["label"],
                    "display": item["display"],
                    "category": item["category"],
                    "role": item["role"],
                    "symbol": item["symbol"],
                    "render_mode": item["render_mode"],
                    "style_name": style_name,
                    "style": spec["style"],
                    "composition": spec["composition"],
                    "constraints": CONSTRAINTS,
                    "out": f"{slug}_{style_name}_{variant:02d}.png",
                }


def main() -> None:
    labels = load_labels()
    jobs = list(iter_jobs(labels))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(json.dumps(job, ensure_ascii=True) + "\n" for job in jobs), encoding="utf-8")

    pilot_labels = {
        "sunlight",
        "rain",
        "soil",
        "tree",
        "fungi",
        "bacteria",
        "pollution",
        "evaporation",
        "groundwater",
        "nutrient",
        "fish",
        "sediment",
    }
    pilot = [
        job
        for job in jobs
        if job["label"] in pilot_labels
        and job["style_name"] in {"eco_flat_icon", "natural_scene_icon"}
        and job["out"].endswith("_01.png")
    ]
    PILOT_OUT_PATH.write_text("".join(json.dumps(job, ensure_ascii=True) + "\n" for job in pilot), encoding="utf-8")
    print(json.dumps({"labels": len(labels), "jobs": len(jobs), "pilot_jobs": len(pilot), "out": str(OUT_PATH), "pilot": str(PILOT_OUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
