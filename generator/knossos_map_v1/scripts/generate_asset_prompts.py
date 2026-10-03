#!/usr/bin/env python3


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "config" / "map_labels.json"
OUT_PATH = ROOT / "assets" / "prompts" / "map_node_asset_jobs.jsonl"
PILOT_OUT_PATH = ROOT / "assets" / "prompts" / "map_node_asset_pilot_24.jsonl"

VARIANTS_PER_STYLE = 2

STYLE_SPECS: Dict[str, Dict[str, str]] = {
    "flat_map_icon": {
        "style": "clean flat map or navigation icon, simple vector-like educational look",
        "composition": "single centered place icon, square crop, plain white background, readable at small size",
        "prompt": (
            "Create a small flat map/navigation icon representing {display}. "
            "Use a clean vector-like educational style, centered in a square crop on a plain white background. "
            "The icon should be immediately recognizable at small size and suitable as a node in a route diagram. "
            "Do not include any text, letters, numbers, map labels, arrows, route lines, brand marks, watermarks, or border."
        ),
    },
    "marker_icon": {
        "style": "simple location marker or transit-map node icon, clean diagram style",
        "composition": "single centered marker-style icon, square crop, plain white background, readable at small size",
        "prompt": (
            "Create a small map marker-style icon representing {display}. "
            "It may use a pin, station marker, doorway marker, or simplified location symbol, but it must remain uncluttered. "
            "Place one centered symbol in a square crop on a plain white background. "
            "Do not include any text, letters, numbers, map labels, arrows, route lines, brand marks, watermarks, or border."
        ),
    },
    "place_scene_icon": {
        "style": "minimal place or facility pictogram with slight scene context, clean and not busy",
        "composition": "single centered place/facility pictogram, light neutral background, readable at small size",
        "prompt": (
            "Create a small minimal pictogram of {display} for a map route diagram node. "
            "Use a single clear main place or facility symbol with very subtle context if helpful. "
            "Keep it clean, centered, and readable at small size on a white or light neutral background. "
            "Do not include any text, letters, numbers, map labels, arrows, route lines, brand marks, watermarks, or border."
        ),
    },
}

CONSTRAINTS = (
    "no text, no letters, no numbers, no map labels, no arrows, no route lines, no roads extending outside the icon, "
    "no brand marks, no watermarks, no border, clean centered single subject"
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_labels() -> List[Dict[str, str]]:
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = data["map"]
    if len(labels) != 60:
        raise ValueError(f"expected 60 map labels, found {len(labels)}")
    return labels


def iter_jobs(labels: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    for item in labels:
        label = item["label"]
        display = item["display"]
        category = item["category"]
        role = item["role"]
        symbol = item["symbol"]
        slug = slugify(label)
        for style_name, spec in STYLE_SPECS.items():
            for variant in range(1, VARIANTS_PER_STYLE + 1):
                yield {
                    "prompt": spec["prompt"].format(display=display),
                    "use_case": "map-route-topology-educational",
                    "label": label,
                    "display": display,
                    "category": category,
                    "role": role,
                    "symbol": symbol,
                    "style_name": style_name,
                    "style": spec["style"],
                    "composition": spec["composition"],
                    "constraints": CONSTRAINTS,
                    "out": f"{slug}_{style_name}_{variant:02d}.png",
                }


def main() -> None:
    labels = load_labels()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs = list(iter_jobs(labels))
    OUT_PATH.write_text("".join(json.dumps(job, ensure_ascii=True) + "\n" for job in jobs), encoding="utf-8")

    pilot_labels = {
        "station",
        "laboratory",
        "library",
        "loading dock",
        "bridge",
        "junction",
        "elevator",
        "checkpoint",
    }
    pilot = [
        job
        for job in jobs
        if job["label"] in pilot_labels
        and job["style_name"] in {"flat_map_icon", "marker_icon", "place_scene_icon"}
        and job["out"].endswith("_01.png")
    ]
    PILOT_OUT_PATH.write_text("".join(json.dumps(job, ensure_ascii=True) + "\n" for job in pilot), encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "pilot_jobs": len(pilot), "out": str(OUT_PATH), "pilot": str(PILOT_OUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
