#!/usr/bin/env python3


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "config" / "circuit_labels.json"
OUT_PATH = ROOT / "assets" / "prompts" / "circuit_node_asset_jobs.jsonl"
PILOT_OUT_PATH = ROOT / "assets" / "prompts" / "circuit_node_asset_pilot_24.jsonl"

VARIANTS_PER_STYLE = 2

STYLE_SPECS: Dict[str, Dict[str, str]] = {
    "textbook_symbol": {
        "style": "clean textbook-style circuit component symbol, simple educational diagram look",
        "composition": "single centered component symbol, square crop, plain white background, readable at small size",
        "prompt": (
            "Create a small textbook-style circuit diagram icon of {display} using pure schematic geometry only. "
            "It should look like a clean educational electronics component symbol, centered in a square crop. "
            "Use simple black or muted-color line art on a plain white background. "
            "Represent the component with abstract shapes, plates, coils, junctions, pins, or component outlines as appropriate. "
            "Do not include any text, letters, numbers, plus signs, minus signs, schematic labels, value markings, brand marks, arrows, wires leaving the frame, or watermarks."
        ),
    },
    "realistic_component": {
        "style": "realistic or semi-realistic electronic component image, clean product-like rendering",
        "composition": "single centered electronic component, square crop, clean white or light neutral background",
        "prompt": (
            "Create a small realistic electronic component image of {display} for a circuit topology diagram node. "
            "Use a single centered subject on a clean white or light neutral background. "
            "The component should be recognizable at small size with minimal clutter. "
            "Do not include any text, letters, numbers, printed labels, value markings, brand marks, arrows, diagram lines, or watermarks."
        ),
    },
    "workbench_scene": {
        "style": "clean electronics workbench or breadboard-like scene, realistic context but not busy",
        "composition": "single clear main component centered, square crop, subtle electronics context, readable at small size",
        "prompt": (
            "Create a small electronics workbench image of {display} for a circuit diagram node. "
            "The component should be clearly visible and centered, with only subtle breadboard or workbench context. "
            "The background must not be busy. "
            "Do not include any text, letters, numbers, printed labels, value markings, brand marks, arrows, diagram lines, or watermarks."
        ),
    },
}

CONSTRAINTS = (
    "no text, no letters, no numbers, no plus signs, no minus signs, no value markings, no labels, no printed markings, "
    "no brand marks, no arrows, no diagram lines, no wires crossing the frame, no watermark, no border"
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_labels() -> List[Dict[str, str]]:
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = data["circuit"]
    if len(labels) != 60:
        raise ValueError(f"expected 60 circuit labels, found {len(labels)}")
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
                    "use_case": "circuit-topology-educational",
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

    pilot_labels = {"battery", "resistor", "capacitor", "led", "switch", "microcontroller", "temperature sensor", "motor"}
    pilot = [job for job in jobs if job["label"] in pilot_labels and job["style_name"] in {"textbook_symbol", "realistic_component", "workbench_scene"} and job["out"].endswith("_01.png")]
    PILOT_OUT_PATH.write_text("".join(json.dumps(job, ensure_ascii=True) + "\n" for job in pilot), encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "pilot_jobs": len(pilot), "out": str(OUT_PATH), "pilot": str(PILOT_OUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
