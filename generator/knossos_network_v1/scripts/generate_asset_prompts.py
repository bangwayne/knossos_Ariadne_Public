#!/usr/bin/env python3


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "config" / "network_labels.json"
OUT_PATH = ROOT / "assets" / "prompts" / "network_node_asset_jobs.jsonl"

VARIANTS_PER_STYLE = 2

STYLE_SPECS: Dict[str, Dict[str, str]] = {
    "textbook_icon": {
        "style": "clean textbook-style network topology icon, simple educational diagram look",
        "composition": "single centered network device or component, square crop, white background, readable at small size",
        "prompt": (
            "Create a small textbook-style network topology icon of {display}. "
            "It should look like a clean educational diagram node asset. "
            "Use a single centered subject on a plain white background with simple shape language. "
            "Do not include any text, letters, numbers, printed labels, brand marks, UI labels, or status-label markings."
        ),
    },
    "realistic_device": {
        "style": "realistic or semi-realistic network device image, clean product-like rendering",
        "composition": "single centered subject, square crop, clean white or light neutral background, readable at small size",
        "prompt": (
            "Create a small realistic device image of {display} for a network topology diagram node. "
            "Use a single centered subject on a clean white or light neutral background. "
            "The subject should be easy to recognize at small size. "
            "Do not include any text, letters, numbers, printed labels, brand marks, UI labels, or status-label markings."
        ),
    },
    "workplace_scene": {
        "style": "natural IT workplace or infrastructure scene, realistic context but not busy",
        "composition": "single clear main subject centered, square crop, enough realistic workspace context, readable at small size",
        "prompt": (
            "Create a small workplace-environment image of {display} for a network topology diagram node. "
            "The subject should be clearly visible and centered in a natural IT or workspace setting. "
            "The background may show realistic context but should not be busy. "
            "Do not include any text, letters, numbers, printed labels, brand marks, UI labels, or status-label markings."
        ),
    },
}

CONSTRAINTS = (
    "no text, no letters, no numbers, no labels, no printed markings, no brand marks, "
    "no UI labels, no status labels, no arrows, no diagram lines, no cables crossing the frame, "
    "no border, no watermark"
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_labels() -> List[Dict[str, str]]:
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = data["network"]
    if len(labels) != 40:
        raise ValueError(f"expected 40 network labels, found {len(labels)}")
    return labels


def iter_jobs(labels: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    for item in labels:
        label = item["label"]
        display = item["display"]
        category = item["category"]
        slug = slugify(label)
        for style_name, spec in STYLE_SPECS.items():
            for variant in range(1, VARIANTS_PER_STYLE + 1):
                yield {
                    "prompt": spec["prompt"].format(display=display),
                    "use_case": "network-topology-educational",
                    "label": label,
                    "display": display,
                    "category": category,
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
    OUT_PATH.write_text(
        "".join(json.dumps(job, ensure_ascii=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {OUT_PATH}")


if __name__ == "__main__":
    main()
