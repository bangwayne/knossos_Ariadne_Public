#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_PATH = ROOT / "scripts" / "generate_preview_samples.py"


def load_preview_module():
    spec = importlib.util.spec_from_file_location("knossos_natural_preview", PREVIEW_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREVIEW = load_preview_module()

STYLE_MODES = ["eco_flat_icon", "natural_scene_icon", "mixed"]

PROFILE_COUNTS = {
    "train": {
        "simple": {"single": 486, "typed": 324, "multi": 0},
        "medium": {"single": 1350, "typed": 945, "multi": 405},
        "difficult": {"single": 864, "typed": 621, "multi": 405},
    },
    "test": {
        "simple": {"single": 54, "typed": 36, "multi": 0},
        "medium": {"single": 150, "typed": 105, "multi": 45},
        "difficult": {"single": 96, "typed": 69, "multi": 45},
    },
}


def expand_profiles(counts: Dict[str, int], rng: random.Random) -> List[str]:
    profiles = []
    for profile, count in counts.items():
        profiles.extend([profile] * count)
    rng.shuffle(profiles)
    return profiles


def sample_preview_images(all_images: List[Path], rng: random.Random, n: int = 60) -> List[Path]:
    if len(all_images) <= n:
        return all_images
    by_bucket: Dict[str, List[Path]] = defaultdict(list)
    for path in all_images:
        split = path.parts[-3]
        difficulty = path.parts[-2]
        by_bucket[f"{split}:{difficulty}"].append(path)
    selected: List[Path] = []
    per_bucket = max(1, n // max(1, len(by_bucket)))
    for paths in by_bucket.values():
        rng.shuffle(paths)
        selected.extend(paths[:per_bucket])
    if len(selected) < n:
        remaining = [path for path in all_images if path not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


def generate(out_dir: Path, seed: int) -> Dict:
    rng = random.Random(seed)
    all_images: List[Path] = []
    summary_rows = []
    counters = {
        "split": Counter(),
        "difficulty": Counter(),
        "relation_profile": Counter(),
        "legend_position": Counter(),
        "edge_type_count": Counter(),
        "image_scale": Counter(),
        "canvas": Counter(),
    }
    global_index = 0

    for split in ["train", "test"]:
        for difficulty in ["simple", "medium", "difficult"]:
            profiles = expand_profiles(PROFILE_COUNTS[split][difficulty], random.Random(f"{seed}:{split}:{difficulty}:profiles"))
            for local_index, relation_profile in enumerate(profiles, 1):
                global_index += 1
                style_mode = STYLE_MODES[(global_index - 1) % len(STYLE_MODES)]
                sample_id = f"{split}_{difficulty}_{local_index:06d}"
                image_path = out_dir / "images" / split / difficulty / f"{sample_id}.png"
                ann_path = out_dir / "annotations" / split / difficulty / f"{sample_id}.json"
                ann = PREVIEW.annotation_for(
                    sample_id=sample_id,
                    split=split,
                    style_mode=style_mode,
                    index=global_index,
                    seed=seed,
                    image_path=image_path,
                    out_dir=out_dir,
                    difficulty=difficulty,
                    relation_profile=relation_profile,
                )
                ann_path.parent.mkdir(parents=True, exist_ok=True)
                ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")
                all_images.append(image_path)

                edge_types = sorted({edge["type"] for edge in ann["edges"]})
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "difficulty": difficulty,
                    "relation_profile": relation_profile,
                    "nodes": len(ann["nodes"]),
                    "edges": len(ann["edges"]),
                    "edge_types": edge_types,
                    "legend_position": ann["render_style"]["legend_position"],
                    "image_scale": ann["render_style"]["image_scale"],
                    "canvas": ann["canvas"],
                    "image": ann["image"],
                    "annotation": str(ann_path.relative_to(out_dir)),
                }
                summary_rows.append(row)
                counters["split"][split] += 1
                counters["difficulty"][difficulty] += 1
                counters["relation_profile"][relation_profile] += 1
                counters["legend_position"][row["legend_position"]] += 1
                counters["edge_type_count"][len(edge_types)] += 1
                counters["image_scale"][f'{row["image_scale"]:.2f}'] += 1
                counters["canvas"][f'{row["canvas"]["width"]}x{row["canvas"]["height"]}'] += 1

    preview_images = sample_preview_images(all_images, rng, n=60)
    PREVIEW.make_contact_sheet(preview_images, out_dir / "previews" / "contact_sheet.png")

    dataset_summary = {
        "dataset": "Knossos-NaturalProcess-v1-6000-adaptive-arrow",
        "total_samples": len(summary_rows),
        "seed": seed,
        "counts": {key: dict(value) for key, value in counters.items()},
        "profile_counts_by_split_difficulty": PROFILE_COUNTS,
        "samples": summary_rows,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_natural_process_v1/outputs/Knossos-NaturalProcess-v1-6000-adaptive-arrow")
    parser.add_argument("--seed", type=int, default=9000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    summary = generate(out_dir, args.seed)
    print(json.dumps({
        "dataset": summary["dataset"],
        "total_samples": summary["total_samples"],
        "out": str(out_dir),
        "counts": summary["counts"],
        "contact_sheet": str(out_dir / "previews" / "contact_sheet.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
