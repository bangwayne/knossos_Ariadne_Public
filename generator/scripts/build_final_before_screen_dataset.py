#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


DIFFICULTIES = ["simple", "medium", "difficult"]
TARGET_SPLITS = {"train": 5400, "test": 600}


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    display_name: str
    source_dir: Path


DOMAIN_SPECS = [
    DomainSpec("foodweb", "FoodWeb", Path("knossos_foodweb_v0/outputs/Knossos-FoodWeb-v0-6000-adaptive")),
    DomainSpec("network", "Network", Path("knossos_network_v1/outputs/Knossos-Network-v1-6000-dense-adaptive-color")),
    DomainSpec("workflow", "Workflow", Path("knossos_workflow_v1/outputs/Knossos-Workflow-v1-6000-v2-adaptive-scale")),
    DomainSpec("natural_process", "Natural Process", Path("knossos_natural_process_v1/outputs/Knossos-NaturalProcess-v1-6000-adaptive-arrow")),
    DomainSpec("circuit", "Circuit", Path("knossos_circuit_v1/outputs/Knossos-Circuit-v1-6000-layered-mixed-unified-resolution")),
    DomainSpec("map_route", "MapRoute", Path("knossos_map_v1/outputs/Knossos-MapRoute-v2-map85-direction-mix-canvaspool-split6000")),
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def source_image_path(source_dir: Path, ann: Dict[str, Any], ann_path: Path) -> Path:
    image_rel = ann.get("image")
    if image_rel:
        candidate = source_dir / image_rel
        if candidate.exists():
            return candidate
    candidates = list((source_dir / "images").glob(f"**/{ann_path.stem}.png"))
    if not candidates:
        raise FileNotFoundError(f"Missing source image for {ann_path}")
    return candidates[0]


def assign_splits(paths: List[Path], seed: int, domain: str) -> Dict[Path, str]:
    by_difficulty: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        ann = load_json(path)
        by_difficulty[ann["difficulty"]].append(path)

    assignment: Dict[Path, str] = {}
    for difficulty in DIFFICULTIES:
        items = sorted(by_difficulty[difficulty])
        rng = random.Random(f"{seed}:{domain}:{difficulty}:final-before-screen-split")
        rng.shuffle(items)
        test_count = round(len(items) * TARGET_SPLITS["test"] / sum(TARGET_SPLITS.values()))
        for idx, path in enumerate(items):
            assignment[path] = "test" if idx < test_count else "train"
    return assignment


def build(out_dir: Path, repo_root: Path, seed: int, overwrite: bool) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"Refusing to overwrite non-empty output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    counters: Dict[str, Counter] = defaultdict(Counter)
    global_index = 0
    local_split_counters: Dict[tuple[str, str, str], int] = defaultdict(int)

    for spec in DOMAIN_SPECS:
        source_dir = repo_root / spec.source_dir
        ann_paths = sorted((source_dir / "annotations").rglob("*.json"))
        if len(ann_paths) != 6000:
            raise ValueError(f"{spec.domain} expected 6000 annotations, found {len(ann_paths)} under {source_dir}")

        assignments = assign_splits(ann_paths, seed, spec.domain)
        for ann_path in ann_paths:
            ann = load_json(ann_path)
            source_sample_id = ann["sample_id"]
            source_split = ann.get("split")
            difficulty = ann["difficulty"]
            split = assignments[ann_path]
            local_split_counters[(spec.domain, split, difficulty)] += 1
            global_index += 1

            sample_id = f"{spec.domain}_{split}_{difficulty}_{local_split_counters[(spec.domain, split, difficulty)]:06d}"
            image_src = source_image_path(source_dir, ann, ann_path)
            image_dst = out_dir / "images" / split / spec.domain / difficulty / f"{sample_id}.png"
            ann_dst = out_dir / "annotations" / split / spec.domain / difficulty / f"{sample_id}.json"
            link_or_copy(image_src, image_dst)

            ann["sample_id"] = sample_id
            ann["split"] = split
            ann["image"] = str(image_dst.relative_to(out_dir))
            generation = ann.setdefault("generation", {})
            generation["domain"] = spec.domain
            generation["domain_name"] = spec.display_name
            generation["source_sample_id"] = source_sample_id
            generation["source_split"] = source_split
            generation["source_annotation"] = str(ann_path.relative_to(repo_root))
            generation["source_image"] = str(image_src.relative_to(repo_root))
            generation["final_before_screen_seed"] = seed

            write_json(ann_dst, ann)

            item = {
                "sample_id": sample_id,
                "domain": spec.domain,
                "domain_name": spec.display_name,
                "split": split,
                "difficulty": difficulty,
                "image": ann["image"],
                "annotation": str(ann_dst.relative_to(out_dir)),
                "source_sample_id": source_sample_id,
                "source_split": source_split,
                "source_annotation": str(ann_path.relative_to(repo_root)),
                "source_image": str(image_src.relative_to(repo_root)),
                "nodes": len(ann.get("nodes", [])),
                "edges": len(ann.get("edges", [])),
                "canvas": ann.get("canvas"),
            }
            manifest.append(item)

            counters["domain"][spec.domain] += 1
            counters["split"][split] += 1
            counters["domain_split"][(spec.domain, split)] += 1
            counters["difficulty"][(split, difficulty)] += 1
            counters["domain_difficulty"][(spec.domain, difficulty)] += 1
            canvas = ann.get("canvas", {})
            counters["canvas"][f"{canvas.get('width')}x{canvas.get('height')}"] += 1

    summary = {
        "dataset": "final-before-screen-data",
        "total_samples": len(manifest),
        "seed": seed,
        "target_splits_per_domain": TARGET_SPLITS,
        "domain_counts": dict(counters["domain"]),
        "split_counts": dict(counters["split"]),
        "domain_split_counts": {f"{domain}/{split}": count for (domain, split), count in counters["domain_split"].items()},
        "split_difficulty_counts": {f"{split}/{difficulty}": count for (split, difficulty), count in counters["difficulty"].items()},
        "domain_difficulty_counts": {f"{domain}/{difficulty}": count for (domain, difficulty), count in counters["domain_difficulty"].items()},
        "canvas_counts": dict(counters["canvas"]),
    }

    write_jsonl(out_dir / "manifest.jsonl", manifest)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/final-before-screen-data")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary = build(Path(args.out), repo_root, args.seed, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
