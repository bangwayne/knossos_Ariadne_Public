#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate_preview_samples.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("knossos_workflow_preview_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = load_generator()


def write_dataset(out_dir: Path, split: str, counts: Dict[str, int], seed: int, preview_per_difficulty: int) -> Dict:
    summary: List[Dict] = []
    preview_paths: List[Path] = []
    total = sum(counts.values())
    done = 0

    for difficulty in ["simple", "medium", "difficult"]:
        for index in range(1, counts[difficulty] + 1):
            sample_id = f"{split}_{difficulty}_{index:06d}"
            image_path = out_dir / "images" / split / difficulty / f"{sample_id}.png"
            annotation_path = out_dir / "annotations" / split / difficulty / f"{sample_id}.json"
            annotation = GEN.annotation_for(sample_id, split, difficulty, index, seed, image_path, out_dir)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")

            if index <= preview_per_difficulty:
                preview_paths.append(image_path)

            summary.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "difficulty": difficulty,
                    "topology": annotation["generation"]["topology"],
                    "orientation": annotation["orientation"],
                    "nodes": len(annotation["nodes"]),
                    "edges": len(annotation["edges"]),
                    "negative_edges": len(annotation["negative_edges"]),
                    "image_scale": annotation["generation"]["image_scale"],
                    "canvas": annotation["canvas"],
                    "image": annotation["image"],
                    "annotation": str(annotation_path.relative_to(out_dir)),
                    "edge_types": sorted({edge["type"] for edge in annotation["edges"]}),
                }
            )
            done += 1
            if done % 100 == 0 or done == total:
                print(json.dumps({"progress": done, "total": total, "difficulty": difficulty}, ensure_ascii=False), flush=True)

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    GEN.make_contact_sheet(preview_paths, out_dir / "previews" / "contact_sheet.png")
    return {
        "samples": len(summary),
        "out": str(out_dir),
        "counts": counts,
        "summary": str(out_dir / "dataset_summary.json"),
        "contact_sheet": str(out_dir / "previews" / "contact_sheet.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_workflow_v1/outputs/Knossos-Workflow-v1-raw6000")
    parser.add_argument("--split", default="raw")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--simple", type=int, default=900)
    parser.add_argument("--medium", type=int, default=3000)
    parser.add_argument("--difficult", type=int, default=2100)
    parser.add_argument("--preview-per-difficulty", type=int, default=8)
    args = parser.parse_args()

    counts = {"simple": args.simple, "medium": args.medium, "difficult": args.difficult}
    result = write_dataset(Path(args.out), args.split, counts, args.seed, args.preview_per_difficulty)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
