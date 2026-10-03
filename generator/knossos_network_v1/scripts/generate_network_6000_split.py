#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dataset as base  


SPLIT_COUNTS = {
    "train": {"simple": 540, "medium": 2700, "difficult": 2160},
    "test": {"simple": 60, "medium": 300, "difficult": 240},
}


def write_split(out_dir: Path, split: str, counts: Dict[str, int], seed: int, preview_per_difficulty: int) -> List[Dict]:
    summary: List[Dict] = []
    preview_paths: List[Path] = []
    total = sum(counts.values())
    done = 0

    for difficulty in ["simple", "medium", "difficult"]:
        for index in range(1, counts[difficulty] + 1):
            sample_id = f"{split}_{difficulty}_{index:06d}"
            image_path = out_dir / "images" / split / difficulty / f"{sample_id}.png"
            annotation_path = out_dir / "annotations" / split / difficulty / f"{sample_id}.json"
            annotation = base.annotation_for(sample_id, split, difficulty, index, seed, image_path, out_dir)
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
                    "edge_color": annotation["render_style"]["edge_color"],
                    "image": annotation["image"],
                    "annotation": str(annotation_path.relative_to(out_dir)),
                    "edge_types": sorted({edge["type"] for edge in annotation["edges"]}),
                }
            )
            done += 1
            if done % 100 == 0 or done == total:
                print(
                    json.dumps(
                        {"split": split, "progress": done, "total": total, "difficulty": difficulty},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    base.GEN.make_contact_sheet(preview_paths, out_dir / "previews" / f"{split}_contact_sheet.png")
    return summary


def count_files(out_dir: Path) -> Dict:
    return {
        "images": len(list((out_dir / "images").rglob("*.png"))),
        "annotations": len(list((out_dir / "annotations").rglob("*.json"))),
        "by_split": {
            split: {
                difficulty: {
                    "images": len(list((out_dir / "images" / split / difficulty).glob("*.png"))),
                    "annotations": len(list((out_dir / "annotations" / split / difficulty).glob("*.json"))),
                }
                for difficulty in ["simple", "medium", "difficult"]
            }
            for split in ["train", "test"]
        },
    }


def validate_schema(out_dir: Path) -> Dict:
    validator = ROOT.parent / "knossos_network_v1" / "scripts" / "validate_foodweb_schema_alignment.py"
    proc = subprocess.run(
        [sys.executable, str(validator), str(out_dir / "annotations"), "--strict-top-level"],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knossos_network_v1/outputs/Knossos-Network-v1-6000-dense-adaptive-color")
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--preview-per-difficulty", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[Dict] = []
    for split, counts in SPLIT_COUNTS.items():
        all_summary.extend(write_split(out_dir, split, counts, args.seed, args.preview_per_difficulty))

    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    report = {
        "out": str(out_dir),
        "total": len(all_summary),
        "counts": SPLIT_COUNTS,
        "summary": str(summary_path),
        "contact_sheets": {
            "train": str(out_dir / "previews" / "train_contact_sheet.png"),
            "test": str(out_dir / "previews" / "test_contact_sheet.png"),
        },
        "file_counts": count_files(out_dir),
        "schema_validation": validate_schema(out_dir),
    }
    (out_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if report["file_counts"]["images"] != 6000 or report["file_counts"]["annotations"] != 6000:
        raise SystemExit("Generated file count mismatch")
    if report["schema_validation"]["returncode"] != 0:
        raise SystemExit("Schema validation failed")


if __name__ == "__main__":
    main()
