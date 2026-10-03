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
    spec = importlib.util.spec_from_file_location("knossos_network_preview_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = load_generator()


def annotation_for(sample_id: str, split: str, difficulty: str, index: int, seed: int, image_path: Path, out_dir: Path) -> Dict:
    nodes, edges, negatives, extra_paths, topology, orientation = GEN.make_sample(difficulty, index, seed)
    image_scale = GEN.choose_image_scale(difficulty, nodes, edges, seed, sample_id)
    canvas = GEN.output_canvas(image_scale)
    render_style = GEN.render_image(f"{difficulty}_{sample_id}", nodes, edges, extra_paths, image_path, seed, image_scale)
    names = GEN.node_names(nodes)
    return {
        "sample_id": sample_id,
        "image": str(image_path.relative_to(out_dir)),
        "split": split,
        "difficulty": difficulty,
        "orientation": orientation,
        "canvas": canvas,
        "nodes": [GEN.node_record(node, image_scale) for node in GEN.reading_order(nodes)],
        "edges": [GEN.edge_record(edge, names, image_scale) for edge in edges],
        "negative_edges": [
            {
                "source": negative.source,
                "source_name": names[negative.source],
                "target": negative.target,
                "target_name": names[negative.target],
                "reason": negative.reason,
                "evidence_point": GEN.scale_point(negative.evidence_point, image_scale),
            }
            for negative in negatives
        ],
        "final_graph": GEN.final_graph(nodes, edges, image_scale),
        "think_graph": GEN.think_graph(nodes, edges, negatives, image_scale),
        "training_text": GEN.training_text(nodes, edges, negatives, image_scale),
        "render_style": render_style,
        "generation": {"seed": seed, "index": index, "domain": "network", "topology": topology, "image_scale": image_scale},
    }


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
            annotation = annotation_for(sample_id, split, difficulty, index, seed, image_path, out_dir)
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
    parser.add_argument("--out", default="knossos_network_v1/outputs/Knossos-Network-v1-raw6000")
    parser.add_argument("--split", default="raw")
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--simple", type=int, default=1200)
    parser.add_argument("--medium", type=int, default=3600)
    parser.add_argument("--difficult", type=int, default=1200)
    parser.add_argument("--preview-per-difficulty", type=int, default=8)
    args = parser.parse_args()

    counts = {"simple": args.simple, "medium": args.medium, "difficult": args.difficult}
    result = write_dataset(Path(args.out), args.split, counts, args.seed, args.preview_per_difficulty)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
