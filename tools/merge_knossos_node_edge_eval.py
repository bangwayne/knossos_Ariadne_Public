#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from evaluate_knossos_node_edge_typed_pipeline import report_for_items


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    shard_dirs = [args.eval_root / f"shard{index}" for index in range(args.num_shards)]
    details = []
    for shard_dir in shard_dirs:
        report_path = shard_dir / "eval_report.json"
        metrics_path = shard_dir / "per_image_metrics.jsonl"
        if not report_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"incomplete shard: {shard_dir}")
        details.extend(load_jsonl(metrics_path))

    first_report = json.loads((shard_dirs[0] / "eval_report.json").read_text(encoding="utf-8"))
    report_args = dict(first_report.get("args") or {})
    report_args["output_dir"] = str(args.output_dir)
    report_args["merged_from"] = [str(path) for path in shard_dirs]
    report = report_for_items(details, report_args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
