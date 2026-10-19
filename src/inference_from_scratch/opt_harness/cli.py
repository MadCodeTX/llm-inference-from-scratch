"""Command-line entry point: deterministic demo and metrics import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demo import load_metrics_file, run_demo
from .selection import select_winner
from .spec import MetricSpec


def _parse_metric(text: str) -> MetricSpec:
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"metric must be name:direction[:priority], got {text!r}"
        )
    return MetricSpec(parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opt_harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the deterministic fake task end-to-end")
    demo.add_argument("--work-dir", default="opt_harness_demo_out")
    demo.add_argument("--k", type=int, default=1)
    demo.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    demo.add_argument("--no-library", action="store_true")

    imp = sub.add_parser("import-metrics", help="select a winner from a metrics JSON artefact")
    imp.add_argument("metrics_json")
    imp.add_argument(
        "--metric",
        action="append",
        type=_parse_metric,
        dest="metrics",
        help="name:direction[:priority] (repeatable); default throughput/latency",
    )
    imp.add_argument("--task-id", default="imported")

    args = parser.parse_args(argv)

    if args.command == "demo":
        result = run_demo(
            args.work_dir, k=args.k, mode=args.mode, use_library=not args.no_library
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        out = Path(args.work_dir) / "result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return 0 if result.ok else 1

    results = load_metrics_file(args.metrics_json)
    metrics = args.metrics or [
        MetricSpec("throughput", "higher", 1),
        MetricSpec("latency_ms", "lower", 2),
    ]
    selection = select_winner(results, metrics, task_id=args.task_id)
    report = {
        "selection": selection.to_dict(),
        "frontier": selection.frontier_labels,
        "results": [r.to_dict() for r in results],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if selection.ok else 1


if __name__ == "__main__":
    sys.exit(main())
