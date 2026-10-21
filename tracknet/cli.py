"""Command line interface: python -m tracknet <demo|process|serve|evaluate>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import Store


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import build_demo, evaluate_pipeline
    store = Store(args.db)
    print(f"Rendering {args.vehicles} synthetic vehicles across the camera graph and running the pipeline...")
    result = build_demo(Path(args.data) / "demo", store, n_vehicles=args.vehicles, seed=args.seed)
    for cam, s in result["stats"].items():
        print(f"  {cam}: {s['tracks']} tracks  ({s['processing_fps']} frames/s)")
    report = evaluate_pipeline(store, result["ground_truth"])
    print("Evaluation against ground truth (synthetic data):")
    print(json.dumps(report, indent=2))
    print(f"\nDatabase: {args.db}\nNext: python -m tracknet serve --db {args.db} --data {args.data}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    from .detector import MotionDetector, YoloDetector
    from .pipeline import process_video
    detector = MotionDetector() if args.detector == "motion" else YoloDetector(args.weights, conf=args.conf)
    store = Store(args.db)

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"\r  frame {done}/{total}", end="", file=sys.stderr)

    res = process_video(args.video, args.camera, store, detector, images_dir=Path(args.data) / "crops",
                        stride=args.stride, progress=progress)
    print(f"\n{args.camera}: {len(res['tracks'])} tracks from {res['frames_processed']} frames "
          f"({res['processing_fps']:.1f} frames/s)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api import create_app
    app = create_app(args.db, args.data)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .demo import evaluate_pipeline, scalability_benchmark
    truth_path = Path(args.data) / "demo" / "ground_truth.json"
    if not truth_path.exists():
        print("No demo ground truth found. Run `python -m tracknet demo` first.", file=sys.stderr)
        return 1
    store = Store(args.db)
    report = evaluate_pipeline(store, json.loads(truth_path.read_text()), min_score=args.min_score)
    print("Matching quality on the synthetic demo:")
    print(json.dumps(report, indent=2))
    print("\nCandidate-search latency vs database size:")
    for row in scalability_benchmark():
        print(f"  {row['tracks']:>5} tracks -> {row['search_ms']:>7} ms per search")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tracknet", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--db", default="data/tracknet.db", help="SQLite database path")
        sp.add_argument("--data", default="data", help="folder for crops, demo videos and summaries")

    d = sub.add_parser("demo", help="generate synthetic multi-camera videos and run the full pipeline")
    common(d)
    d.add_argument("--vehicles", type=int, default=30)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=_cmd_demo)

    pr = sub.add_parser("process", help="detect and track vehicles in one video")
    common(pr)
    pr.add_argument("--video", required=True)
    pr.add_argument("--camera", required=True, help="camera id, e.g. c017")
    pr.add_argument("--detector", choices=["yolo", "motion"], default="yolo")
    pr.add_argument("--weights", default="yolov8n.pt")
    pr.add_argument("--conf", type=float, default=0.35)
    pr.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    pr.set_defaults(func=_cmd_process)

    s = sub.add_parser("serve", help="start the REST API")
    common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.set_defaults(func=_cmd_serve)

    e = sub.add_parser("evaluate", help="score cross-camera matching on the synthetic demo")
    common(e)
    e.add_argument("--min-score", type=float, default=0.55)
    e.set_defaults(func=_cmd_evaluate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
