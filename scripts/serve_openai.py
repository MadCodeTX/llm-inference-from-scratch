#!/usr/bin/env python3
"""CPU fake-backend OpenAI server (contract smoke, no weights).

  python scripts/serve_openai.py --port 8000
Then: python benchmarks/run_bench.py --smoke --base-url http://127.0.0.1:8000 --model fake-model
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.openai_server import FakeBackend, serve_forever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fake-model")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    a = ap.parse_args()
    serve_forever(FakeBackend(model_id=a.model), a.host, a.port, a.request_timeout)
