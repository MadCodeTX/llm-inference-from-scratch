#!/usr/bin/env python3
"""Run the deterministic opt-harness demo or import a metrics artefact.

Usage:
    python scripts/opt_harness.py demo --work-dir /tmp/opt_demo --k 1
    python scripts/opt_harness.py import-metrics path/to/metrics.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.opt_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
