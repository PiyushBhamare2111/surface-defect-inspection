"""Batch-inspect a folder of images and write a CSV inspection log.

Usage:  python scripts/inspect_folder.py samples --out reports/inspection_log.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from defect_inspect.config import DEFAULT_CRITICAL, DEFAULT_WEIGHTS  # noqa: E402
from defect_inspect.inference import Inspector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "inspection_log.csv")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--accept", type=float, default=0.20)
    ap.add_argument("--reject", type=float, default=0.80)
    ap.add_argument("--min-confidence", type=float, default=0.50)
    ap.add_argument("--critical", nargs="+", default=DEFAULT_CRITICAL)
    a = ap.parse_args()
    files = sorted(p for p in a.folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    log = Inspector(a.weights).inspect_batch(((p.name, p) for p in files), a.accept, a.reject,
                                             a.critical, a.min_confidence)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(a.out, index=False)
    print(log["verdict"].value_counts().to_string())
    print(f"Wrote {len(log)} rows to {a.out}")


if __name__ == "__main__":
    main()
