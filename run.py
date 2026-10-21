#!/usr/bin/env python3
"""
ChiralMeta — Single-Command Launcher
=====================================
Usage:
    python run.py [--port 8000] [--skip-cleaning] [--host 0.0.0.0]

Steps:
  1. Run data cleaning pipeline  (generates data/cleaned/)
  2. Start FastAPI server on port 8000
"""

import sys
import argparse
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ChiralMeta.Launcher")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="ChiralMeta Launcher")
    parser.add_argument("--port",           type=int,  default=8000)
    parser.add_argument("--host",           type=str,  default="0.0.0.0")
    parser.add_argument("--skip-cleaning",  action="store_true",
                        help="Skip data cleaning (use existing cleaned data)")
    parser.add_argument("--reload",         action="store_true",
                        help="Enable FastAPI hot-reload (development mode)")
    args = parser.parse_args()

    # ── Step 1: Data cleaning ─────────────────────────────────────────────
    if not args.skip_cleaning:
        log.info("=" * 60)
        log.info("STEP 1: Data Cleaning Pipeline (8 Stages)")
        log.info("=" * 60)
        try:
            from src.data_cleaning import run_cleaning_pipeline
            run_cleaning_pipeline()
        except Exception as e:
            log.error(f"Data cleaning failed: {e}")
            log.warning("Continuing with existing cleaned data (if any)...")
    else:
        log.info("Skipping data cleaning (--skip-cleaning flag set)")
        cleaned_csv = ROOT / "data" / "cleaned" / "cleaned_reactions_combined.csv"
        if not cleaned_csv.exists():
            log.warning("Cleaned data not found! Running cleaning anyway...")
            from src.data_cleaning import run_cleaning_pipeline
            run_cleaning_pipeline()

    # ── Step 2: Start FastAPI server ─────────────────────────────────────
    log.info("=" * 60)
    log.info(f"STEP 2: Starting ChiralMeta API on http://{args.host}:{args.port}")
    log.info("=" * 60)
    log.info(f"  → Open your browser at: http://localhost:{args.port}")
    log.info(f"  → API docs at:           http://localhost:{args.port}/docs")
    log.info("  → Press Ctrl+C to stop")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
