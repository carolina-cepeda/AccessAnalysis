"""Command-line interface for local authentication log analysis."""

import argparse
import json
import logging

from auth_anomaly.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-agnostic authentication anomaly detection with Isolation Forest.")
    parser.add_argument("path", help="Authentication-log CSV, Parquet, JSON, JSONL, or NDJSON")
    parser.add_argument("--schema-map", help="JSON mapping semantic roles to column names")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--sample-rows", type=int, help="Limits rows; useful for large files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mapping = json.loads(args.schema_map) if args.schema_map else None
    clean_df, matrix, _, scores, _ = run_pipeline(
        args.path, mapping, args.contamination, args.sample_rows
    )
    print(f"OK: {len(clean_df)} clean rows, X={matrix.shape}, mean score={scores.mean():.4f}")
