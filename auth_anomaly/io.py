"""Input loading for authentication log files."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_data(path: str | Path, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV, Parquet or JSON from a local path."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(file_path)
            return frame.head(sample_rows) if sample_rows else frame
        if suffix in {".json", ".jsonl", ".ndjson"}:
            lines = suffix in {".jsonl", ".ndjson"}
            if lines:
                return pd.read_json(file_path, lines=True, nrows=sample_rows)
            frame = pd.read_json(file_path)
            return frame.head(sample_rows) if sample_rows else frame
    except Exception as exc:
        raise ValueError(f"Could not read {file_path} as {suffix}: {exc}") from exc
    raise ValueError("Unsupported format. Use CSV, Parquet, JSON, JSONL, or NDJSON.")
