"""File-based pipeline orchestration and result contract."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from auth_anomaly.detection import evaluate, explain_anomalies, train_isolation_forest
from auth_anomaly.features import build_feature_matrix, engineer_features
from auth_anomaly.io import load_data
from auth_anomaly.schema import clean_data, infer_schema


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Explicit outputs for local interfaces that consume a file path."""

    cleaned_data: pd.DataFrame
    feature_data: pd.DataFrame
    feature_matrix: np.ndarray
    model: IsolationForest
    scores: np.ndarray
    predictions: np.ndarray
    report: Dict[str, Any]

    @property
    def scored_data(self) -> pd.DataFrame:
        """Return cleaned source rows with row-aligned scores and predictions."""
        return self.cleaned_data.assign(
            anomaly_score=self.scores,
            is_anomaly=self.predictions.astype(bool),
        )


def analyze_file(
    path: str | Path,
    schema_map: Optional[Dict[str, str]] = None,
    contamination: float = 0.02,
    sample_rows: Optional[int] = None,
) -> PipelineResult:
    """Run the full analysis from a local file path without writing a report to logs."""
    raw = load_data(path, sample_rows=sample_rows)
    schema = infer_schema(raw, schema_map=schema_map)
    clean_df, cleaning_report = clean_data(raw, schema)
    feature_df, engineering_report = engineer_features(clean_df, schema)
    matrix, matrix_report = build_feature_matrix(feature_df, schema, return_metadata=True)
    model, scores, predictions = train_isolation_forest(matrix, contamination=contamination)
    anomaly_explanations = explain_anomalies(feature_df, schema, scores, predictions)
    label = schema["roles"].get("attack_label")
    try:
        evaluation = evaluate(clean_df[label] if label else None, predictions, scores)
    except ValueError as exc:
        schema["warnings"].append(f"Etiqueta no evaluable: {exc}")
        evaluation = evaluate(None, predictions, scores)
    report = {
        "input_rows": len(raw),
        "clean_rows": len(clean_df),
        "schema_mapping": schema["roles"],
        "inferred_column_types": schema["column_types"],
        "warnings": schema["warnings"],
        "cleaning": cleaning_report,
        "features": engineering_report,
        "feature_matrix": matrix_report,
        "evaluation": evaluation,
        "anomaly_explanations": anomaly_explanations,
    }
    return PipelineResult(clean_df, feature_df, matrix, model, scores, predictions, report)


def _print_report(report: Dict[str, Any]) -> None:
    """Log the compact report used by the existing CLI."""
    summary = {
        "rows": {"input": report["input_rows"], "clean": report["clean_rows"]},
        "schema_mapping": report["schema_mapping"],
        "duplicates_removed": report["cleaning"]["duplicates_removed"],
        "generated_features": report["features"]["generated_features"],
        "omitted_features": report["features"]["omitted_features"],
        "matrix": {
            "n_features": len(report["feature_matrix"]["feature_names"]),
            "encoded": report["feature_matrix"]["encoded_columns"],
            "dropped": report["feature_matrix"]["dropped_columns"],
        },
        "evaluation": report["evaluation"],
        "anomaly_explanations": report["anomaly_explanations"],
        "warnings": report["warnings"],
    }
    LOGGER.info("Pipeline report:\n%s", json.dumps(summary, indent=2, default=str, ensure_ascii=False))


def run_pipeline(
    path: str | Path,
    schema_map: Optional[Dict[str, str]] = None,
    contamination: float = 0.02,
    sample_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray, IsolationForest, np.ndarray, Dict[str, Any]]:
    """Backward-compatible path API returning the original five-item tuple."""
    result = analyze_file(path, schema_map, contamination, sample_rows)
    _print_report(result.report)
    result.cleaned_data.attrs["pipeline_report"] = result.report
    result.cleaned_data.attrs["feature_dataframe"] = result.feature_data
    return (
        result.cleaned_data,
        result.feature_matrix,
        result.model,
        result.scores,
        result.report,
    )
