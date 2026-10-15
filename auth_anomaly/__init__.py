"""Core components for local authentication anomaly analysis."""

from auth_anomaly.detection import evaluate, explain_anomalies, train_isolation_forest
from auth_anomaly.features import build_feature_matrix, engineer_features
from auth_anomaly.io import load_data
from auth_anomaly.pipeline import PipelineResult, analyze_file, run_pipeline
from auth_anomaly.schema import clean_data, infer_schema, normalize_column_name

__all__ = [
    "PipelineResult",
    "analyze_file",
    "build_feature_matrix",
    "clean_data",
    "engineer_features",
    "evaluate",
    "explain_anomalies",
    "infer_schema",
    "load_data",
    "normalize_column_name",
    "run_pipeline",
    "train_isolation_forest",
]
