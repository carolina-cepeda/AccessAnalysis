"""Compatibility entry point for the local authentication anomaly pipeline."""

from auth_anomaly.cli import main as _main
from auth_anomaly.detection import (
    _binary_labels,
    evaluate,
    explain_anomalies,
    train_isolation_forest,
)
from auth_anomaly.features import (
    _ordered_frame,
    build_feature_matrix,
    engineer_features,
)
from auth_anomaly.io import load_data
from auth_anomaly.pipeline import PipelineResult, _print_report, analyze_file, run_pipeline
from auth_anomaly.schema import (
    BOOLEAN_VALUES,
    ROLE_HINTS,
    ROLE_KEYS,
    _classify_column,
    _looks_boolean,
    _looks_datetime,
    _normalise_columns,
    _parse_datetime,
    _role_score,
    _string_sample,
    _to_boolean,
    clean_data,
    infer_schema,
    normalize_column_name,
)


if __name__ == "__main__":
    _main()
