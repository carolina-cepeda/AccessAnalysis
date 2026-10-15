from pathlib import Path

import pandas as pd

from auth_anomaly.features import engineer_features
from auth_anomaly.pipeline import analyze_file, run_pipeline
from auth_anomaly.schema import clean_data, infer_schema


FIXTURE = Path(__file__).parent / "fixtures" / "auth_logs.csv"


def test_analyze_file_returns_row_aligned_results_and_excludes_label():
    result = analyze_file(
        FIXTURE,
        schema_map={"attack_label": "Is Account Takeover"},
        contamination=0.2,
    )

    scored = result.scored_data
    assert len(scored) == result.report["clean_rows"]
    assert len(result.scores) == len(result.predictions) == len(scored)
    assert {"anomaly_score", "is_anomaly"}.issubset(scored.columns)
    assert "is_account_takeover" in result.report["feature_matrix"]["dropped_columns"]
    assert result.report["evaluation"]["evaluation_type"] == "supervised"


def test_run_pipeline_preserves_legacy_tuple_contract():
    result = run_pipeline(
        FIXTURE,
        schema_map={"attack_label": "Is Account Takeover"},
        contamination=0.2,
    )

    assert len(result) == 5
    clean_data, matrix, model, scores, report = result
    assert len(clean_data) == report["clean_rows"]
    assert matrix.shape[0] == len(scores) == len(clean_data)
    assert model is not None


def test_numeric_epoch_timestamp_is_converted_before_feature_engineering():
    raw = pd.DataFrame(
        {
            "User ID": ["u1", "u1", "u2"],
            "Login Timestamp": [1704067200000, 1704070800000, 1704074400000],
            "IP Address": ["192.0.2.1", "192.0.2.2", "192.0.2.3"],
            "Login Successful": [True, False, True],
        }
    )
    schema = infer_schema(raw)
    cleaned, cleaning_report = clean_data(raw, schema)
    featured, feature_report = engineer_features(cleaned, schema)

    assert schema["column_types"]["login_timestamp"] == "datetime"
    assert cleaning_report["timestamp_validation"]["login_timestamp"]["numeric_epoch_unit"] == "ms"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["login_timestamp"])
    assert "feature_login_hour" in featured.columns
    assert "feature_login_hour" in feature_report["generated_features"]
