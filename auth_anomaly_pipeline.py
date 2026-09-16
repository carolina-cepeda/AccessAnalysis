"""Schema-agnostic authentication anomaly detection with Isolation Forest."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler


LOGGER = logging.getLogger(__name__)
ROLE_KEYS = (
    "timestamp", "user_id", "ip", "country", "region", "city", "asn",
    "user_agent", "login_result", "attack_label", "rtt",
)
BOOLEAN_VALUES = {
    "true": True, "false": False, "yes": True, "no": False,
    "y": True, "n": False, "1": True, "0": False,
    "success": True, "successful": True, "failed": False, "failure": False,
}
ROLE_HINTS = {
    "timestamp": ("timestamp", "time", "date", "datetime", "fecha", "hora"),
    "user_id": ("user", "usuario", "account", "cuenta", "member", "identity", "id"),
    "ip": ("ip", "ipaddress", "ip_address", "source_address", "origen"),
    "country": ("country", "pais", "nation"),
    "region": ("region", "state", "province", "provincia"),
    "city": ("city", "ciudad", "town"),
    "asn": ("asn", "autonomous_system", "autonomous"),
    "user_agent": ("user_agent", "useragent", "agent", "browser", "navegador"),
    "login_result": ("login_success", "successful", "success", "resultado", "result", "outcome", "status"),
    "attack_label": ("account_takeover", "takeover", "attack", "fraud", "malicious", "label", "target"),
    "rtt": ("round_trip", "rtt", "latency", "response_time", "duration_ms"),
}


def normalize_column_name(name: Any) -> str:
    """Return a stable, ASCII-only name so mappings work across sources."""
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value or "unnamed_column"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    names = [normalize_column_name(c) for c in result.columns]
    duplicated = pd.Index(names)[pd.Index(names).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Two or more columns have the same name after normalization: "
            f"{duplicated}. Rename them or use unambiguous column names."
        )
    result.columns = names
    return result


def load_data(path: str | Path, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV, Parquet or JSON. ``sample_rows`` makes large CSV validation safe."""
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
    except Exception as exc:  # Add format/path context to pandas' error.
        raise ValueError(f"Could not read {file_path} as {suffix}: {exc}") from exc
    raise ValueError("Unsupported format. Use CSV, Parquet, JSON, JSONL, or NDJSON.")


def _string_sample(series: pd.Series, limit: int = 2_000) -> pd.Series:
    return series.dropna().astype("string").head(limit).str.strip().str.lower()


def _looks_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    sample = _string_sample(series)
    return bool(len(sample) and set(sample.unique()).issubset(BOOLEAN_VALUES))


def _looks_datetime(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Parsing arbitrary IDs as dates is unsafe; require a date-like name or values.
    sample = _string_sample(series)
    if sample.empty or pd.api.types.is_numeric_dtype(series):
        return False
    name_hint = any(token in name for token in ROLE_HINTS["timestamp"])
    value_hint = sample.str.contains(r"[-/:T]|\bjan\b|\bfeb\b", regex=True).mean() > 0.7
    if not (name_hint or value_hint):
        return False
    # Mixed timestamp formats are common in exported logs.  Pandas may warn
    # while inferring them, but the conversion ratio below is the decision rule.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return parsed.notna().mean() >= 0.8


def _classify_column(series: pd.Series, name: str) -> str:
    """Classify without relying on a particular dataset's column vocabulary."""
    n = max(len(series), 1)
    non_null = series.dropna()
    unique = non_null.nunique(dropna=True)
    unique_ratio = unique / max(len(non_null), 1)
    if _looks_boolean(series):
        return "boolean"
    if _looks_datetime(series, name):
        return "datetime"
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if not is_numeric and not non_null.empty:
        numeric_ratio = pd.to_numeric(_string_sample(series), errors="coerce").notna().mean()
        is_numeric = numeric_ratio >= 0.98
    id_name = bool(re.search(r"(^|_)(id|uuid|guid|index|identifier)($|_)", name))
    if id_name and unique_ratio > 0.5:
        return "identifier_or_free_text"
    if is_numeric:
        return "numeric"
    sample = _string_sample(series)
    avg_length = float(sample.str.len().mean()) if not sample.empty else 0.0
    # Long unstructured comments/requests are discarded. IPs and user agents are
    # changed to categorical_high later because their frequency is informative.
    if (unique_ratio > 0.9 and avg_length > 80) or (unique_ratio > 0.98 and avg_length > 35):
        return "identifier_or_free_text"
    # An absolute ceiling prevents a source with millions of rows and thousands
    # of values (browser versions, cities, etc.) from creating an enormous
    # one-hot matrix merely because its unique ratio is small.
    low_cardinality_limit = min(100, max(20, int(0.01 * n)))
    return "categorical_low" if unique <= low_cardinality_limit else "categorical_high"


def _role_score(column: str, role: str) -> int:
    """Name-only, approximate semantic matching; explicit maps always win."""
    score = 0
    for hint in ROLE_HINTS[role]:
        if column == hint:
            score = max(score, 100)
        elif column.startswith(hint + "_") or column.endswith("_" + hint):
            score = max(score, 80)
        elif hint in column:
            score = max(score, 50)
    # Avoid declaring every generic *_id column a user identifier.
    if role == "user_id" and column in {"id", "index", "event_id", "session_id"}:
        score -= 45
    return score


def infer_schema(df: pd.DataFrame, schema_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Infer column kinds and semantic roles, applying optional user overrides."""
    if df.empty:
        raise ValueError("The dataset is empty; its schema cannot be inferred.")
    work = _normalise_columns(df)
    types = {col: _classify_column(work[col], col) for col in work.columns}
    roles: Dict[str, Optional[str]] = {role: None for role in ROLE_KEYS}
    warnings = []
    explicit = schema_map or {}
    unknown_roles = set(explicit).difference(ROLE_KEYS)
    if unknown_roles:
        warnings.append(f"Ignored schema_map keys: {sorted(unknown_roles)}")
    for role in ROLE_KEYS:
        supplied = explicit.get(role)
        if supplied is not None:
            normalised = normalize_column_name(supplied)
            if normalised not in work.columns:
                raise ValueError(f"schema_map['{role}'] points to missing column '{supplied}'.")
            roles[role] = normalised
    for role in ROLE_KEYS:
        if roles[role] is not None:
            continue
        candidates = sorted(
            (( _role_score(col, role), col) for col in work.columns), reverse=True
        )
        if candidates and candidates[0][0] >= 50:
            roles[role] = candidates[0][1]
            warnings.append(f"Inferred mapping: {role} -> {candidates[0][1]}")
    # A full UA and IP must use cardinality reduction even when they look like free text.
    for role in ("ip", "user_agent"):
        if roles[role]:
            types[roles[role]] = "categorical_high"
    # A mapped user ID remains available for historical features, never one-hot encoded.
    if roles["user_id"]:
        types[roles["user_id"]] = "identifier_or_free_text"
    if not roles["timestamp"]:
        warnings.append("No timestamp was inferred: temporal and history features will be skipped.")
    if not roles["user_id"]:
        warnings.append("No user_id was inferred: user-history features will be skipped.")
    schema = {"column_types": types, "roles": roles, "warnings": warnings}
    return schema


def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    return text.map(BOOLEAN_VALUES).astype("boolean")


def clean_data(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize values, remove exact duplicates and record every imputation."""
    clean = _normalise_columns(df)
    expected = set(schema["column_types"])
    absent = expected.difference(clean.columns)
    if absent:
        raise ValueError(f"The DataFrame is missing columns expected by the schema: {sorted(absent)}")
    before = len(clean)
    clean = clean.drop_duplicates().copy()
    report = {"duplicates_removed": before - len(clean), "imputations": {}, "timestamp_validation": {}}
    now = pd.Timestamp.now(tz="UTC")
    for col, kind in schema["column_types"].items():
        missing_before = int(clean[col].isna().sum())
        if kind == "datetime":
            values = pd.to_datetime(clean[col], errors="coerce", utc=True)
            # Authentication records outside this broad interval are usually parse errors.
            implausible = values.notna() & ((values < pd.Timestamp("1990-01-01", tz="UTC")) | (values > now + pd.Timedelta(days=365)))
            values.loc[implausible] = pd.NaT
            valid = values.dropna()
            fill = valid.median() if not valid.empty else pd.Timestamp("2000-01-01", tz="UTC")
            missing_count = int(values.isna().sum())
            clean[col] = values.fillna(fill)
            report["timestamp_validation"][col] = {"invalid_range_set_missing": int(implausible.sum())}
            report["imputations"][col] = {"strategy": "median_timestamp", "count": missing_count}
        elif kind == "numeric":
            values = pd.to_numeric(clean[col], errors="coerce")
            fill = values.median() if values.notna().any() else 0.0
            clean[col] = values.fillna(fill).astype(float)
            report["imputations"][col] = {"strategy": "median", "value": float(fill), "count": int(values.isna().sum())}
        elif kind == "boolean":
            values = _to_boolean(clean[col])
            mode = values.dropna().mode()
            fill = bool(mode.iloc[0]) if not mode.empty else False
            clean[col] = values.fillna(fill).astype(bool)
            report["imputations"][col] = {"strategy": "mode_boolean", "value": fill, "count": int(values.isna().sum())}
        else:
            # Do not impose a vocabulary: only trim/case-fold and preserve values.
            values = clean[col].astype("string").str.strip().str.lower().replace("", pd.NA)
            clean[col] = values.fillna("unknown")
            report["imputations"][col] = {"strategy": "explicit_unknown", "count": int(values.isna().sum())}
    clean.attrs["schema"] = schema
    return clean, report


def _ordered_frame(df: pd.DataFrame, timestamp: Optional[str]) -> pd.DataFrame:
    """Return chronological order while retaining the original index for assignment."""
    return df.sort_values(timestamp, kind="stable") if timestamp else df


def engineer_features(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate authentication features only when their required roles are present."""
    out = df.copy()
    roles = schema["roles"]
    generated, omitted = [], []

    def add(name: str, values: pd.Series) -> None:
        out[name] = values.reindex(out.index).astype(float)
        generated.append(name)

    timestamp, user = roles["timestamp"], roles["user_id"]
    if timestamp:
        add("feature_login_hour", out[timestamp].dt.hour)
        add("feature_login_day_of_week", out[timestamp].dt.dayofweek)
    else:
        omitted.extend(["feature_login_hour (missing timestamp)", "feature_login_day_of_week (missing timestamp)"])
    if user:
        add("feature_user_login_frequency", out.groupby(user, dropna=False)[user].transform("size"))
    else:
        omitted.append("feature_user_login_frequency (missing user_id)")

    chronological = _ordered_frame(out, timestamp)
    if user and timestamp:
        position = chronological.groupby(user, dropna=False).cumcount()
        for role, feature in (("country", "feature_country_changed"), ("city", "feature_city_changed")):
            column = roles[role]
            if column:
                previous = chronological.groupby(user, dropna=False)[column].shift()
                changed = ((chronological[column] != previous) & previous.notna()).astype(float)
                add(feature, changed)
            else:
                omitted.append(f"{feature} (missing {role})")
        for role, feature in (("ip", "feature_new_ip_for_user"), ("asn", "feature_new_asn_for_user")):
            column = roles[role]
            if column:
                first_seen = chronological.groupby([user, column], dropna=False).cumcount().eq(0).astype(float)
                # The first record of each user has no historical baseline.
                first_seen.loc[position.eq(0)] = 0.0
                add(feature, first_seen)
            else:
                omitted.append(f"{feature} (missing {role})")
        result = roles["login_result"]
        if result:
            successful = _to_boolean(chronological[result]).fillna(False).astype(int)
            failures = 1 - successful
            prior_failures = failures.groupby(chronological[user], dropna=False).cumsum() - failures
            add("feature_user_prior_failure_rate", prior_failures / position.replace(0, np.nan).fillna(1))
        else:
            omitted.append("feature_user_prior_failure_rate (missing login_result)")
    else:
        omitted.extend([
            "history features (require user_id and timestamp)",
            "feature_user_prior_failure_rate (requires user_id, timestamp, and login_result)",
        ])
    if roles["rtt"]:
        rtt = pd.to_numeric(out[roles["rtt"]], errors="coerce").clip(lower=0).fillna(0)
        add("feature_log_rtt", np.log1p(rtt))
    else:
        omitted.append("feature_log_rtt (missing rtt)")
    out.attrs["schema"] = schema
    return out, {"generated_features": generated, "omitted_features": omitted}


def build_feature_matrix(
    df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None, return_metadata: bool = False
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """Return a scaled, entirely numeric matrix; high-cardinality values use frequency encoding."""
    schema = schema or df.attrs.get("schema")
    if not schema:
        raise ValueError("A schema is required to build the feature matrix.")
    types, roles = schema["column_types"], schema["roles"]
    protected = {value for key, value in roles.items() if value and key in {"timestamp", "user_id", "attack_label"}}
    # Never train on a likely target just because a source includes two labels
    protected.update(col for col in df.columns if _role_score(col, "attack_label") >= 50)
    numeric_parts, encoded, dropped = [], [], []
    for col in df.columns:
        if col.startswith("feature_"):
            numeric_parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0).rename(col))
            continue
        kind = types.get(col, "identifier_or_free_text")
        if col in protected or kind in {"datetime", "identifier_or_free_text"}:
            dropped.append(col)
        elif kind in {"numeric", "boolean"}:
            numeric_parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0).rename(col))
        elif kind == "categorical_low":
            dummies = pd.get_dummies(df[col].astype("string"), prefix=col, dtype=float)
            numeric_parts.extend(dummies[name] for name in dummies.columns)
            encoded.append({"column": col, "strategy": "one_hot", "output_columns": len(dummies.columns)})
        elif kind == "categorical_high":
            frequency = df[col].astype("string").value_counts(normalize=True, dropna=False)
            numeric_parts.append(df[col].astype("string").map(frequency).fillna(0).astype(float).rename(f"{col}__frequency"))
            encoded.append({"column": col, "strategy": "frequency", "output_columns": 1})
        else:
            dropped.append(col)
    if not numeric_parts:
        raise ValueError("No numeric features remain. Add useful columns or provide a schema_map.")
    feature_df = pd.concat(numeric_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = RobustScaler()
    X = scaler.fit_transform(feature_df).astype(np.float64)
    metadata = {"feature_names": feature_df.columns.tolist(), "encoded_columns": encoded, "dropped_columns": dropped, "scaler": "RobustScaler"}
    return (X, metadata) if return_metadata else X


def train_isolation_forest(
    X: np.ndarray, contamination: float = 0.02, random_state: int = 42
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Fit Isolation Forest and return model, high-is-anomalous scores, and 0/1 predictions."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination debe estar entre 0 y 0.5.")
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X)
    scores = -model.score_samples(X) 
    predictions = (model.predict(X) == -1).astype(int)
    return model, scores, predictions


def _binary_labels(values: Iterable[Any]) -> np.ndarray:
    series = pd.Series(values)
    boolean = _to_boolean(series)
    if boolean.notna().mean() >= 0.8:
        return boolean.fillna(False).astype(int).to_numpy()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return (numeric.fillna(0) > 0).astype(int).to_numpy()
    # A non-empty label with two arbitrary categories: least frequent is positive.
    normal = series.astype("string").str.strip().str.lower()
    counts = normal.value_counts()
    if len(counts) == 2:
        return (normal == counts.idxmin()).astype(int).to_numpy()
    raise ValueError("The reference label cannot be reliably converted to binary values.")


def evaluate(
    y_true: Optional[Iterable[Any]], y_pred: np.ndarray, scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Report supervised metrics if labels exist, otherwise descriptive unsupervised metrics."""
    if y_true is None:
        result: Dict[str, Any] = {"evaluation_type": "unsupervised", "anomaly_rate": float(np.mean(y_pred))}
        if scores is not None:
            result["score_quantiles"] = {str(q): float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.95, 0.99)}
        return result
    truth = _binary_labels(y_true)
    if len(truth) != len(y_pred):
        raise ValueError("y_true y y_pred tienen longitudes distintas.")
    negatives = max(int((truth == 0).sum()), 1)
    false_positives = int(((truth == 0) & (y_pred == 1)).sum())
    return {
        "evaluation_type": "supervised",
        "precision": float(precision_score(truth, y_pred, zero_division=0)),
        "recall": float(recall_score(truth, y_pred, zero_division=0)),
        "f1": float(f1_score(truth, y_pred, zero_division=0)),
        "false_positive_rate": false_positives / negatives,
        "support_positive": int(truth.sum()),
        "anomaly_rate": float(np.mean(y_pred)),
    }


def explain_anomalies(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    scores: np.ndarray,
    predictions: np.ndarray,
    max_examples: int = 10,
) -> Dict[str, Any]:
    """Return plain-language, rule-based context for the highest-scoring alerts.
    """
    if len(df) != len(scores) or len(df) != len(predictions):
        raise ValueError("df, scores, and predictions must have the same length.")

    roles = schema["roles"]
    flagged_positions = np.flatnonzero(predictions == 1)
    ranked_positions = flagged_positions[np.argsort(scores[flagged_positions])[::-1]]
    rtt_cutoff: Optional[float] = None
    if "feature_log_rtt" in df.columns:
        rtt_cutoff = float(df["feature_log_rtt"].quantile(0.95))

    examples = []
    for position in ranked_positions[:max_examples]:
        row = df.iloc[position]
        reasons = []
        if row.get("feature_new_ip_for_user", 0) >= 1:
            reasons.append("New IP address for this user")
        if row.get("feature_new_asn_for_user", 0) >= 1:
            reasons.append("New network provider (ASN) for this user")
        if row.get("feature_country_changed", 0) >= 1:
            reasons.append("Country changed since this user's previous login")
        if row.get("feature_city_changed", 0) >= 1:
            reasons.append("City changed since this user's previous login")
        prior_failure_rate = float(row.get("feature_user_prior_failure_rate", 0))
        if prior_failure_rate >= 0.25:
            reasons.append(f"Elevated prior failed-login rate for this user ({prior_failure_rate:.0%})")
        if rtt_cutoff is not None and float(row["feature_log_rtt"]) >= rtt_cutoff:
            reasons.append("Connection latency is among the highest 5% of observed logins")
        if not reasons:
            reasons.append("Unusual combination of modeled values; no single rule-based security signal stood out")

        example: Dict[str, Any] = {
            "row_index": str(df.index[position]),
            "anomaly_score": float(scores[position]),
            "reasons": reasons,
        }
        if roles["timestamp"]:
            example["timestamp"] = str(row[roles["timestamp"]])
        if roles["user_id"]:
            example["user_id"] = str(row[roles["user_id"]])
        examples.append(example)

    return {
        "anomalies_found": int(len(flagged_positions)),
        "examples_shown": int(len(examples)),
        "note": "Reasons are rule-based investigation cues, not proof that a login is malicious.",
        "examples": examples,
    }


def _print_report(report: Dict[str, Any]) -> None:
    """Log a compact JSON report. """
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
) -> Tuple[pd.DataFrame, np.ndarray, IsolationForest, np.ndarray, Dict[str, Any]]:
    """Run load → infer → clean → engineer → encode → train → evaluate end 2 end."""
    raw = load_data(path, sample_rows=sample_rows)
    schema = infer_schema(raw, schema_map=schema_map)
    clean_df, cleaning_report = clean_data(raw, schema)
    feature_df, engineering_report = engineer_features(clean_df, schema)
    X, matrix_report = build_feature_matrix(feature_df, schema, return_metadata=True)
    model, scores, predictions = train_isolation_forest(X, contamination=contamination)
    anomaly_explanations = explain_anomalies(feature_df, schema, scores, predictions)
    label = schema["roles"].get("attack_label")
    try:
        evaluation = evaluate(clean_df[label] if label else None, predictions, scores)
    except ValueError as exc:
        schema["warnings"].append(f"Etiqueta no evaluable: {exc}")
        evaluation = evaluate(None, predictions, scores)
    report = {
        "input_rows": len(raw), "clean_rows": len(clean_df), "schema_mapping": schema["roles"],
        "inferred_column_types": schema["column_types"], "warnings": schema["warnings"],
        "cleaning": cleaning_report, "features": engineering_report,
        "feature_matrix": matrix_report, "evaluation": evaluation,
        "anomaly_explanations": anomaly_explanations,
    }
    _print_report(report)
    clean_df.attrs["pipeline_report"] = report
    clean_df.attrs["feature_dataframe"] = feature_df
    return clean_df, X, model, scores, report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Authentication-log CSV, Parquet, JSON, JSONL, or NDJSON")
    parser.add_argument("--schema-map", help="JSON mapping semantic roles to column names")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--sample-rows", type=int, help="Limits rows; useful for large files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mapping = json.loads(args.schema_map) if args.schema_map else None
    clean_df, X, _, scores, _ = run_pipeline(args.path, mapping, args.contamination, args.sample_rows)
    print(f"OK: {len(clean_df)} clean rows, X={X.shape}, mean score={scores.mean():.4f}")


if __name__ == "__main__":
    _main()
