"""Column normalization, semantic schema inference and data cleaning."""

import re
import unicodedata
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd


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
    names = [normalize_column_name(column) for column in result.columns]
    duplicated = pd.Index(names)[pd.Index(names).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Two or more columns have the same name after normalization: "
            f"{duplicated}. Rename them or use unambiguous column names."
        )
    result.columns = names
    return result


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
    if (unique_ratio > 0.9 and avg_length > 80) or (unique_ratio > 0.98 and avg_length > 35):
        return "identifier_or_free_text"
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
    if role == "user_id" and column in {"id", "index", "event_id", "session_id"}:
        score -= 45
    return score


def infer_schema(df: pd.DataFrame, schema_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Infer column kinds and semantic roles, applying optional user overrides."""
    if df.empty:
        raise ValueError("The dataset is empty; its schema cannot be inferred.")
    work = _normalise_columns(df)
    types = {column: _classify_column(work[column], column) for column in work.columns}
    roles: Dict[str, Optional[str]] = {role: None for role in ROLE_KEYS}
    warnings_list = []
    explicit = schema_map or {}
    unknown_roles = set(explicit).difference(ROLE_KEYS)
    if unknown_roles:
        warnings_list.append(f"Ignored schema_map keys: {sorted(unknown_roles)}")
    for role in ROLE_KEYS:
        supplied = explicit.get(role)
        if supplied is not None:
            normalized = normalize_column_name(supplied)
            if normalized not in work.columns:
                raise ValueError(f"schema_map['{role}'] points to missing column '{supplied}'.")
            roles[role] = normalized
    for role in ROLE_KEYS:
        if roles[role] is not None:
            continue
        candidates = sorted(((_role_score(column, role), column) for column in work.columns), reverse=True)
        if candidates and candidates[0][0] >= 50:
            roles[role] = candidates[0][1]
            warnings_list.append(f"Inferred mapping: {role} -> {candidates[0][1]}")
    if roles["timestamp"] and types[roles["timestamp"]] == "numeric":
        types[roles["timestamp"]] = "datetime"
    for role in ("ip", "user_agent"):
        if roles[role]:
            types[roles[role]] = "categorical_high"
    if roles["user_id"]:
        types[roles["user_id"]] = "identifier_or_free_text"
    if not roles["timestamp"]:
        warnings_list.append("No timestamp was inferred: temporal and history features will be skipped.")
    if not roles["user_id"]:
        warnings_list.append("No user_id was inferred: user-history features will be skipped.")
    return {"column_types": types, "roles": roles, "warnings": warnings_list}


def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    return text.map(BOOLEAN_VALUES).astype("boolean")


def _parse_datetime(series: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    if not pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True), None

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.to_datetime(numeric, errors="coerce", utc=True), None
    magnitude = float(valid.abs().median())
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    elif magnitude >= 1e8:
        unit = "s"
    else:
        raise ValueError(
            "Numeric timestamp values do not look like Unix seconds, milliseconds, "
            "microseconds, or nanoseconds. Provide timestamps in a supported format."
        )
    return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True), unit


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
    for column, kind in schema["column_types"].items():
        if kind == "datetime":
            values, epoch_unit = _parse_datetime(clean[column])
            implausible = values.notna() & (
                (values < pd.Timestamp("1990-01-01", tz="UTC"))
                | (values > now + pd.Timedelta(days=365))
            )
            values.loc[implausible] = pd.NaT
            valid = values.dropna()
            fill = valid.median() if not valid.empty else pd.Timestamp("2000-01-01", tz="UTC")
            missing_count = int(values.isna().sum())
            clean[column] = values.fillna(fill)
            report["timestamp_validation"][column] = {
                "invalid_range_set_missing": int(implausible.sum()),
                "numeric_epoch_unit": epoch_unit,
            }
            report["imputations"][column] = {"strategy": "median_timestamp", "count": missing_count}
        elif kind == "numeric":
            values = pd.to_numeric(clean[column], errors="coerce")
            fill = values.median() if values.notna().any() else 0.0
            clean[column] = values.fillna(fill).astype(float)
            report["imputations"][column] = {
                "strategy": "median", "value": float(fill), "count": int(values.isna().sum())
            }
        elif kind == "boolean":
            values = _to_boolean(clean[column])
            mode = values.dropna().mode()
            fill = bool(mode.iloc[0]) if not mode.empty else False
            clean[column] = values.fillna(fill).astype(bool)
            report["imputations"][column] = {
                "strategy": "mode_boolean", "value": fill, "count": int(values.isna().sum())
            }
        else:
            values = clean[column].astype("string").str.strip().str.lower().replace("", pd.NA)
            clean[column] = values.fillna("unknown")
            report["imputations"][column] = {"strategy": "explicit_unknown", "count": int(values.isna().sum())}
    clean.attrs["schema"] = schema
    return clean, report
