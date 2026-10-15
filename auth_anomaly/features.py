"""Authentication feature engineering and numeric matrix construction."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from auth_anomaly.schema import _role_score, _to_boolean


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
    protected.update(column for column in df.columns if _role_score(column, "attack_label") >= 50)
    numeric_parts, encoded, dropped = [], [], []
    for column in df.columns:
        if column.startswith("feature_"):
            numeric_parts.append(pd.to_numeric(df[column], errors="coerce").fillna(0).rename(column))
            continue
        kind = types.get(column, "identifier_or_free_text")
        if column in protected or kind in {"datetime", "identifier_or_free_text"}:
            dropped.append(column)
        elif kind in {"numeric", "boolean"}:
            numeric_parts.append(pd.to_numeric(df[column], errors="coerce").fillna(0).rename(column))
        elif kind == "categorical_low":
            dummies = pd.get_dummies(df[column].astype("string"), prefix=column, dtype=float)
            numeric_parts.extend(dummies[name] for name in dummies.columns)
            encoded.append({"column": column, "strategy": "one_hot", "output_columns": len(dummies.columns)})
        elif kind == "categorical_high":
            frequency = df[column].astype("string").value_counts(normalize=True, dropna=False)
            numeric_parts.append(
                df[column].astype("string").map(frequency).fillna(0).astype(float).rename(f"{column}__frequency")
            )
            encoded.append({"column": column, "strategy": "frequency", "output_columns": 1})
        else:
            dropped.append(column)
    if not numeric_parts:
        raise ValueError("No numeric features remain. Add useful columns or provide a schema_map.")
    feature_df = pd.concat(numeric_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = RobustScaler()
    matrix = scaler.fit_transform(feature_df).astype(np.float64)
    metadata = {
        "feature_names": feature_df.columns.tolist(),
        "encoded_columns": encoded,
        "dropped_columns": dropped,
        "scaler": "RobustScaler",
    }
    return (matrix, metadata) if return_metadata else matrix
