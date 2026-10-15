"""Isolation Forest training, evaluation and rule-based alert context."""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from auth_anomaly.schema import _to_boolean


def train_isolation_forest(
    matrix: np.ndarray, contamination: float = 0.02, random_state: int = 42
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Fit Isolation Forest and return model, high-is-anomalous scores, and 0/1 predictions."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination debe estar entre 0 y 0.5.")
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(matrix)
    scores = -model.score_samples(matrix)
    predictions = (model.predict(matrix) == -1).astype(int)
    return model, scores, predictions


def _binary_labels(values: Iterable[Any]) -> np.ndarray:
    series = pd.Series(values)
    boolean = _to_boolean(series)
    if boolean.notna().mean() >= 0.8:
        return boolean.fillna(False).astype(int).to_numpy()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return (numeric.fillna(0) > 0).astype(int).to_numpy()
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
            result["score_quantiles"] = {
                str(quantile): float(np.quantile(scores, quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99)
            }
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
    """Return plain-language, rule-based context for the highest-scoring alerts."""
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
