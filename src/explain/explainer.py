"""SHAP explainability explainer module for anomaly detection features."""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest


def format_reason(feature_name: str, val: float) -> str:
    """Formats a feature name and value into a human-readable reason string."""
    labels = {
        "degree_centrality": "degree centrality",
        "co_spend_cluster_size": "co-spend cluster size",
        "unique_ips": "unique IP count",
        "tor_broadcast_ratio": "Tor exit node broadcast ratio",
        "ip_switching_frequency": "IP switching frequency",
        "tx_count": "transaction count",
        "total_volume_btc": "total transaction volume",
        "avg_tx_amount": "average transaction amount",
        "max_tx_amount": "maximum transaction amount",
        "std_tx_amount": "transaction amount volatility",
        "burst_ratio": "high-frequency burst ratio",
        "peeling_chain_score": "peeling chain pattern score",
    }
    label = labels.get(feature_name, feature_name.replace("_", " "))

    # Format values based on column semantics
    if "ratio" in feature_name or "frequency" in feature_name or "score" in feature_name:
        pct = val * 100.0
        return f"High {label} ({pct:.1f}%)" if val > 0.2 else f"Low {label} ({pct:.1f}%)"
    elif "volume" in feature_name or "amount" in feature_name:
        return f"Unusual {label} ({val:.4f} BTC)"
    elif "ips" in feature_name or "count" in feature_name or "size" in feature_name or "centrality" in feature_name:
        # Check if integer casting is safe
        try:
            val_int = int(round(val))
        except (ValueError, TypeError):
            val_int = 0
        return f"High {label} ({val_int})" if val > 2 else f"Low {label} ({val_int})"
    else:
        return f"Unusual {label} ({val:.4f})"


from src.detection.eif import ExtendedIsolationForest


def explain_anomalies(
    model: object,
    feature_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Computes SHAP values and attaches human-readable reasons to each wallet.

    For each entity, it runs either TreeSHAP (for standard models) or KernelSHAP
    (for Extended Isolation Forest models), selects the top_n features by absolute
    magnitude, and constructs a semicolon-separated string of explanations.

    Args:
        model: Trained Isolation Forest or Extended Isolation Forest model.
        feature_df: Feature DataFrame indexed by wallet address (used for training).
        top_n: Number of top contributing features to include in explanations.

    Returns:
        pd.DataFrame: Scored DataFrame with an additional 'top_reasons' column.
    """
    if feature_df.empty:
        df_out = feature_df.copy()
        df_out["top_reasons"] = []
        return df_out

    # Make copy
    df_out = feature_df.copy()

    # Pre-clean inputs for SHAP (drop output metadata columns added during scoring)
    extra_cols = ["anomaly_score", "anomaly_confidence", "attribution_confidence", "top_reasons"]
    X = feature_df.drop(columns=[col for col in extra_cols if col in feature_df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Compute SHAP values based on model type
    if isinstance(model, ExtendedIsolationForest):
        # EIH uses KernelSHAP for model-agnostic explanation.
        # Summarize feature space with 5 representative k-means clusters.
        background = shap.kmeans(X, 5)
        explainer = shap.KernelExplainer(model.decision_function, background)
        
        # Optimization: Only explain the top 100 highest scored anomalies (minimum score of 0.45)
        # to ensure fast execution times under offline demo conditions.
        if "anomaly_score" in feature_df.columns:
            scores_sorted = feature_df["anomaly_score"].sort_values(ascending=False)
            if len(scores_sorted) > 100:
                threshold = max(0.45, scores_sorted.iloc[99])
            else:
                threshold = 0.45
            explain_mask = feature_df["anomaly_score"] >= threshold
        else:
            explain_mask = pd.Series(True, index=feature_df.index)
            
        X_explain = X[explain_mask]
        
        if not X_explain.empty:
            shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 0]
        else:
            shap_values = np.zeros((0, X.shape[1]))
    else:
        # Standard Isolation Forest uses native TreeSHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]
        explain_mask = pd.Series(True, index=feature_df.index)

    reasons = []
    shap_idx = 0
    for idx in range(len(X)):
        if explain_mask.iloc[idx]:
            row_features = X.iloc[idx]
            row_shap = shap_values[shap_idx]
            shap_idx += 1

            # Get feature indices sorted by absolute SHAP contribution (descending)
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]

            row_reasons = []
            for f_idx in top_indices:
                feature_name = X.columns[f_idx]
                val = row_features.iloc[f_idx]
                row_reasons.append(format_reason(feature_name, val))

            reasons.append("; ".join(row_reasons))
        else:
            reasons.append("Normal behavior (no anomaly flagged)")

    df_out["top_reasons"] = reasons
    return df_out
