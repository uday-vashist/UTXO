"""End-to-end pipeline CLI runner for transaction ingestion, graph building,

feature engineering, anomaly detection, and SHAP explainability.
"""

import argparse
import os
import pickle
import sys

from src.ingestion.loader import load_csv
from src.graph.builder import build_graph
from src.detection.features import compute_features
from src.detection.model import train_and_score
from src.explain.explainer import explain_anomalies


def run_pipeline(
    input_path: str,
    out_path: str,
    graph_out_path: str,
    contamination: float = 0.05,
    model_type: str = "eif",
    seed: int = 42,
) -> None:
    """Runs the complete Bitcoin monitoring pipeline end-to-end.

    Args:
        input_path: Path to the raw telemetry transaction CSV.
        out_path: Path to write the ranked alerts CSV output.
        graph_out_path: Path to write the NetworkX graph pickle file.
        contamination: Proportion of anomalies to flag in the dataset.
        model_type: The model algorithm to use ('standard' or 'eif').
        seed: Random seed for model training.
    """
    print(f"[*] Starting BitSentinel pipeline...")
    print(f"[*] Input File: {input_path}")
    print(f"[*] Model Type: {model_type}, Contamination: {contamination}, Seed: {seed}")

    # 1. Ingest
    print("[1/5] Ingesting and validating transaction CSV...")
    try:
        df = load_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"    - Ingested {len(df)} telemetry records.")

    # 2. Build Graph
    print("[2/5] Constructing heterogeneous entity graph...")
    try:
        G = build_graph(df)
    except Exception as e:
        print(f"[ERROR] Graph construction failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"    - Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # 3. Feature Engineering
    print("[3/5] Computing wallet features...")
    try:
        feature_df = compute_features(G)
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"    - Engineered features for {len(feature_df)} wallets.")

    # 4. Anomaly Scoring
    print(f"[4/5] Training {model_type.upper()} model and scoring anomalies...")
    try:
        clf, scored_df = train_and_score(
            feature_df=feature_df,
            G=G,
            contamination=contamination,
            model_type=model_type,
            random_seed=seed,
        )
    except Exception as e:
        print(f"[ERROR] Anomaly model scoring failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("    - Completed model training and confidence categorization.")

    # 5. Explanations
    print("[5/5] Generating SHAP explanations for anomalies...")
    try:
        final_df = explain_anomalies(clf, scored_df, top_n=3)
    except Exception as e:
        print(f"[ERROR] SHAP explanation generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("    - Generated human-readable reasons for flagged wallet entities.")

    # Sort final alerts by anomaly score descending
    final_df = final_df.sort_values(by="anomaly_score", ascending=False)

    # Reorganize columns for standard output representation
    cols = [
        "anomaly_score",
        "anomaly_confidence",
        "attribution_confidence",
        "top_reasons",
    ]
    other_cols = [c for c in final_df.columns if c not in cols]
    final_df = final_df[cols + other_cols]

    # Write outputs
    print("[*] Exporting results...")
    try:
        # Create output directories if needed
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        graph_dir = os.path.dirname(graph_out_path)
        if graph_dir:
            os.makedirs(graph_dir, exist_ok=True)

        # Save alerts CSV
        final_df.to_csv(out_path, index=True, index_label="address")
        print(f"    - Saved ranked alerts CSV to: {out_path}")

        # Save graph object
        with open(graph_out_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    - Saved NetworkX graph pickle to: {graph_out_path}")

    except Exception as e:
        print(f"[ERROR] Failed to export pipeline outputs: {e}", file=sys.stderr)
        sys.exit(1)

    print("[SUCCESS] BitSentinel pipeline execution completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="BitSentinel (SIH26146): Bitcoin Ledger & P2P network traffic monitoring pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic/transactions.csv",
        help="Path to the input transaction telemetry CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/alerts.csv",
        help="Path to write the ranked anomaly alerts CSV.",
    )
    parser.add_argument(
        "--graph-out",
        type=str,
        default="data/entity_graph.pkl",
        help="Path to write the serialized NetworkX graph object.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="eif",
        choices=["standard", "eif"],
        help="The anomaly detection model type to train ('standard' or 'eif').",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Proportion of anomalies to expect (Isolation Forest contamination parameter).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training and reproducibility.",
    )

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        out_path=args.out,
        graph_out_path=args.graph_out,
        contamination=args.contamination,
        model_type=args.model_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
