"""Unit and integration tests for synthetic data generation module."""

import os
import pandas as pd
from src.data_gen.generate import generate

def test_synthetic_data_generation():
    out_csv = "data/synthetic/test_txns.csv"
    gt_csv = "data/ground_truth/test_gt.csv"

    df = generate(
        n_wallets=100,
        n_txns=500,
        illicit_ratio=0.08,
        seed=123,
        out_path=out_csv,
        ground_truth_path=gt_csv,
    )

    # 1. Check return DataFrame and file existence
    assert isinstance(df, pd.DataFrame)
    assert os.path.exists(out_csv)
    assert os.path.exists(gt_csv)
    assert len(df) >= 500

    # 2. Check exact schema columns as specified in PRD §8
    expected_cols = [
        "timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
        "txid", "input_addresses", "output_addresses", "output_amounts", "amount_btc",
        "fee_btc", "script_type", "geo_country", "asn", "is_tor_exit"
    ]
    assert list(df.columns) == expected_cols, f"Column mismatch: {list(df.columns)} vs {expected_cols}"

    # 3. Check for absence of null/NaN values in key columns
    assert df["txid"].isnull().sum() == 0
    assert df["timestamp"].isnull().sum() == 0
    assert df["src_ip"].isnull().sum() == 0
    assert df["input_addresses"].isnull().sum() == 0
    assert df["output_addresses"].isnull().sum() == 0
    assert df["amount_btc"].isnull().sum() == 0
    assert (df["amount_btc"] > 0).all()

    # 4. Check ground truth file
    gt_df = pd.read_csv(gt_csv)
    expected_gt_cols = ["cluster_id", "entity_type", "entity_id", "illicit_type", "pattern_details"]
    assert list(gt_df.columns) == expected_gt_cols
    assert len(gt_df) > 0

    # Check illicit types
    illicit_types = set(gt_df["illicit_type"].unique())
    assert len(illicit_types.intersection({"mixing_service", "peeling_chain", "rapid_ip_hopping", "tor_ransomware_payout"})) >= 2

    # Clean up test files
    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(gt_csv):
        os.remove(gt_csv)

    print("[OK] All data generation tests passed successfully!")

if __name__ == "__main__":
    test_synthetic_data_generation()
