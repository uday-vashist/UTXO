"""Ingestion loader and validation logic for Bitcoin transaction telemetry logs."""

import os
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "txid",
    "input_addresses",
    "output_addresses",
    "output_amounts",
    "amount_btc",
    "fee_btc",
    "script_type",
    "geo_country",
    "asn",
    "is_tor_exit",
]


def load_csv(file_path: str) -> pd.DataFrame:
    """Loads and validates a transaction telemetry CSV file.

    Args:
        file_path: Absolute or relative path to the telemetry CSV file.

    Returns:
        pd.DataFrame: A cleaned and validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV violates the schema or data type constraints.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    try:
        # Load CSV
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e

    # 1. Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in telemetry CSV: {missing_cols}")

    # 2. Basic cleanup: strip string whitespace
    string_cols = [
        "src_ip",
        "dst_ip",
        "txid",
        "input_addresses",
        "output_addresses",
        "output_amounts",
        "script_type",
        "geo_country",
        "asn",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3. Handle data conversions and type validation
    try:
        # Convert timestamp to UTC ISO format (standard string representation)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        raise ValueError(f"Failed to parse 'timestamp' as valid datetime: {e}") from e

    try:
        df["amount_btc"] = pd.to_numeric(df["amount_btc"], errors="raise").astype(float)
        df["fee_btc"] = pd.to_numeric(df["fee_btc"], errors="raise").astype(float)
    except Exception as e:
        raise ValueError(f"Numeric parse error in 'amount_btc' or 'fee_btc': {e}") from e

    try:
        df["src_port"] = pd.to_numeric(df["src_port"], errors="raise").astype(int)
        df["dst_port"] = pd.to_numeric(df["dst_port"], errors="raise").astype(int)
    except Exception as e:
        raise ValueError(f"Integer parse error in 'src_port' or 'dst_port': {e}") from e

    # Convert is_tor_exit to boolean
    df["is_tor_exit"] = df["is_tor_exit"].astype(bool)

    # 4. Critical null checks (key fields cannot be null/empty)
    critical_fields = ["txid", "timestamp", "src_ip", "input_addresses", "output_addresses"]
    for field in critical_fields:
        null_count = df[field].isnull().sum()
        empty_count = (df[field] == "").sum()
        if null_count > 0 or empty_count > 0:
            raise ValueError(f"Field '{field}' contains null or empty values.")

    return df
