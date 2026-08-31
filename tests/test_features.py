"""Unit tests for the feature engineering module."""

import networkx as nx
import pandas as pd
import numpy as np
from src.graph.builder import build_graph
from src.detection.features import compute_features


def _create_test_dataframe() -> pd.DataFrame:
    """Creates a sample transaction DataFrame for testing."""
    rows = [
        # Normal tx 1: 1SenderA -> 1ReceiverB, 1ChangeC. IP is 73.15.1.1
        {
            "timestamp": "2026-03-01T00:00:00Z",
            "src_ip": "73.15.1.1",
            "src_port": 1001,
            "dst_ip": "8.8.8.8",
            "dst_port": 8333,
            "txid": "tx1",
            "input_addresses": "1SenderA",
            "output_addresses": "1ReceiverB;1ChangeC",
            "output_amounts": "10.0;0.5",
            "amount_btc": 10.5,
            "fee_btc": 0.0001,
            "script_type": "P2PKH",
            "geo_country": "US",
            "asn": "AS7922",
            "is_tor_exit": False,
        },
        # Normal tx 2: 1SenderA -> 1ReceiverD. IP switches to 73.15.1.2
        {
            "timestamp": "2026-03-01T00:00:05Z",  # Within 10s of tx1 (burst)
            "src_ip": "73.15.1.2",
            "src_port": 1002,
            "dst_ip": "8.8.8.8",
            "dst_port": 8333,
            "txid": "tx2",
            "input_addresses": "1SenderA",
            "output_addresses": "1ReceiverD",
            "output_amounts": "2.0",
            "amount_btc": 2.0,
            "fee_btc": 0.0001,
            "script_type": "P2PKH",
            "geo_country": "US",
            "asn": "AS7922",
            "is_tor_exit": False,
        },
        # Tor broadcast: 1SenderE -> 1ReceiverB via Tor
        {
            "timestamp": "2026-03-01T00:05:00Z",
            "src_ip": "185.220.101.5",
            "src_port": 1003,
            "dst_ip": "8.8.8.8",
            "dst_port": 8333,
            "txid": "tx3",
            "input_addresses": "1SenderE",
            "output_addresses": "1ReceiverB",
            "output_amounts": "1.0",
            "amount_btc": 1.0,
            "fee_btc": 0.0001,
            "script_type": "P2PKH",
            "geo_country": "DE",
            "asn": "AS3320",
            "is_tor_exit": True,
        },
        # Co-spend tx: 1SenderA & 1SenderE -> 1ReceiverF
        {
            "timestamp": "2026-03-01T00:10:00Z",
            "src_ip": "73.15.1.1",
            "src_port": 1004,
            "dst_ip": "8.8.8.8",
            "dst_port": 8333,
            "txid": "tx4",
            "input_addresses": "1SenderA;1SenderE",
            "output_addresses": "1ReceiverF",
            "output_amounts": "5.0",
            "amount_btc": 5.0,
            "fee_btc": 0.0002,
            "script_type": "P2PKH",
            "geo_country": "US",
            "asn": "AS7922",
            "is_tor_exit": False,
        }
    ]
    return pd.DataFrame(rows)


def test_features_computation():
    """Test feature engineering calculations against expected behavior."""
    df = _create_test_dataframe()
    G = build_graph(df)
    features_df = compute_features(G)

    # 1. Assert correct wallets are present
    wallets = ["1SenderA", "1ReceiverB", "1ChangeC", "1ReceiverD", "1SenderE", "1ReceiverF"]
    for wallet in wallets:
        assert wallet in features_df.index, f"{wallet} missing from index"

    # 2. Check Tor ratio of 1SenderE (1 tor broadcast of 2 total broadcasts)
    # tx3: 1SenderE (tor), tx4: 1SenderE (non-tor) -> tor ratio should be 0.5
    tor_ratio_e = features_df.loc["1SenderE", "tor_broadcast_ratio"]
    assert tor_ratio_e == 0.5

    # 3. Check IP switching of 1SenderA
    # tx1 (73.15.1.1), tx2 (73.15.1.2), tx4 (73.15.1.1) -> 2 unique IPs, 3 broadcasts -> 2/3 = ~0.666
    ip_switch_a = features_df.loc["1SenderA", "ip_switching_frequency"]
    assert np.isclose(ip_switch_a, 2 / 3)

    # 4. Check degree centrality of 1SenderA is higher than 1ChangeC
    deg_a = features_df.loc["1SenderA", "degree_centrality"]
    deg_c = features_df.loc["1ChangeC", "degree_centrality"]
    assert deg_a > deg_c

    # 5. Check co-spend cluster size for 1SenderA (co-spent with 1SenderE in tx4)
    co_spend_a = features_df.loc["1SenderA", "co_spend_cluster_size"]
    assert co_spend_a == 1.0

    # 6. Check burst ratio of 1SenderA: tx1 (00:00:00) and tx2 (00:00:05) are < 10s apart.
    # tx4 (00:10:00) is far away.
    # Sorted timestamps: 00:00:00, 00:00:05, 00:10:00. Intervals: 5s, 595s.
    # Burst intervals: 1 (5s) out of 2. Burst ratio = 0.5.
    burst_a = features_df.loc["1SenderA", "burst_ratio"]
    assert burst_a == 0.5

    # 7. Check peeling chain score:
    # 1SenderA has sent 3 transactions (tx1, tx2, tx4).
    # tx1 has 2 outputs ("1ReceiverB", "1ChangeC"). 1ChangeC degree is very low.
    # So peeling_chain_score should be > 0.
    peel_a = features_df.loc["1SenderA", "peeling_chain_score"]
    assert peel_a > 0.0

    print("[OK] Feature engineering tests passed successfully!")


if __name__ == "__main__":
    test_features_computation()
