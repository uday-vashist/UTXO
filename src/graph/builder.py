"""Entity graph builder for Bitcoin blockchain and network propagation telemetry."""

from typing import List
import networkx as nx
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


def _parse_semicolon_list(val: object) -> List[str]:
    """Parse a semicolon-delimited string into a list of non-empty stripped strings."""
    if val is None or pd.isna(val):
        return []
    return [item.strip() for item in str(val).split(";") if item.strip()]


def _parse_output_amounts(val: object, row_idx: int) -> List[float]:
    """Parse a semicolon-delimited string of floats."""
    items = _parse_semicolon_list(val)
    parsed: List[float] = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError as e:
            raise ValueError(
                f"Row {row_idx}: Invalid float in output_amounts: '{item}'"
            ) from e
    return parsed


from collections import Counter


def _is_coinjoin(input_addrs: List[str], output_amts: List[float]) -> bool:
    """Heuristic to detect CoinJoin/mixing transactions.

    A transaction is identified as a CoinJoin if it has multiple inputs and
    multiple outputs of identical denomination (e.g. Wasabi/Whirlpool equal-output pattern).
    Applying the Common-Input Ownership Heuristic to such transactions creates false clusters.
    """
    if len(input_addrs) < 2 or len(output_amts) < 2:
        return False
    counts = Counter(output_amts)
    max_equal_outputs = max(counts.values()) if counts else 0
    # If 3+ inputs and 2+ equal outputs, or 2 inputs and 2 equal outputs
    if len(input_addrs) >= 3 and max_equal_outputs >= 2:
        return True
    if len(input_addrs) >= 2 and max_equal_outputs >= 2 and len(output_amts) >= 3:
        return True
    return False


def _calculate_attribution_confidence(is_tor_exit: bool) -> float:
    """Calculate attribution confidence score for a first-broadcast edge.

    Direct residential/datacenter broadcasts receive high base confidence (0.85).
    Tor-exit node broadcasts receive penalized confidence (0.35) due to relay anonymization.
    """
    if is_tor_exit:
        return 0.35
    return 0.85


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Builds a NetworkX MultiDiGraph from a validated 15-column Bitcoin transaction DataFrame.

    Nodes created:
      - wallet: node_type="wallet"
      - txid: node_type="txid", timestamp, amount_btc, fee_btc, script_type
      - ip: node_type="ip", geo_country, asn, is_tor_exit

    Edges created:
      - flow:
          - input_wallet -> txid (inflow)
          - txid -> output_wallet (outflow, with amount_btc from output_amounts)
      - co_spend:
          - input_wallet_a -> input_wallet_b (and reverse) for distinct inputs in multi-input txs
      - first_broadcast:
          - src_ip -> input_wallet for the earliest broadcast event of each txid,
            weighted by attribution_confidence

    Args:
        df: DataFrame containing the 15 required telemetry columns.

    Returns:
        networkx.MultiDiGraph: Heterogeneous directed multigraph representing entities and interactions.

    Raises:
        ValueError: If required columns are missing or if output_addresses and output_amounts
                    have mismatched lengths.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # 1. Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    G = nx.MultiDiGraph()

    if df.empty:
        return G

    # Optimize: Pre-clean and strip string columns in pandas to avoid per-row string operation overhead
    df_clean = df.copy()
    df_clean["txid"] = df_clean["txid"].astype(str).str.strip()
    df_clean["src_ip"] = df_clean["src_ip"].astype(str).str.strip()
    df_clean["geo_country"] = df_clean["geo_country"].astype(str).str.strip()
    df_clean["asn"] = df_clean["asn"].astype(str).str.strip()
    df_clean["script_type"] = df_clean["script_type"].astype(str).str.strip()

    # Find earliest broadcast event per txid by chronological timestamp ordering
    sorted_df = df_clean.sort_values(by="timestamp").reset_index(drop=True)
    seen_txids = set()

    for idx, row in sorted_df.iterrows():
        txid = row["txid"]
        if not txid:
            continue

        ts = str(row["timestamp"])
        src_ip = row["src_ip"]
        geo_country = row["geo_country"]
        asn = row["asn"]
        is_tor = bool(row["is_tor_exit"])
        script_type = row["script_type"]
        amount_btc = float(row["amount_btc"])
        fee_btc = float(row["fee_btc"])

        # Parse inputs, outputs, and output amounts
        input_addrs = _parse_semicolon_list(row["input_addresses"])
        output_addrs = _parse_semicolon_list(row["output_addresses"])
        output_amts = _parse_output_amounts(row["output_amounts"], row_idx=idx)

        # Validate matching output lengths (strict 1-to-1 requirement)
        if len(output_addrs) != len(output_amts):
            raise ValueError(
                f"Row {idx} (txid {txid}): Count mismatch between output_addresses ({len(output_addrs)}) "
                f"and output_amounts ({len(output_amts)})."
            )

        is_cj = _is_coinjoin(input_addrs, output_amts)

        # 2. Add Transaction Node
        if txid not in G:
            G.add_node(
                txid,
                node_type="txid",
                timestamp=ts,
                amount_btc=amount_btc,
                fee_btc=fee_btc,
                script_type=script_type,
                is_coinjoin=is_cj,
            )

        # 3. Add IP Node
        if src_ip and src_ip not in G:
            G.add_node(
                src_ip,
                node_type="ip",
                geo_country=geo_country,
                asn=asn,
                is_tor_exit=is_tor,
            )

        # 4. Add Wallet Nodes and Flow Edges
        # Input Wallets & Inflow Edges (wallet -> txid)
        # Attribute proportional contributed input share to avoid gross-volume inflation (Finding B2)
        per_input_amt = round(amount_btc / len(input_addrs), 8) if input_addrs else amount_btc
        for in_addr in input_addrs:
            if in_addr not in G:
                G.add_node(in_addr, node_type="wallet")
            G.add_edge(
                in_addr,
                txid,
                edge_type="flow",
                txid=txid,
                timestamp=ts,
                amount_btc=per_input_amt,
            )

        # Output Wallets & Outflow Edges (txid -> wallet)
        for out_addr, out_amt in zip(output_addrs, output_amts):
            if out_addr not in G:
                G.add_node(out_addr, node_type="wallet")
            G.add_edge(
                txid,
                out_addr,
                edge_type="flow",
                txid=txid,
                timestamp=ts,
                amount_btc=out_amt,
            )

        # 5. Add Co-Spend Edges (multi-input non-CoinJoin transactions only)
        # Prevents false clustering of unrelated CoinJoin/mixer participants (Finding B1)
        if len(input_addrs) > 1 and not is_cj:
            for i in range(len(input_addrs)):
                for j in range(i + 1, len(input_addrs)):
                    addr_a = input_addrs[i]
                    addr_b = input_addrs[j]
                    if addr_a != addr_b:
                        # Bidirectional co-spend edges
                        G.add_edge(
                            addr_a,
                            addr_b,
                            edge_type="co_spend",
                            txid=txid,
                            timestamp=ts,
                        )
                        G.add_edge(
                            addr_b,
                            addr_a,
                            edge_type="co_spend",
                            txid=txid,
                            timestamp=ts,
                        )

        # 6. Add First-Broadcast Edges (src_ip -> input_wallet)
        if txid not in seen_txids and src_ip:
            seen_txids.add(txid)
            conf = _calculate_attribution_confidence(is_tor_exit=is_tor)
            for in_addr in input_addrs:
                G.add_edge(
                    src_ip,
                    in_addr,
                    edge_type="first_broadcast",
                    txid=txid,
                    timestamp=ts,
                    attribution_confidence=conf,
                )

    return G
