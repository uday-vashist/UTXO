"""Synthetic Bitcoin transaction generator fusing on-chain ledger and P2P network traffic.

Implements realistic normal transactions, mixing/layering patterns, peeling chains,
and injected illicit clusters with offline static reference datasets.
"""

import argparse
import datetime
import os
import random
from typing import Dict, List, Optional, Set, Tuple

from faker import Faker
import numpy as np
import pandas as pd

from src.data_gen.constants import (
    GEO_ASN_POOLS,
    SCRIPT_TYPES,
    SCRIPT_TYPE_WEIGHTS,
    STATIC_TOR_EXIT_NODES,
    generate_btc_address,
    generate_txid,
)


class SyntheticDataGenerator:
    """Generates synthetic Bitcoin blockchain and network propagation telemetry."""

    def __init__(
        self,
        n_wallets: int = 500,
        n_txns: int = 5000,
        illicit_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.n_wallets = max(50, n_wallets)
        self.n_txns = max(100, n_txns)
        self.illicit_ratio = max(0.0, min(0.5, illicit_ratio))
        self.seed = seed

        # Initialize RNGs
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.fake = Faker()
        Faker.seed(seed)

        # Pre-generate regular wallet pool
        self.wallets: List[Dict] = []
        self.wallet_addresses: List[str] = []
        self.ip_pool: List[Dict] = []
        self.tor_exit_set: Set[str] = set(STATIC_TOR_EXIT_NODES)

        self._init_ip_pool()
        self._init_wallets()

    def _init_ip_pool(self, size: int = 200) -> None:
        """Initialize static pool of IP addresses mapped to Geo and ASN data."""
        self.ip_pool = []
        for i in range(size):
            pool_entry = self.rng.choice(GEO_ASN_POOLS)
            prefix = pool_entry["subnet_prefix"]
            host = f"{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}"
            ip = f"{prefix}{host}"
            self.ip_pool.append({
                "ip": ip,
                "country": pool_entry["country"],
                "asn": pool_entry["asn"],
                "category": pool_entry["category"],
                "is_tor": ip in self.tor_exit_set,
            })

        # Add all static Tor exit nodes to ip pool
        for tor_ip in STATIC_TOR_EXIT_NODES:
            self.ip_pool.append({
                "ip": tor_ip,
                "country": self.rng.choice(["DE", "NL", "CH", "US", "PA", "SC"]),
                "asn": self.rng.choice([
                    "AS200019 Bulletproof Hosting Panama",
                    "AS53667 Offshore VPS Solutions Seychelles",
                    "AS24940 Hetzner Online GmbH",
                    "AS16276 OVH SAS Netherlands",
                ]),
                "category": "tor_exit",
                "is_tor": True,
            })

    def _init_wallets(self) -> None:
        """Create regular wallet entities with preferred script types and primary IPs."""
        self.wallets = []
        self.wallet_addresses = []
        for _ in range(self.n_wallets):
            st = self.rng.choices(SCRIPT_TYPES, weights=SCRIPT_TYPE_WEIGHTS, k=1)[0]
            addr = generate_btc_address(script_type=st, rng=self.rng)
            primary_ip = self.rng.choice([entry for entry in self.ip_pool if not entry["is_tor"]])

            wallet_obj = {
                "address": addr,
                "script_type": st,
                "primary_ip": primary_ip,
                "balance": float(self.np_rng.lognormal(mean=0.5, sigma=1.2)),
            }
            self.wallets.append(wallet_obj)
            self.wallet_addresses.append(addr)

    def _get_random_ip(self, tor_only: bool = False, non_tor_only: bool = False) -> Dict:
        """Fetch an IP entry with optional filtering."""
        if tor_only:
            tor_entries = [entry for entry in self.ip_pool if entry["is_tor"]]
            return self.rng.choice(tor_entries)
        if non_tor_only:
            non_tor_entries = [entry for entry in self.ip_pool if not entry["is_tor"]]
            return self.rng.choice(non_tor_entries)
        return self.rng.choice(self.ip_pool)

    def _generate_normal_txns(
        self, count: int, start_time: datetime.datetime, time_step_seconds: float
    ) -> List[Dict]:
        """Generate standard benign Bitcoin transactions."""
        txns = []
        current_time = start_time

        for i in range(count):
            # Advance time realistically
            delta_seconds = float(self.np_rng.exponential(scale=time_step_seconds))
            current_time += datetime.timedelta(seconds=delta_seconds)

            sender_wallet = self.rng.choice(self.wallets)
            receiver_wallet = self.rng.choice(self.wallets)
            while receiver_wallet["address"] == sender_wallet["address"]:
                receiver_wallet = self.rng.choice(self.wallets)

            # 1 or 2 inputs (co-spend heuristic simulation)
            use_co_spend = self.rng.random() < 0.25
            if use_co_spend:
                sibling_wallet = self.rng.choice(self.wallets)
                input_addrs = [sender_wallet["address"], sibling_wallet["address"]]
            else:
                input_addrs = [sender_wallet["address"]]

            # 1 or 2 outputs (payment + optional change)
            has_change = self.rng.random() < 0.70
            if has_change:
                change_addr = generate_btc_address(script_type=sender_wallet["script_type"], rng=self.rng)
                output_addrs = [receiver_wallet["address"], change_addr]
            else:
                output_addrs = [receiver_wallet["address"]]

            amount_btc = round(float(self.np_rng.lognormal(mean=-1.5, sigma=1.0)), 6)
            amount_btc = max(0.0001, min(50.0, amount_btc))
            if has_change:
                change_amount = round(self.rng.uniform(0.05, 1.5) * amount_btc, 6)
                output_amounts = [amount_btc, change_amount]
            else:
                output_amounts = [amount_btc]
            fee_btc = round(float(self.np_rng.uniform(0.00002, 0.00030)), 8)

            # Network telemetry
            # Usually uses the sender's known residential/datacenter IP
            ip_info = sender_wallet["primary_ip"]
            # 5% chance of minor IP roaming (e.g. mobile hotspot / different wifi)
            if self.rng.random() < 0.05:
                ip_info = self._get_random_ip(non_tor_only=True)

            dst_ip_info = self.rng.choice(self.ip_pool)

            txid = generate_txid(rng=self.rng)
            txns.append({
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "src_ip": ip_info["ip"],
                "src_port": self.rng.randint(1025, 65530),
                "dst_ip": dst_ip_info["ip"],
                "dst_port": 8333,
                "txid": txid,
                "input_addresses": ";".join(input_addrs),
                "output_addresses": ";".join(output_addrs),
                "output_amounts": ";".join(str(x) for x in output_amounts),
                "amount_btc": amount_btc,
                "fee_btc": fee_btc,
                "script_type": sender_wallet["script_type"],
                "geo_country": ip_info["country"],
                "asn": ip_info["asn"],
                "is_tor_exit": ip_info["is_tor"],
            })

        return txns

    def _generate_mixing_cluster(
        self, cluster_idx: int, base_time: datetime.datetime
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate CoinJoin/Wasabi-style mixing service transactions."""
        txns = []
        ground_truth = []
        cluster_id = f"ILLICIT_MIXING_{cluster_idx:02d}"

        # 8-15 participant inputs, 8-15 equal denomination outputs
        n_participants = self.rng.randint(8, 15)
        st = "P2WPKH"
        input_wallets = [
            generate_btc_address(script_type=st, rng=self.rng)
            for _ in range(n_participants)
        ]
        uniform_amount = self.rng.choice([0.05, 0.1, 0.5, 1.0])
        output_wallets = [
            generate_btc_address(script_type=st, rng=self.rng)
            for _ in range(n_participants)
        ]
        # Add change addresses
        change_wallets = [
            generate_btc_address(script_type=st, rng=self.rng)
            for _ in range(self.rng.randint(2, 5))
        ]

        # Broadcasted via Tor exit node
        tor_ip_info = self._get_random_ip(tor_only=True)
        dst_ip_info = self.rng.choice(self.ip_pool)

        total_amount = round(uniform_amount * n_participants + self.rng.uniform(0.01, 0.5), 6)
        fee_btc = round(0.0005 + (0.00005 * n_participants), 8)

        # Distribute amounts cleanly matching the shuffle
        output_mappings = [(addr, uniform_amount) for addr in output_wallets]
        change_total = round(total_amount - (uniform_amount * n_participants), 6)
        if len(change_wallets) > 0:
            change_share = round(change_total / len(change_wallets), 6)
            for addr in change_wallets:
                output_mappings.append((addr, change_share))
        else:
            total_amount = round(uniform_amount * n_participants, 6)

        self.rng.shuffle(output_mappings)
        shuffled_addrs = [item[0] for item in output_mappings]
        shuffled_amounts = [item[1] for item in output_mappings]

        txid = generate_txid(rng=self.rng)
        timestamp = (base_time + datetime.timedelta(seconds=self.rng.randint(10, 300))).strftime("%Y-%m-%dT%H:%M:%SZ")

        txns.append({
            "timestamp": timestamp,
            "src_ip": tor_ip_info["ip"],
            "src_port": self.rng.randint(1025, 65530),
            "dst_ip": dst_ip_info["ip"],
            "dst_port": 8333,
            "txid": txid,
            "input_addresses": ";".join(input_wallets),
            "output_addresses": ";".join(shuffled_addrs),
            "output_amounts": ";".join(str(x) for x in shuffled_amounts),
            "amount_btc": total_amount,
            "fee_btc": fee_btc,
            "script_type": st,
            "geo_country": tor_ip_info["country"],
            "asn": tor_ip_info["asn"],
            "is_tor_exit": True,
        })

        # Ground truth labels
        for addr in input_wallets:
            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "wallet",
                "entity_id": addr,
                "illicit_type": "mixing_service",
                "pattern_details": "CoinJoin/Mixer multi-input co-spend participant",
            })
        for addr in output_wallets:
            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "wallet",
                "entity_id": addr,
                "illicit_type": "mixing_service",
                "pattern_details": "CoinJoin/Mixer anonymized uniform output",
            })
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "txid",
            "entity_id": txid,
            "illicit_type": "mixing_service",
            "pattern_details": f"Many-in/many-out ({n_participants} peers) via Tor exit node",
        })
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "ip",
            "entity_id": tor_ip_info["ip"],
            "illicit_type": "mixing_service",
            "pattern_details": "Tor exit node used for CoinJoin coordination",
        })

        return txns, ground_truth

    def _generate_peeling_chain(
        self, cluster_idx: int, base_time: datetime.datetime
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate peeling chain: large initial input peeled off to destination + fresh change."""
        txns = []
        ground_truth = []
        cluster_id = f"ILLICIT_PEELING_{cluster_idx:02d}"

        hops = self.rng.randint(6, 12)
        initial_stash = round(self.rng.uniform(30.0, 100.0), 6)
        current_amount = initial_stash
        st = "P2PKH"

        current_input_wallet = generate_btc_address(script_type=st, rng=self.rng)
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "wallet",
            "entity_id": current_input_wallet,
            "illicit_type": "peeling_chain",
            "pattern_details": "Peeling chain root source address",
        })

        # Peeling chain uses bulletproof host or VPN
        ip_info = self._get_random_ip()
        current_time = base_time

        for hop in range(hops):
            peel_amount = round(self.rng.uniform(0.5, 2.5), 6)
            fee = 0.00015
            change_amount = round(current_amount - peel_amount - fee, 6)
            if change_amount <= 0:
                break

            peel_dest_wallet = generate_btc_address(script_type=st, rng=self.rng)
            next_change_wallet = generate_btc_address(script_type=st, rng=self.rng)

            txid = generate_txid(rng=self.rng)
            # Rapid sequential hops (10 to 90 seconds between hops)
            current_time += datetime.timedelta(seconds=self.rng.randint(15, 90))

            txns.append({
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "src_ip": ip_info["ip"],
                "src_port": self.rng.randint(1025, 65530),
                "dst_ip": self.rng.choice(self.ip_pool)["ip"],
                "dst_port": 8333,
                "txid": txid,
                "input_addresses": current_input_wallet,
                "output_addresses": f"{peel_dest_wallet};{next_change_wallet}",
                "output_amounts": f"{peel_amount};{change_amount}",
                "amount_btc": peel_amount,
                "fee_btc": fee,
                "script_type": st,
                "geo_country": ip_info["country"],
                "asn": ip_info["asn"],
                "is_tor_exit": ip_info["is_tor"],
            })

            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "txid",
                "entity_id": txid,
                "illicit_type": "peeling_chain",
                "pattern_details": f"Peeling chain hop {hop + 1}/{hops} (peeled {peel_amount} BTC)",
            })
            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "wallet",
                "entity_id": peel_dest_wallet,
                "illicit_type": "peeling_chain",
                "pattern_details": "Peeled destination/merchant cashout address",
            })
            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "wallet",
                "entity_id": next_change_wallet,
                "illicit_type": "peeling_chain",
                "pattern_details": f"Peeling chain intermediary change address hop {hop + 1}",
            })

            current_input_wallet = next_change_wallet
            current_amount = change_amount

        return txns, ground_truth

    def _generate_rapid_ip_hopping(
        self, cluster_idx: int, base_time: datetime.datetime
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate high-velocity burst transactions hopping across multiple ASNs/subnets."""
        txns = []
        ground_truth = []
        cluster_id = f"ILLICIT_BURST_HOPPER_{cluster_idx:02d}"

        st = "P2WPKH"
        source_wallet = generate_btc_address(script_type=st, rng=self.rng)
        burst_count = self.rng.randint(8, 16)

        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "wallet",
            "entity_id": source_wallet,
            "illicit_type": "rapid_ip_hopping",
            "pattern_details": f"High frequency burst wallet hopping across {burst_count} distinct IP locations",
        })

        current_time = base_time
        for i in range(burst_count):
            # Burst within 5-20 seconds each
            current_time += datetime.timedelta(seconds=self.rng.randint(3, 20))
            # Pick a totally different IP and ASN each time
            ip_info = self.rng.choice(self.ip_pool)
            dest_wallet = generate_btc_address(script_type=st, rng=self.rng)
            txid = generate_txid(rng=self.rng)
            amount_val = round(self.rng.uniform(0.1, 2.0), 6)
            txns.append({
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "src_ip": ip_info["ip"],
                "src_port": self.rng.randint(1025, 65530),
                "dst_ip": self.rng.choice(self.ip_pool)["ip"],
                "dst_port": 8333,
                "txid": txid,
                "input_addresses": source_wallet,
                "output_addresses": dest_wallet,
                "output_amounts": str(amount_val),
                "amount_btc": amount_val,
                "fee_btc": 0.00025,
                "script_type": st,
                "geo_country": ip_info["country"],
                "asn": ip_info["asn"],
                "is_tor_exit": ip_info["is_tor"],
            })

            ground_truth.append({
                "cluster_id": cluster_id,
                "entity_type": "txid",
                "entity_id": txid,
                "illicit_type": "rapid_ip_hopping",
                "pattern_details": f"Rapid burst transaction {i + 1}/{burst_count} from {ip_info['country']} / {ip_info['asn']}",
            })

        return txns, ground_truth

    def _generate_tor_ransomware_payout(
        self, cluster_idx: int, base_time: datetime.datetime
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate high-value ransomware / darknet payout via Tor exit node."""
        txns = []
        ground_truth = []
        cluster_id = f"ILLICIT_RANSOM_TOR_{cluster_idx:02d}"

        st = "P2SH"
        victim_wallet = generate_btc_address(script_type=st, rng=self.rng)
        ransom_wallet = generate_btc_address(script_type=st, rng=self.rng)

        tor_ip_info = self._get_random_ip(tor_only=True)
        # Spike amount: unusually large
        large_amount = round(self.rng.uniform(45.0, 150.0), 6)
        txid = generate_txid(rng=self.rng)
        timestamp = base_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        txns.append({
            "timestamp": timestamp,
            "src_ip": tor_ip_info["ip"],
            "src_port": self.rng.randint(1025, 65530),
            "dst_ip": self.rng.choice(self.ip_pool)["ip"],
            "dst_port": 8333,
            "txid": txid,
            "input_addresses": victim_wallet,
            "output_addresses": ransom_wallet,
            "output_amounts": str(large_amount),
            "amount_btc": large_amount,
            "fee_btc": 0.0008,
            "script_type": st,
            "geo_country": tor_ip_info["country"],
            "asn": tor_ip_info["asn"],
            "is_tor_exit": True,
        })

        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "wallet",
            "entity_id": victim_wallet,
            "illicit_type": "tor_ransomware_payout",
            "pattern_details": "Ransomware extortion funding input wallet",
        })
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "wallet",
            "entity_id": ransom_wallet,
            "illicit_type": "tor_ransomware_payout",
            "pattern_details": f"Darknet / Ransomware destination wallet ({large_amount} BTC payout)",
        })
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "txid",
            "entity_id": txid,
            "illicit_type": "tor_ransomware_payout",
            "pattern_details": f"High amount z-score ({large_amount} BTC) broadcast exclusively through Tor exit relay",
        })
        ground_truth.append({
            "cluster_id": cluster_id,
            "entity_type": "ip",
            "entity_id": tor_ip_info["ip"],
            "illicit_type": "tor_ransomware_payout",
            "pattern_details": "Tor exit node used for ransomware transmission",
        })

        return txns, ground_truth

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate full synthetic transactions dataframe and ground-truth labels dataframe."""
        target_illicit_txns = int(self.n_txns * self.illicit_ratio)
        target_normal_txns = self.n_txns - target_illicit_txns

        start_time = datetime.datetime(2026, 3, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        # Average time spacing ~ 60 seconds
        normal_txns = self._generate_normal_txns(
            count=target_normal_txns, start_time=start_time, time_step_seconds=60.0
        )

        all_txns = list(normal_txns)
        all_ground_truth: List[Dict] = []

        # Inject illicit clusters across timeline
        illicit_generated_count = 0
        cluster_idx = 1
        pattern_generators = [
            self._generate_mixing_cluster,
            self._generate_peeling_chain,
            self._generate_rapid_ip_hopping,
            self._generate_tor_ransomware_payout,
        ]

        while illicit_generated_count < target_illicit_txns:
            # Pick a random insertion time within the dataset's timespan
            insert_offset_seconds = self.rng.randint(1800, max(3600, target_normal_txns * 50))
            cluster_base_time = start_time + datetime.timedelta(seconds=insert_offset_seconds)

            gen_fn = self.rng.choice(pattern_generators)
            c_txns, c_gt = gen_fn(cluster_idx, cluster_base_time)

            all_txns.extend(c_txns)
            all_ground_truth.extend(c_gt)
            illicit_generated_count += len(c_txns)
            cluster_idx += 1

        # Convert to DataFrame
        tx_df = pd.DataFrame(all_txns)

        # Sort chronologically by timestamp
        tx_df["_sort_ts"] = pd.to_datetime(tx_df["timestamp"])
        tx_df = tx_df.sort_values(by="_sort_ts").reset_index(drop=True)
        tx_df = tx_df.drop(columns=["_sort_ts"])

        # Truncate or limit exactly to desired n_txns if needed, or keep complete clusters
        tx_df = tx_df.head(self.n_txns + (illicit_generated_count - target_illicit_txns))

        # Re-ensure exact schema column order as specified in PRD §8
        expected_columns = [
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
        tx_df = tx_df[expected_columns]

        gt_df = pd.DataFrame(all_ground_truth).drop_duplicates().reset_index(drop=True)

        return tx_df, gt_df


def export_static_reference_data(output_dir: str = "data/synthetic") -> None:
    """Export static offline reference tables for GeoIP and Tor exit nodes."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. GeoIP lookup table
    geo_rows = []
    for entry in GEO_ASN_POOLS:
        geo_rows.append({
            "country": entry["country"],
            "asn": entry["asn"],
            "subnet_prefix": entry["subnet_prefix"],
            "category": entry["category"],
        })
    pd.DataFrame(geo_rows).to_csv(os.path.join(output_dir, "geoip_lookup.csv"), index=False)

    # 2. Tor exit node list
    pd.DataFrame({"ip_address": STATIC_TOR_EXIT_NODES}).to_csv(
        os.path.join(output_dir, "tor_exit_nodes.csv"), index=False
    )


def generate(
    n_wallets: int = 500,
    n_txns: int = 5000,
    illicit_ratio: float = 0.05,
    seed: int = 42,
    out_path: str = "data/synthetic/transactions.csv",
    ground_truth_path: str = "data/ground_truth/illicit_clusters.csv",
) -> pd.DataFrame:
    """Public API conforming to Architecture §8 module interface.

    Generates synthetic transaction DataFrame and saves CSVs.
    """
    generator = SyntheticDataGenerator(
        n_wallets=n_wallets,
        n_txns=n_txns,
        illicit_ratio=illicit_ratio,
        seed=seed,
    )
    tx_df, gt_df = generator.generate()

    # Create output directories
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gt_dir = os.path.dirname(ground_truth_path)
    if gt_dir:
        os.makedirs(gt_dir, exist_ok=True)

    # Save transaction CSV
    tx_df.to_csv(out_path, index=False)

    # Save ground truth CSV
    gt_df.to_csv(ground_truth_path, index=False)

    # Export static reference datasets to the synthetic data dir
    synthetic_dir = out_dir if out_dir else "data/synthetic"
    export_static_reference_data(synthetic_dir)

    return tx_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic Bitcoin blockchain and P2P network telemetry."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic/transactions.csv",
        help="Path for generated transactions CSV (default: data/synthetic/transactions.csv)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/ground_truth/illicit_clusters.csv",
        help="Path for ground truth labels CSV (default: data/ground_truth/illicit_clusters.csv)",
    )
    parser.add_argument(
        "--n-wallets",
        type=int,
        default=500,
        help="Number of simulated wallets in the pool (default: 500)",
    )
    parser.add_argument(
        "--n-txns",
        type=int,
        default=5000,
        help="Number of transactions to generate (default: 5000)",
    )
    parser.add_argument(
        "--illicit-ratio",
        type=float,
        default=0.05,
        help="Proportion of illicit/anomalous transactions (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print(f"Generating {args.n_txns} transactions across {args.n_wallets} wallets...")
    print(f"Illicit ratio: {args.illicit_ratio:.2%}, Random seed: {args.seed}")

    df = generate(
        n_wallets=args.n_wallets,
        n_txns=args.n_txns,
        illicit_ratio=args.illicit_ratio,
        seed=args.seed,
        out_path=args.out,
        ground_truth_path=args.ground_truth,
    )

    print(f"[OK] Successfully generated {len(df)} transactions -> {args.out}")
    print(f"[OK] Ground-truth illicit labels saved -> {args.ground_truth}")
    print(f"[OK] Static GeoIP and Tor exit node reference datasets saved.")


if __name__ == "__main__":
    main()
