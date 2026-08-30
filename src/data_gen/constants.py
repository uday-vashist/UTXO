"""Static reference datasets, IP pools, and Bitcoin address generators for offline data generation."""

import hashlib
import random
import string

# Base58 and Bech32 character sets
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

# Script Types
SCRIPT_TYPES = ["P2PKH", "P2WPKH", "P2SH", "P2TR"]
SCRIPT_TYPE_WEIGHTS = [0.35, 0.45, 0.15, 0.05]

# Static GeoIP and ASN pool (Country code -> List of (ASN, Subnet prefix, is_hosting/residential))
GEO_ASN_POOLS = [
    {"country": "US", "asn": "AS7922 Comcast Cable Communications", "subnet_prefix": "73.15.", "category": "residential"},
    {"country": "US", "asn": "AS7018 AT&T Services Inc.", "subnet_prefix": "12.186.", "category": "residential"},
    {"country": "US", "asn": "AS16509 Amazon.com Inc.", "subnet_prefix": "54.210.", "category": "datacenter"},
    {"country": "US", "asn": "AS13335 Cloudflare Inc.", "subnet_prefix": "104.28.", "category": "datacenter"},
    {"country": "DE", "asn": "AS3320 Deutsche Telekom AG", "subnet_prefix": "80.187.", "category": "residential"},
    {"country": "DE", "asn": "AS24940 Hetzner Online GmbH", "subnet_prefix": "188.40.", "category": "datacenter"},
    {"country": "NL", "asn": "AS16276 OVH SAS Netherlands", "subnet_prefix": "51.89.", "category": "datacenter"},
    {"country": "NL", "asn": "AS49981 WorldStream B.V.", "subnet_prefix": "194.187.", "category": "datacenter"},
    {"country": "CH", "asn": "AS3303 Swisscom (Switzerland) Ltd", "subnet_prefix": "195.141.", "category": "residential"},
    {"country": "CH", "asn": "AS6730 Sunrise Communications AG", "subnet_prefix": "178.197.", "category": "residential"},
    {"country": "RU", "asn": "AS12389 PJSC Rostelecom", "subnet_prefix": "188.162.", "category": "residential"},
    {"country": "RU", "asn": "AS20485 PJSC Vimpelcom", "subnet_prefix": "217.118.", "category": "residential"},
    {"country": "SG", "asn": "AS45431 Singtel", "subnet_prefix": "116.14.", "category": "residential"},
    {"country": "SG", "asn": "AS4657 StarHub Ltd", "subnet_prefix": "182.19.", "category": "residential"},
    {"country": "PA", "asn": "AS200019 Bulletproof Hosting Panama", "subnet_prefix": "185.220.", "category": "bulletproof"},
    {"country": "SC", "asn": "AS53667 Offshore VPS Solutions Seychelles", "subnet_prefix": "193.32.", "category": "bulletproof"},
]

# Static Tor Exit Node list (Offline bundled dataset)
STATIC_TOR_EXIT_NODES = [
    "185.220.101.5", "185.220.101.6", "185.220.101.7", "185.220.101.8",
    "185.220.102.4", "185.220.102.5", "185.220.102.6", "185.220.102.8",
    "193.32.160.10", "193.32.160.15", "193.32.161.22", "193.32.161.89",
    "51.15.43.205", "51.15.54.19", "51.15.67.111", "51.15.89.44",
    "199.249.230.70", "199.249.230.71", "199.249.230.72", "199.249.230.73",
    "171.25.193.20", "171.25.193.25", "171.25.193.77", "171.25.193.88",
    "109.70.100.24", "109.70.100.25", "109.70.100.26", "109.70.100.27",
    "198.98.56.140", "198.98.56.141", "198.98.57.210", "198.98.58.95",
    "185.246.188.66", "185.246.188.75", "185.246.188.80", "185.246.188.92",
    "162.247.74.200", "162.247.74.201", "162.247.74.202", "162.247.74.203",
    "45.66.35.3", "45.66.35.4", "45.66.35.12", "45.66.35.19",
    "185.100.87.199", "185.100.87.202", "185.100.87.205", "185.100.87.210",
]


def generate_btc_address(script_type: str = "P2PKH", rng: random.Random = None) -> str:
    """Generate a realistic Bitcoin address matching the specified script type."""
    if rng is None:
        rng = random.Random()

    if script_type == "P2PKH":
        # Legacy address starts with 1
        body = "".join(rng.choice(BASE58_ALPHABET) for _ in range(33))
        return f"1{body}"
    elif script_type == "P2SH":
        # Script address starts with 3
        body = "".join(rng.choice(BASE58_ALPHABET) for _ in range(33))
        return f"3{body}"
    elif script_type == "P2WPKH":
        # Native SegWit starts with bc1q
        body = "".join(rng.choice(BECH32_ALPHABET) for _ in range(38))
        return f"bc1q{body}"
    elif script_type == "P2TR":
        # Taproot starts with bc1p
        body = "".join(rng.choice(BECH32_ALPHABET) for _ in range(58))
        return f"bc1p{body}"
    else:
        body = "".join(rng.choice(BASE58_ALPHABET) for _ in range(33))
        return f"1{body}"


def generate_txid(rng: random.Random = None) -> str:
    """Generate a 64-character hex transaction ID."""
    if rng is None:
        rng = random.Random()
    raw = rng.randbytes(32) if hasattr(rng, "randbytes") else bytes(rng.getrandbits(8) for _ in range(32))
    return hashlib.sha256(raw).hexdigest()
