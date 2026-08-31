"""Ingestion package exposing transaction CSV loading and validation utility."""

from src.ingestion.loader import load_csv

__all__ = ["load_csv"]
