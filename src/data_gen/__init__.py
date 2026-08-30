"""Data generation package for synthetic Bitcoin blockchain and P2P network telemetry."""

# Expose generate function
def get_generator():
    from src.data_gen.generate import generate
    return generate

__all__ = ["get_generator"]
