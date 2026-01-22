"""
CAFÉ-GB: Chunk-wise Aggregated Feature Estimation using Gradient Boosting

This package implements the core components of the CAFÉ-GB feature
selection method used in Paper 1.

Modules:
- chunking: Overlapping chunk generation
- importance: Local feature importance estimation
- aggregate: Aggregation and top-k selection
"""

from .chunking import generate_overlapping_chunks
from .importance import compute_chunk_importance
from .aggregate import aggregate_importances, select_top_k

__all__ = [
    "generate_overlapping_chunks",
    "compute_chunk_importance",
    "aggregate_importances",
    "select_top_k",
]
