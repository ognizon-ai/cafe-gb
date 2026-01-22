# fs/cafe_gb/chunking.py

import pandas as pd

def generate_overlapping_chunks(df, chunk_size, overlap_ratio):
    """
    Generate overlapping chunks from a DataFrame.

    Args:
        df (pd.DataFrame): input dataset
        chunk_size (int): number of samples per chunk
        overlap_ratio (float): overlap percentage (0–1)

    Yields:
        pd.DataFrame: chunk
    """
    step = int(chunk_size * (1 - overlap_ratio))
    if step <= 0:
        raise ValueError("Overlap ratio too high; step size <= 0")

    n = len(df)
    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        yield df.iloc[start:end]
