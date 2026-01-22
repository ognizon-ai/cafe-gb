# fs/cafe_gb/aggregate.py

from collections import defaultdict
import pandas as pd

def aggregate_importances(list_of_importance_dicts):
    """
    Aggregate feature importances across chunks.

    Args:
        list_of_importance_dicts (list[dict])

    Returns:
        pd.DataFrame with columns [feature, importance]
    """
    agg = defaultdict(float)

    for imp_dict in list_of_importance_dicts:
        for feature, value in imp_dict.items():
            agg[feature] += value

    df = pd.DataFrame(
        {"feature": agg.keys(), "importance": agg.values()}
    ).sort_values("importance", ascending=False)

    return df.reset_index(drop=True)


def select_top_k(agg_df, k):
    return agg_df.head(k)["feature"].tolist()
