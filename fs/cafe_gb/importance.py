# fs/cafe_gb/importance.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def compute_chunk_importance(chunk_df, target_col, random_state):
    """
    Train GB on a chunk and return positive feature importances.

    Returns:
        dict: {feature_name: importance}
    """
    X = chunk_df.drop(columns=[target_col])
    y = chunk_df[target_col]

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X, y)

    importances = model.feature_importances_
    importance_dict = {
        feature: imp
        for feature, imp in zip(X.columns, importances)
        if imp > 0
    }
    return importance_dict
