"""Shared data schema and feature loading."""
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """Load the real Wisconsin Diagnostic Breast Cancer dataset packaged with sklearn.
    Target: 1 = malignant, 0 = benign (re-coded from sklearn target).
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [col.replace(' ', '_') for col in X.columns]
    y = pd.Series(1 - raw.target.to_numpy(), name='malignant', index=X.index)
    return X, y

def validate_input(X):
    if X.isna().any().any():
        raise ValueError('Missing predictor values detected')
    if not X.select_dtypes(exclude='number').empty:
        raise ValueError('Non-numeric predictors detected')
    if (X < 0).any().any():
        raise ValueError('Negative measurement values detected')
