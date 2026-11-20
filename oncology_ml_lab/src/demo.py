"""Helpers for the educational sample explorer.

The split is intentionally reconstructed using the same settings as src.train.
This lets the UI label which records were held out for the final test.
"""
from sklearn.model_selection import train_test_split


def sample_groups(X, y):
    """Return row IDs for the exact train, validation and test groups."""
    trainval_ids, test_ids = train_test_split(
        X.index.to_numpy(), test_size=0.20, random_state=42, stratify=y
    )
    train_ids, validation_ids = train_test_split(
        trainval_ids, test_size=0.25, random_state=42,
        stratify=y.loc[trainval_ids],
    )
    return {
        "train": list(map(int, train_ids)),
        "validation": list(map(int, validation_ids)),
        "test": list(map(int, test_ids)),
    }


def class_name(label):
    """The training pipeline uses 1=malignant, 0=benign."""
    return "Malignant" if int(label) == 1 else "Benign"


def predicted_label(score, threshold):
    """Apply the same decision threshold as used in src.train."""
    return 1 if float(score) >= float(threshold) else 0
