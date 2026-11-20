"""Sanity checks for the friendly explorer and exact split reproduction."""
import joblib
from pathlib import Path
from src.common import load_data
from src.demo import sample_groups, class_name, predicted_label


def test_sample_groups_match_training_counts_and_do_not_overlap():
    X, y = load_data()
    groups = sample_groups(X, y)
    assert {group: len(ids) for group, ids in groups.items()} == {
        "train": 341, "validation": 114, "test": 114
    }
    assert len(set(groups["train"] + groups["validation"] + groups["test"])) == 569


def test_labels_and_threshold_match_training_convention():
    assert class_name(1) == "Malignant"
    assert class_name(0) == "Benign"
    assert predicted_label(0.67, 0.675) == 0
    assert predicted_label(0.675, 0.675) == 1


def test_reconstructed_test_split_matches_saved_report():
    X, y = load_data()
    groups = sample_groups(X, y)
    root = Path(__file__).resolve().parents[1]
    artifact = joblib.load(root / "models" / "model.joblib")
    case_ids = groups["test"]
    scores = artifact["pipeline"].predict_proba(X.loc[case_ids, artifact["columns"]])[:, 1]
    predictions = [predicted_label(s, artifact["threshold"]) for s in scores]
    actual = [int(y.loc[i]) for i in case_ids]
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predictions))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predictions))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predictions))
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predictions))
    assert (tn, fp, fn, tp) == (72, 0, 3, 39)
