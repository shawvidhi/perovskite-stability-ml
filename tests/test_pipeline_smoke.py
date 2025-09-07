from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from perostab.dataset import (
    generate_synthetic_dataset,
    save_dataset_csv,
    train_test_split_features,
)


def test_end_to_end_smoke(tmp_path: Path):
    # Generate small dataset
    df = generate_synthetic_dataset(n=300, seed=123)
    csv = tmp_path / "abx3.csv"
    save_dataset_csv(df, csv)

    splits = train_test_split_features(csv, test_size=0.2, seed=123)

    # Fit a tiny RF quickly
    clf = RandomForestClassifier(n_estimators=50, random_state=123)
    clf.fit(splits.X_train, splits.y_train)
    proba = clf.predict_proba(splits.X_test)[:, 1]
    auc = roc_auc_score(splits.y_test, proba)
    assert 0.5 < auc < 1.0

    # Save/load model
    model_path = tmp_path / "rf.joblib"
    joblib.dump(clf, model_path)
    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict_proba")
