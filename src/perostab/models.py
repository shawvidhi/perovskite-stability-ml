from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, list]


def logreg_spec(numeric_features: list) -> ModelSpec:
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(range(len(numeric_features))))],
        remainder="passthrough",
    )
    base = Pipeline(
        steps=[
            ("scale", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )
    calibrated = Pipeline(
        steps=[
            ("base", base),
            (
                "cal",
                CalibratedClassifierCV(base_estimator=None, method="sigmoid", cv=5),
            ),
        ]
    )
    grid = {
        "base__clf__C": [0.1, 1.0, 10.0],
        "base__clf__penalty": ["l1", "l2"],
        "base__clf__class_weight": [None, "balanced"],
    }
    return ModelSpec("logreg", calibrated, grid)


def rf_spec() -> ModelSpec:
    base = Pipeline(steps=[("clf", RandomForestClassifier(random_state=42))])
    calibrated = Pipeline(
        steps=[
            ("base", base),
            (
                "cal",
                CalibratedClassifierCV(base_estimator=None, method="isotonic", cv=5),
            ),
        ]
    )
    grid = {
        "base__clf__n_estimators": [200, 400],
        "base__clf__max_depth": [None, 8, 16],
        "base__clf__min_samples_split": [2, 5],
        "base__clf__min_samples_leaf": [1, 2],
        "base__clf__class_weight": [None, "balanced"],
    }
    return ModelSpec("random_forest", calibrated, grid)


def build_grid_search(model_spec: ModelSpec, cv, scoring: str = "roc_auc") -> GridSearchCV:
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    gs = GridSearchCV(
        estimator=model_spec.pipeline,
        param_grid=model_spec.param_grid,
        scoring=scorer if scoring == "roc_auc" else scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    return gs

