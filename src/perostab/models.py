from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, list[object]]


def logreg_spec(numeric_features: list) -> ModelSpec:
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(range(len(numeric_features))))],
        remainder="passthrough",
    )
    calibrated_lr = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000, solver="liblinear"), method="sigmoid", cv=5
    )
    pipe = Pipeline(steps=[("scale", preprocessor), ("clf", calibrated_lr)])
    grid: Dict[str, list[object]] = {
        "clf__estimator__C": [0.1, 1.0, 10.0],
        "clf__estimator__penalty": ["l1", "l2"],
        "clf__estimator__class_weight": [None, "balanced"],
    }
    return ModelSpec("logreg", pipe, grid)


def rf_spec() -> ModelSpec:
    calibrated_rf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(random_state=42), method="isotonic", cv=5
    )
    pipe = Pipeline(steps=[("clf", calibrated_rf)])
    grid: Dict[str, list[object]] = {
        "clf__estimator__n_estimators": [200, 400],
        "clf__estimator__max_depth": [None, 8, 16],
        "clf__estimator__min_samples_split": [2, 5],
        "clf__estimator__min_samples_leaf": [1, 2],
        "clf__estimator__class_weight": [None, "balanced"],
    }
    return ModelSpec("random_forest", pipe, grid)


def build_grid_search(model_spec: ModelSpec, cv, scoring: str = "roc_auc") -> GridSearchCV:
    gs = GridSearchCV(
        estimator=model_spec.pipeline,
        param_grid=model_spec.param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    return gs
