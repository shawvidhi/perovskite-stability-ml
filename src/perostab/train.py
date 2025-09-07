from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .dataset import get_cv, save_feature_list, train_test_split_features
from .models import build_grid_search, logreg_spec, rf_spec
from .utils.logger import get_logger
from .utils.seeds import set_global_seed


logger = get_logger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train perovskite stability models")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_csv = Path(cfg.get("data_csv", "data/synthetic/abx3_synthetic.csv"))
    splits = train_test_split_features(data_csv, test_size=cfg.get("test_size", 0.2), seed=seed)
    feature_list_path = Path(cfg.get("feature_list_path", "models/features.json"))
    save_feature_list(splits.feature_names, feature_list_path)

    model_type = cfg.get("model", "random_forest")
    if model_type == "random_forest":
        spec = rf_spec()
    elif model_type == "logreg":
        # numeric feature count equals first block; build list for scaling
        numeric_features = list(range(7))  # t, mu, delta_t, delta_mu, dAX, dBX, ratio
        spec = logreg_spec(numeric_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cv = get_cv(n_splits=cfg.get("cv_folds", 5), seed=seed)
    gs = build_grid_search(spec, cv=cv, scoring=cfg.get("scoring", "roc_auc"))

    logger.info("Fitting GridSearchCV for %s", model_type)
    gs.fit(splits.X_train, splits.y_train)
    best = gs.best_estimator_
    logger.info("Best params: %s", gs.best_params_)

    # Evaluate on validation (here, use test set for simplicity; downstream eval script repeats)
    proba = best.predict_proba(splits.X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(splits.y_test, proba),
        "average_precision": average_precision_score(splits.y_test, proba),
        "f1": f1_score(splits.y_test, preds),
    }
    logger.info("Quick metrics on holdout: %s", metrics)

    models_dir = Path(cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.get("output_model_path", f"models/{'rf' if model_type=='random_forest' else 'logreg'}.joblib"))
    joblib.dump(best, out_path)
    with open(models_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": gs.best_params_, "metrics_est": metrics}, f, indent=2)

    # Save config copy
    with open(models_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    logger.info("Saved model to %s", out_path)


if __name__ == "__main__":
    main()

