from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from .dataset import build_features_from_csv
from .utils.logger import get_logger

logger = get_logger(__name__)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def get_base_estimator(model):
    # Return (pipeline, underlying classifier) unwrapping CalibratedClassifierCV if present
    base_pipe = model if hasattr(model, "named_steps") else None
    clf = None
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        step = model.named_steps["clf"]
        try:
            # CalibratedClassifierCV exposes .estimator
            if hasattr(step, "estimator") and step.estimator is not None:
                clf = step.estimator
            else:
                clf = step
        except Exception:
            clf = step
    return base_pipe, clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for a saved model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-csv", type=str, default="data/synthetic/abx3_synthetic.csv")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    try:
        import shap  # type: ignore
    except Exception as e:
        raise SystemExit(f"shap is required for explanations: {e}")

    model = joblib.load(args.model_path)
    X_all, y_all, feature_names, _ = build_features_from_csv(Path(args.data_csv))

    n = min(args.n_samples, len(X_all))
    X = X_all.iloc[:n]
    y = y_all.iloc[:n]

    base_pipe, clf = get_base_estimator(model)
    if clf is None:
        raise RuntimeError("Could not locate base classifier inside the model pipeline for SHAP")

    # Try to access the calibrator object for fitted estimators
    calibrator = None
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        calibrator = model.named_steps["clf"]

    is_rf = clf.__class__.__name__.startswith("RandomForest")

    if is_rf:
        # RF: use model-agnostic unified API for stability across SHAP versions
        rf_est = None
        if calibrator is not None and hasattr(calibrator, "calibrated_classifiers_"):
            cals = calibrator.calibrated_classifiers_
            if cals:
                rf_est = getattr(cals[0], "estimator", None)
        if rf_est is None:
            rf_est = clf
        explainer = shap.Explainer(rf_est, X, feature_names=feature_names)
        sv = explainer(X)
        shap.plots.bar(sv, show=False)
        save_fig(Path("reports/figures/shap_summary_bar.png"))
        shap.plots.beeswarm(sv, show=False)
        save_fig(Path("reports/figures/shap_beeswarm.png"))
    else:
        # LogisticRegression: explain the fitted linear model on transformed space
        if (
            base_pipe is None
            or not hasattr(base_pipe, "named_steps")
            or "scale" not in base_pipe.named_steps
        ):
            raise RuntimeError("Expected scaling step for logistic regression pipeline")
        scaler = base_pipe.named_steps["scale"]
        Xtr = scaler.transform(X)
        try:
            feat_out = list(scaler.get_feature_names_out())  # type: ignore[attr-defined]
        except Exception:
            feat_out = feature_names
        # Use fitted base estimator from calibrator if available
        lr_est = None
        if calibrator is not None and hasattr(calibrator, "calibrated_classifiers_"):
            cals = calibrator.calibrated_classifiers_
            if cals:
                lr_est = getattr(cals[0], "estimator", None)
        if lr_est is None:
            lr_est = clf
        explainer = shap.Explainer(lr_est, Xtr, feature_names=feat_out)
        sv = explainer(Xtr)
        shap.plots.bar(sv, show=False)
        save_fig(Path("reports/figures/shap_summary_bar.png"))
        shap.plots.beeswarm(sv, show=False)
        save_fig(Path("reports/figures/shap_beeswarm.png"))

    # Local waterfalls for three representative samples: stable, unstable, borderline
    proba = model.predict_proba(X)[:, 1]
    idx_stable = int(np.argmax(y.values)) if y.sum() > 0 else 0
    idx_unstable = int(np.argmin(y.values))
    idx_borderline = int(np.argmin(np.abs(proba - 0.5)))

    examples = [(idx_stable, "stable"), (idx_unstable, "unstable"), (idx_borderline, "borderline")]
    for idx, tag in examples:
        try:
            if is_rf:
                shap.plots.waterfall(sv[idx], show=False)
            else:
                shap.plots.waterfall(sv[idx], show=False)
            save_fig(Path(f"reports/figures/shap_waterfall_{tag}.png"))
        except Exception as e:
            logger.warning("Failed waterfall for %s: %s", tag, e)


if __name__ == "__main__":
    main()
