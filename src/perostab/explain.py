from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import build_features_from_csv
from .utils.logger import get_logger


logger = get_logger(__name__)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def get_base_estimator(model):
    # Pipeline('base' -> inner pipeline ('scale','clf'), 'cal' -> calibrator)
    base_pipe = None
    clf = None
    if hasattr(model, "named_steps"):
        if "base" in model.named_steps:
            base_pipe = model.named_steps["base"]
            if hasattr(base_pipe, "named_steps") and "clf" in base_pipe.named_steps:
                clf = base_pipe.named_steps["clf"]
        elif "clf" in model.named_steps:
            clf = model.named_steps["clf"]
    elif hasattr(model, "base_estimator"):
        clf = model.base_estimator
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

    is_rf = clf.__class__.__name__.startswith("RandomForest")

    if is_rf:
        # RF does not need scaling
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)[1]  # class 1
        shap.summary_plot(
            shap_values, X, feature_names=feature_names, show=False, plot_type="bar"
        )
        save_fig(Path("reports/figures/shap_summary_bar.png"))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        save_fig(Path("reports/figures/shap_beeswarm.png"))
    else:
        # LogisticRegression: transform numeric features then explain linear model on transformed space
        if base_pipe is None or "scale" not in base_pipe.named_steps:
            raise RuntimeError("Expected scaling step for logistic regression pipeline")
        scaler = base_pipe.named_steps["scale"]
        Xtr = scaler.transform(X)
        try:
            feat_out = list(scaler.get_feature_names_out())  # type: ignore[attr-defined]
        except Exception:
            feat_out = feature_names
        explainer = shap.LinearExplainer(clf, Xtr)
        shap_values = explainer.shap_values(Xtr)
        shap.summary_plot(shap_values, Xtr, feature_names=feat_out, show=False, plot_type="bar")
        save_fig(Path("reports/figures/shap_summary_bar.png"))
        shap.summary_plot(shap_values, Xtr, feature_names=feat_out, show=False)
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
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(X.iloc[[idx]])[1][0]
                base_val = expl.expected_value[1]
            else:
                sv = shap_values[idx]
                base_val = explainer.expected_value  # type: ignore[attr-defined]
            shap.plots._waterfall.waterfall_legacy(
                base_val,
                sv,
                feature_names=(feature_names if is_rf else feat_out),
                show=False,
            )
            save_fig(Path(f"reports/figures/shap_waterfall_{tag}.png"))
        except Exception as e:
            logger.warning("Failed waterfall for %s: %s", tag, e)


if __name__ == "__main__":
    main()
