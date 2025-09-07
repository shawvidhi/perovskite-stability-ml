from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .dataset import build_features_from_csv, load_dataset_csv, train_test_split_features
from .utils.logger import get_logger


logger = get_logger(__name__)


def bootstrap_ci(
    vec_true: np.ndarray, vec_score: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(vec_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(vec_true[idx], vec_score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def save_plot(fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on test set")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-csv", type=str, default="data/synthetic/abx3_synthetic.csv")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    csv_path = Path(args.data_csv)
    splits = train_test_split_features(csv_path, test_size=0.2, seed=42)

    proba = model.predict_proba(splits.X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(splits.y_test, proba),
        "average_precision": average_precision_score(splits.y_test, proba),
        "f1": f1_score(splits.y_test, preds),
        "brier": brier_score_loss(splits.y_test, proba),
    }
    logger.info("Test metrics: %s", metrics)

    # Save metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Bootstrap CIs for ROC-AUC and F1
    y_np = splits.y_test.values
    auc_lo, auc_hi = bootstrap_ci(y_np, proba, roc_auc_score, n_boot=1000, seed=42)
    f1_preds = (proba >= 0.5).astype(int)
    f1_lo, f1_hi = bootstrap_ci(y_np, f1_preds, f1_score, n_boot=1000, seed=42)
    with open(reports_dir / "metrics_with_ci.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **metrics,
                "roc_auc_ci": [auc_lo, auc_hi],
                "f1_ci": [f1_lo, f1_hi],
            },
            f,
            indent=2,
        )

    # ROC curve
    RocCurveDisplay.from_predictions(splits.y_test, proba)
    save_plot(Path("reports/figures/roc_curve.png"))

    # PR curve
    PrecisionRecallDisplay.from_predictions(splits.y_test, proba)
    save_plot(Path("reports/figures/pr_curve.png"))

    # Confusion matrix
    cm = confusion_matrix(splits.y_test, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(colorbar=False)
    save_plot(Path("reports/figures/confusion_matrix.png"))

    # Calibration
    frac_pos, mean_pred = calibration_curve(splits.y_test, proba, n_bins=10)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction positive")
    plt.legend()
    save_plot(Path("reports/figures/calibration_curve.png"))

    # Decision space on t, mu
    full_df = load_dataset_csv(csv_path)
    X_all, _, feature_names, _ = build_features_from_csv(csv_path)
    all_proba = model.predict_proba(X_all)[:, 1]
    plt.figure()
    plt.scatter(full_df["t"], full_df["mu"], c=all_proba, cmap="viridis", s=10, alpha=0.7)
    plt.xlabel("t")
    plt.ylabel("mu")
    plt.colorbar(label="Predicted stability prob")
    save_plot(Path("reports/figures/decision_space_t_mu.png"))

    # Slice metrics by X and B (test set only)
    df_test = full_df.iloc[splits.y_test.index]
    for col, fname in [("X", "slice_metrics_by_X.png"), ("B", "slice_metrics_by_B.png")]:
        gvals = []
        for g in df_test[col].unique():
            m = (df_test[col] == g).values
            y_g = splits.y_test.values[m]
            p_g = proba[m]
            if len(y_g) < 5:
                continue
            auc_g = roc_auc_score(y_g, p_g)
            f1_g = f1_score(y_g, (p_g >= 0.5).astype(int))
            gvals.append((g, auc_g, f1_g))
        if not gvals:
            continue
        labels, aucs, f1s = zip(*gvals)
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(6, 3))
        plt.bar(x - width / 2, aucs, width, label="ROC-AUC")
        plt.bar(x + width / 2, f1s, width, label="F1")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.title(f"Slice metrics by {col}")
        save_plot(Path(f"reports/figures/{fname}"))


if __name__ == "__main__":
    main()
