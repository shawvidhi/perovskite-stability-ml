from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .descriptors import build_feature_frame, goldschmidt_t, octahedral_mu
from .labeling import apply_label_rule

ION_TABLE: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "A": {
        "K": {"r": (1.45, 1.75), "chi": (0.7, 1.0)},
        "Rb": {"r": (1.55, 1.85), "chi": (0.7, 0.9)},
        "Cs": {"r": (1.65, 1.95), "chi": (0.6, 0.9)},
    },
    "B": {
        "Ge": {"r": (0.73, 0.90), "chi": (2.0, 2.3)},
        "Sn": {"r": (0.80, 1.05), "chi": (1.7, 2.0)},
        "Pb": {"r": (0.98, 1.20), "chi": (1.8, 2.3)},
    },
    "X": {
        "Cl": {"r": (1.80, 2.00), "chi": (3.0, 3.3)},
        "Br": {"r": (1.90, 2.10), "chi": (2.7, 3.0)},
        "I": {"r": (2.05, 2.25), "chi": (2.3, 2.7)},
    },
}


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    meta: Dict[str, str]


def sample_from_range(low_high: Tuple[float, float], rng: np.random.Generator) -> float:
    lo, hi = low_high
    return float(rng.uniform(lo, hi))


def generate_synthetic_dataset(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    As = list(ION_TABLE["A"].keys())
    Bs = list(ION_TABLE["B"].keys())
    Xs = list(ION_TABLE["X"].keys())
    rows = []
    for _ in range(n):
        A = rng.choice(As)
        B = rng.choice(Bs)
        X = rng.choice(Xs)
        r_A = sample_from_range(ION_TABLE["A"][A]["r"], rng)
        r_B = sample_from_range(ION_TABLE["B"][B]["r"], rng)
        r_X = sample_from_range(ION_TABLE["X"][X]["r"], rng)
        chi_A = sample_from_range(ION_TABLE["A"][A]["chi"], rng)
        chi_B = sample_from_range(ION_TABLE["B"][B]["chi"], rng)
        chi_X = sample_from_range(ION_TABLE["X"][X]["chi"], rng)

        t = goldschmidt_t(r_A, r_B, r_X)
        mu = octahedral_mu(r_B, r_X)
        d_ax = abs(chi_A - chi_X)
        d_bx = abs(chi_B - chi_X)
        p, y = apply_label_rule(t, mu, d_bx, d_ax, rng)

        rows.append(
            {
                "A": A,
                "B": B,
                "X": X,
                "r_A": r_A,
                "r_B": r_B,
                "r_X": r_X,
                "chi_A": chi_A,
                "chi_B": chi_B,
                "chi_X": chi_X,
                "t": t,
                "mu": mu,
                "p_stable": p,
                "y": y,
            }
        )

    df = pd.DataFrame(rows)
    # Optional slight resampling near decision boundary to balance
    # Keep dataset as generated for simplicity; class balance should be reasonable
    return df


def save_dataset_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features_from_csv(
    path: Path,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    df = load_dataset_csv(path)
    X, meta = build_feature_frame(df)
    y = df["y"].astype(int)
    return X, y, list(X.columns), meta


def train_test_split_features(
    csv_path: Path, test_size: float = 0.2, seed: int = 42
) -> DatasetSplits:
    X, y, feature_names, meta = build_features_from_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return DatasetSplits(X_train, X_test, y_train, y_test, feature_names, meta)


def save_feature_list(feature_names: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names}, f, indent=2)


def get_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
