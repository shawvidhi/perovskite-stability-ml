from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SQRT2 = np.sqrt(2.0)


@dataclass
class Ion:
    symbol: str
    charge: int


def goldschmidt_t(r_a: float, r_b: float, r_x: float) -> float:
    return float((r_a + r_x) / (SQRT2 * (r_b + r_x)))


def octahedral_mu(r_b: float, r_x: float) -> float:
    return float(r_b / r_x)


def electronegativity_deltas(chi_a: float, chi_b: float, chi_x: float) -> Tuple[float, float, float]:
    d_ax = abs(chi_a - chi_x)
    d_bx = abs(chi_b - chi_x)
    ratio = d_bx / (d_ax + 1e-6)
    return float(d_ax), float(d_bx), float(ratio)


def is_charge_neutral(a: Ion, b: Ion, x: Ion) -> bool:
    return (a.charge + b.charge + 3 * x.charge) == 0


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Expect columns: r_A, r_B, r_X, chi_A, chi_B, chi_X, A, B, X
    Produces numeric features and one-hot encoding for A,B,X.
    Returns: (X_df, metadata)
    """
    r_a = df["r_A"].astype(float)
    r_b = df["r_B"].astype(float)
    r_x = df["r_X"].astype(float)
    chi_a = df["chi_A"].astype(float)
    chi_b = df["chi_B"].astype(float)
    chi_x = df["chi_X"].astype(float)

    t = goldschmidt_t(r_a, r_b, r_x)
    mu = octahedral_mu(r_b, r_x)
    delta_t = np.abs(t - 0.9)
    delta_mu = np.abs(mu - 0.57)
    d_ax, d_bx, d_ratio = electronegativity_deltas(chi_a, chi_b, chi_x)

    X_num = pd.DataFrame(
        {
            "t": t,
            "mu": mu,
            "delta_t": delta_t,
            "delta_mu": delta_mu,
            "delta_chi_AX": d_ax,
            "delta_chi_BX": d_bx,
            "delta_chi_ratio": d_ratio,
        }
    )

    # One-hot encodings for A, B, X
    cats = pd.get_dummies(df[["A", "B", "X"]].astype(str), prefix=["A", "B", "X"], dtype=int)
    X = pd.concat([X_num, cats], axis=1)

    meta = {"n_numeric": str(X_num.shape[1]), "n_categorical": str(cats.shape[1])}
    return X, meta

