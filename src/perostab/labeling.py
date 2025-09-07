from __future__ import annotations

from typing import Tuple

import numpy as np


def stability_probability(
    t: float,
    mu: float,
    delta_chi_bx: float,
    delta_chi_ax: float,
    alpha: float = 8.0,
    beta: float = 10.0,
    gamma: float = 2.5,
    bias: float = 1.0,
) -> float:
    """
    Smooth logistic over distance to target windows with electronegativity effects.
    Higher prob when 0.80 <= t <= 1.00 and 0.414 <= mu <= 0.732.
    """
    delta_t = abs(t - 0.9)
    delta_mu = abs(mu - 0.57)
    # Penalties/bonuses
    chi_term = gamma * (1.2 - delta_chi_bx) - 0.5 * max(0.0, 1.0 - delta_chi_ax)
    lin = -(alpha * delta_t + beta * delta_mu) + chi_term + bias
    p = 1.0 / (1.0 + np.exp(-lin))
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def sample_label(p: float, rng: np.random.Generator) -> int:
    return int(rng.random() < p)


def apply_label_rule(
    t: float, mu: float, delta_chi_bx: float, delta_chi_ax: float, rng: np.random.Generator
) -> Tuple[float, int]:
    p = stability_probability(t, mu, delta_chi_bx, delta_chi_ax)
    y = sample_label(p, rng)
    return p, y

