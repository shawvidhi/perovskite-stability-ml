import numpy as np

from perostab.labeling import stability_probability


def test_stability_probability_bounds_and_sense():
    # In-window parameters should yield higher probability
    p_good = stability_probability(t=0.9, mu=0.57, delta_chi_bx=0.8, delta_chi_ax=1.2)
    p_bad = stability_probability(t=1.2, mu=0.3, delta_chi_bx=1.5, delta_chi_ax=0.2)
    assert 0.0 < p_good < 1.0
    assert 0.0 < p_bad < 1.0
    assert p_good > p_bad + 0.2

