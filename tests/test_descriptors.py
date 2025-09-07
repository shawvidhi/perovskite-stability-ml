import numpy as np

from perostab.descriptors import goldschmidt_t, octahedral_mu


def test_goldschmidt_t_formula_and_monotonicity():
    t = goldschmidt_t(1.6, 1.0, 2.0)
    assert np.isclose(t, (1.6 + 2.0) / (np.sqrt(2) * (1.0 + 2.0)))

    t1 = goldschmidt_t(1.6, 1.0, 2.0)
    t2 = goldschmidt_t(1.7, 1.0, 2.0)
    assert t2 > t1  # increasing r_A increases t

    t3 = goldschmidt_t(1.6, 0.9, 2.0)
    assert t3 > t1  # decreasing r_B increases t


def test_octahedral_mu_monotonicity():
    mu1 = octahedral_mu(1.0, 2.0)
    mu2 = octahedral_mu(1.1, 2.0)
    assert mu2 > mu1  # increasing r_B increases mu

    mu3 = octahedral_mu(1.0, 2.1)
    assert mu3 < mu1  # increasing r_X decreases mu

