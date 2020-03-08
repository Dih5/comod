import comod

import numpy as np
from scipy.integrate import odeint


def test_monomial():
    """Test monomyial conversion"""
    states = list("$NXY")
    coeffs = list("abc")
    cases = [
        ["", (1, [0, 0, 0], [0, 0, 0])],
        ["1.2", (1.2, [0, 0, 0], [0, 0, 0])],
        ["-2", (-2.0, [0, 0, 0], [0, 0, 0])],
        ["X", (1, [0, 1, 0], [0, 0, 0])],
        ["Y", (1, [0, 0, 1], [0, 0, 0])],
        ["X Y X X", (1, [0, 3, 1], [0, 0, 0])],
        ["1.5 a / X", (1.5, [0, -1, 0], [1, 0, 0])],
        ["a /     X c", (1, [0, -1, 0], [1, 0, 1])],
        ["a/X c", (1, [0, -1, 0], [1, 0, 1])]
    ]
    for s, (coeff, states_exponents, coeffs_exponents) in cases:
        a, b, c = comod._monomial_from_str(s, states, coeffs)

        assert a == coeff
        assert np.array_equal(b, states_exponents)
        assert np.array_equal(c, coeffs_exponents)


def test_SIR():
    """Test the SIR model"""

    t = np.linspace(0, 100, 101)
    initial = (999, 1, 0)
    beta = 0.2
    gamma = 0.1

    # Manually solve with scipy
    # SIR differential equations
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return dS_dt, dI_dt, dR_dt

    solution = odeint(deriv, initial, t, args=(sum(initial), beta, gamma)).T

    # Solve with the package interface
    model = comod.Model(list("SIR"), list("bg"), [("S", "I", "b I / N"),
                                                  ("I", "R", "g")],
                        sum_state="N")

    print(model.rules)
    print(model._rules)

    solution2 = model.solve(initial, [beta, gamma], t)

    assert np.allclose(solution, solution2)
