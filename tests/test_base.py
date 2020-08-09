import comod

import numpy as np
from scipy.integrate import solve_ivp


def test_monomial():
    """Test monomial conversion"""
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
        a, b, c = comod.base._monomial_from_str(s, states, coeffs)

        assert a == coeff
        assert np.array_equal(b, states_exponents)
        assert np.array_equal(c, coeffs_exponents)


def test_SIR():
    """Test the SIR model"""

    t = np.linspace(0, 100, 101)
    initial = (999, 1, 0)
    beta = 0.2
    gamma = 0.1

    N = sum(initial)

    # Manually solve with scipy
    # SIR differential equations
    def deriv(t, y):
        S, I, R = y
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return dS_dt, dI_dt, dR_dt

    solution = solve_ivp(deriv, (t[0], t[-1]), initial, t_eval=t).y

    # Solve with the package interface
    model = comod.Model(list("SIR"), list("bg"), [("S", "I", "b I / N"),
                                                  ("I", "R", "g")],
                        sum_state="N")

    solution2 = model.solve(initial, [beta, gamma], t)

    assert np.allclose(solution, solution2)

    # Test the function-based definition

    def S_I(I=None, N=None, b=None, **kwargs):
        return b * I / N

    def I_R(g=None, **kwargs):
        return g

    model = comod.FunctionModel("SIR",
                                "bg",
                                [
                                    ("S", "I", S_I),
                                    ("I", "R", I_R)
                                ])

    solution3 = model.solve(initial, [beta, gamma], t)
    assert np.allclose(solution, solution3)
