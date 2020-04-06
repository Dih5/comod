"""comod - Compartment model Python package"""

__version__ = '0.1.1'
__author__ = 'Dih5 <dihedralfive@gmail.com>'
__all__ = []

import re

import numpy as np
import pandas as pd
from scipy import integrate, optimize
import igraph


def _monomial_from_str(s, states, coeffs):
    """Transform a rule from a Model into a rule of _NumericalModel"""
    dividing = False
    coeff = 1.0
    states_exponents = np.zeros(len(states) - 1, np.int8)
    coeffs_exponents = np.zeros(len(coeffs), np.int8)
    # Ensure / is padded
    s = " / ".join(s.split("/"))
    for token in s.split():
        if token == "/":
            if dividing:
                raise ValueError("Syntax error: Double division in %s" % s)
            dividing = True
        elif token in states:
            i = states.index(token) - 1
            if i == -1:
                raise ValueError("Syntax error: Nihil state cannot be used as coefficient.")
            if dividing:
                states_exponents[i] -= 1
                dividing = False
            else:
                states_exponents[i] += 1
        elif token in coeffs:
            i = coeffs.index(token)
            if dividing:
                coeffs_exponents[i] -= 1
                dividing = False
            else:
                coeffs_exponents[i] += 1
        else:
            value = None
            try:
                value = float(token)
            except ValueError:
                pass
            if value is None:
                raise ValueError("Syntax error: Unknown token %s in %s" % (token, s))
            else:
                coeff *= value
    return coeff, states_exponents, coeffs_exponents


def _window(a, window_size=4, step_size=2, copy=False):
    """
    Get a sliding window of an array

    Args:
        a (np.array): The original array.
        window_size (int): Window size.
        step_size (int): Window step size.
        copy (bool): Whether to return a copy of the array instead of a readonly view.

    Returns:
        np.array: The sliding window.

    """
    sh = (a.size - window_size + 1, window_size)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh, writeable=False)[0::step_size]
    if copy:
        return view.copy()
    else:
        return view


def _sliding_time(t, window_size=4, step_size=2, criterion="last"):
    """Convenience function to extract times to represent a sliding window"""
    _t = _window(t, window_size=window_size, step_size=step_size)
    if criterion == "first":
        return _t[:, 0]
    elif criterion == "last":
        return _t[:, -1]
    elif criterion == "median":
        return _t[:, int(window_size / 2)]
    elif criterion == "mean":
        return np.mean(_t, axis=1)
    else:
        raise ValueError("Invalid criterion")


class _NumericalModel:
    """Mathematical form of a compartment."""

    def __init__(self, n_states, parameters, rules):
        """

        Args:
            n_states (int): Number of states (not including sum or nihil states)
            parameters (list of float): Values for the parameters.
            rules (list of tuples): Tuples of the form (origin, destination, (coeff, degree_states, degree_parameters)),
                                    those being:
                                    - origin: Index of the origin state.
                                    - destination: Index of the destination state.
                                    - coeff: Real number multiplying the coefficient.
                                    - degree_states: Tuple with the degree of the states in the monomial
                                                     (0 being the sum state and -1 the nihil state)
                                    - degree_parameters: Tuple with the degrees of the parameters in the monomial.
        """
        self.n_states = n_states
        self.parameters = parameters
        self.rules = rules

    def __call__(self, y, t, *args):
        dy = np.zeros(self.n_states)
        # [N, Q_1, Q_2, ...]
        y = np.concatenate([[np.sum(y)], y])
        for origin, destination, (coeff, degree_states, degree_parameters) in self.rules:
            # Note this implementation relies on 0.0**0==1.0
            if origin == -1:
                dy[destination - 1] += coeff * np.prod(y ** degree_states) * np.prod(
                    self.parameters ** degree_parameters)
            elif destination == -1:
                dy[origin - 1] -= coeff * np.prod(y ** degree_states) * np.prod(
                    self.parameters ** degree_parameters) * y[
                                      origin]
            else:
                dy[origin - 1] -= coeff * np.prod(y ** degree_states) * np.prod(
                    self.parameters ** degree_parameters) * y[
                                      origin]
                dy[destination - 1] += coeff * np.prod(y ** degree_states) * np.prod(
                    self.parameters ** degree_parameters) * \
                                       y[origin]
        return dy


class Model:
    """A compartment model"""

    def __init__(self, states=None, parameters=None, rules=None, sum_state="N", nihil_state="$"):
        """

        Args:
            states (str or list of str): Names of the states. If str, one letter per state is assumed.
            parameters (str or list of str): Names of the coefficients. If str, one letter per state is assumed.
            rules (list of (str, str, str) tuples): Transition rules defined by origin state, destination state and
                                            multiplicative coefficient. The proportionality on the population of the
                                            origin state is automatically assumed (unless the rule is describing
                                            births). The coefficient might be a product of real numbers, states, and
                                            parameters. Division is allowed, parenthesis are not. Use multiple rules to
                                            define additions or subtractions.
            sum_state (str): Name of a special state with the total population. Can be used in the coefficients.
            nihil_state (str): Name of a special state used to described birth rules (when used as origin) and death
                               rules (when used as destination).

        """

        self.states = list(states) if states is not None else []
        self.parameters = list(parameters) if parameters is not None else []
        self.rules = list(rules) if rules is not None else []

        self.sum_state = sum_state
        self.nihil_state = nihil_state

        all_states = [nihil_state, sum_state] + self.states

        self._rules = [(all_states.index(origin) - 1,
                        all_states.index(destination) - 1,
                        _monomial_from_str(s, [self.nihil_state, self.sum_state] + self.states, self.parameters))
                       for origin, destination, s in self.rules]

    def to_latex(self):
        """
        Get a LaTeX representation of the differential equations of the model.

        Returns:
            str: LaTeX code suitable for use in an equation environment.

        """
        dy = {state: "" for state in self.states}

        for origin, destination, coeff in self.rules:
            # Note this implementation relies on 0.0**0==1.0
            if origin == self.nihil_state:
                dy[destination] += " + %s" % coeff
            elif destination == self.nihil_state:
                dy[origin] += " - %s %s" % (coeff, origin)
            else:
                dy[destination] += " + %s %s" % (coeff, origin)
                dy[origin] += " - %s %s" % (coeff, origin)

        # Replace leading " +" and " -"
        for state in self.states:
            dy[state] = re.sub(r"^ \+ ", "", dy[state])
            dy[state] = re.sub(r"^ - ", "-", dy[state])

        array_code = "\\\\\n".join("\\dot{%s} &=& %s" % (state, dy[state]) for state in self.states)
        return "\\begin{array}{lcl} %s \n\\end{array}" % array_code

    def solve(self, initial, parameters, t):
        """
        Solve the model numerically.

        The solution is found using scipy.integrate.odeint, which uses lsoda from the FORTRAN library odepack.

        Args:
            initial (list of float): Initial population for each of the states.
            parameters (list of float): Values for the parameters.
            t (list of float): Mesh of time values for which the solution is found.

        Returns:

        """
        return integrate.odeint(_NumericalModel(len(self.states), parameters, self._rules), initial, t).T

    def best_fit(self, data, t, initial_pars, target="dy", component_weights=None, ls_kwargs=None):
        """
        Get a best fit of the model to the provided data

        The first point in the data is considered to be defining the initial conditions

        Args:
            data (list of list of float): 2D array whose first index is the component (e.g., S, I, R....) and its
                                          second index is the time.
            t (list of float): Time mesh. Must be consistent with data.
            initial_pars (list of float): Initial values of the parameters.
            target (str): Metric to reproduce. Available options are:
                          - "y": The curves themselves.
                          - "dy": The changes of the curves.
            component_weights (list of float): A set of weights for each component of the model
            ls_kwargs (dict): Additional kwargs to pass to the least squares solver. Cf. scipy.optimize.least_squares.

        Returns:
            scipy.optimize.OptimizeResult: Object describing the result of the fitting. To obtain the parameters access
                                           its x attribute.

        """
        if component_weights is None:
            component_weights = np.ones((len(data),))
        else:
            component_weights = np.asarray(component_weights)
            assert len(component_weights) == len(data)

        if ls_kwargs is None:
            ls_kwargs = {}

        # Time to first index, component to second
        data = np.asarray(data).T

        t = np.asarray(t)

        if target == "dy":
            # Fit to increments
            time_increments = np.reshape(t[1:] - t[:-1], (-1, 1))
            real_increments = (data[1:] - data[:-1]) / time_increments

            def get_residuals(parameters):
                m = _NumericalModel(len(self.states), parameters, self._rules)
                predicted_increments = [m(datum, time) for time, datum in zip(t, data)][:-1]

                return np.reshape(real_increments - predicted_increments, (-1,)) * np.tile(component_weights,
                                                                                           len(data) - 1)
        elif target == "y":
            # Fit to curves
            initials = data[0]

            def get_residuals(parameters):
                predicted_values = self.solve(initials, parameters, t)
                return np.reshape((data.T - predicted_values), (-1,)) * np.repeat(component_weights, len(data))
        else:
            raise ValueError("Invalid method")

        return optimize.least_squares(get_residuals, initial_pars, **ls_kwargs)

    def _best_sliding_fit(self, data, t, initial_pars, window_size, step_size, target="dy", component_weights=None,
                          ls_kwargs=None):
        """Iterator yielding the values for _best_sliding_fit"""
        data = np.asarray(data)

        t = np.asarray(t)

        for data_subset in list(
            zip(*[_window(x, window_size, step_size) for x in data], _window(t, window_size, step_size))):
            yield self.best_fit(data_subset[:-1], data_subset[-1], initial_pars, target=target,
                                component_weights=component_weights, ls_kwargs=ls_kwargs).x

    def best_sliding_fit(self, data, t, initial_pars, window_size, step_size, target="dy", component_weights=None,
                         ls_kwargs=None, time_criterion="last"):
        """
        Get a best fit of the model to the provided data

        The first point in the data is considered to be defining the initial conditions

        Args:
            data (list of list of float): 2D array whose first index is the component (e.g., S, I, R....) and its
                                          second index is the time.
            t (list of float): Time mesh. Must be consistent with data.
            initial_pars (list of float): Initial values of the parameters.
            target (str): Metric to reproduce. Available options are:
                          - "y": The curves themselves.
                          - "dy": The changes of the curves.
            component_weights (list of float): A set of weights for each component of the model
            ls_kwargs (dict): Additional kwargs to pass to the least squares solver. Cf. scipy.optimize.least_squares.
            time_criterion (str): Criterion used to asign a time value for each of the windows. Available values are:
                                  - "last": The last time in the window.
                                  - "first": The first time in the window.
                                  - "median": The median time of the points in the window.
                                  - "mean": The mean time of the points in the window.

        Returns:
            pd.DataFrame: A dataframe with the fits in each of the windows.

        """
        values = np.asarray(list(self._best_sliding_fit(data, t, initial_pars, window_size, step_size, target=target,
                                                        component_weights=component_weights, ls_kwargs=ls_kwargs)))
        t2 = _sliding_time(t, window_size=window_size, step_size=step_size, criterion=time_criterion)

        return pd.DataFrame(values, index=t2, columns=self.parameters)

    def plot_graph(self, **kwargs):
        """
        Get a plot of a graph representing the model using igraph.

        Args:
            **kwargs: Additional arguments to pass to igraph.plot.

        Returns:
            igraph.Plot: The plot of the graph.

        """
        g = igraph.Graph(directed=True)
        g.add_vertices([self.nihil_state] + self.states)
        g.add_edges([r[:2] for r in self.rules])
        g.es["label"] = [r[2] for r in self.rules]

        if "vertex_label" not in kwargs:
            kwargs["vertex_label"] = [""] + self.states
        if "vertex_color" not in kwargs:
            kwargs["vertex_color"] = ["red"] + ["blue"] * len(self.states)

        return igraph.plot(g, **kwargs)
