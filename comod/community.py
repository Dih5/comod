from .base import Model

import re
import numpy as np


def _full_adjacency(n):
    """Get the adjacency matrix of a complete graph"""
    return np.ones((n, n)) - np.identity(n)


def _linear_adjacency(n):
    """Get the adjacency matrix of a linear graph"""
    return np.diag(np.ones([n - 1]), k=-1) + np.diag(np.ones([n - 1]), k=1)


def _cycle_adjacency(n):
    """Get the adjacency matrix of a cycle graph"""
    if n == 1:
        return np.asarray([[0.0]])
    m = np.diag(np.ones([n - 1]), k=-1) + np.diag(np.ones([n - 1]), k=1)
    m[0][-1] = 1
    m[-1][0] = 1
    return m


def _star_adjacency(n):
    """Get the adjacency matrix of a star graph"""
    m = np.zeros((n, n))
    m[0] = np.ones(n)
    m[:, 0] = np.ones(n)
    m[0, 0] = 0
    return m


_graph_models = {"full": _full_adjacency, "linear": _linear_adjacency, "cycle": _cycle_adjacency,
                 "star": _star_adjacency}


class CommunityModel(Model):

    def __init__(self, base_model, communities, topology=None, exchange_term="m", equal_parameters=False,
                 symmetric_matrix=False,
                 labels=None):
        """

        Args:
            base_model (Model): A base model used for each of the communities.
            communities (int or list of list): The number of communities or a matrix describing their connections.
            topology (str): A model to use to define community relationships when communities is an integer.
                         Available values are:
                         - "full": All communities are connected (complete graph model $K_n$).
                         - "linear": Linear connection of communities (path graph model $P_n$).
                         - "cycle": Cyclical connection of communities (cycle graph model $C_n$).
                         - "star": One community (the first one) is connected to the rest¸ with no other connection
                                   (star graph model $S_n=K_{1,n-1}$).
            exchange_term (str): Name of the parameter associated to intercommunity transitions.
            equal_parameters (bool): Whether to assume the epidemiological parameters do not depend on the community.
            symmetric_matrix (bool): Whether to assume the community transition matrix is symmetric.
            labels (list of str): List with the names of the communities. Defaults to [1, 2, ..., n].

        """
        if isinstance(communities, int):
            if topology is None:
                topology = "full"
            try:
                communities = _graph_models[topology](communities)
            except KeyError as e:
                raise ValueError("Invalid model. Must be one the following:\n%s" % ", ".join(_graph_models.keys()))
        n = len(communities)

        community_labels = [str(i) for i in range(1, n + 1)] if labels is None else list(labels)

        assert len(community_labels) == n

        states = base_model.states[:]  # Make a copy
        all_states = sum(([s + "_" + str(i) for s in states] for i in community_labels), [])
        parameters = base_model.parameters[:]  # Make a copy
        if equal_parameters:
            all_parameters = parameters
        else:
            all_parameters = sum(([p + "_" + str(i) for p in parameters] for i in community_labels), [])

        all_rules = []

        # Epidemiological rules
        for i in community_labels:
            for origin, destination, rule in base_model.rules:
                s = rule
                suffix = "_" + str(i)
                for w in states + ([] if equal_parameters else parameters) + [base_model.sum_state]:
                    s = re.sub(r"\b%s\b" % w, w + suffix, s)
                all_rules.append((origin + suffix, destination + suffix, s))

        # Community transition rules
        for i, i_label in enumerate(community_labels):
            for j, j_label in enumerate(community_labels):
                if communities[i][j]:
                    suffix = "_{%s,%s}" % (
                        (i_label, j_label) if not symmetric_matrix else (min(i_label, j_label), max(i_label, j_label)))
                    if exchange_term + suffix not in all_parameters:
                        all_parameters += [exchange_term + suffix]
                    for state in states:
                        all_rules.append((state + "_" + str(i_label), state + "_" + str(j_label),
                                          "%s%s / N_%s" % (exchange_term, suffix, i_label)))

        super().__init__(all_states, all_parameters, all_rules, base_model.sum_state, base_model.nihil_state)
