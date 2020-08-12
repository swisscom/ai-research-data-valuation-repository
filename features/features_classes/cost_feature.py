"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import numpy as np
from .feature import Feature


class CostMargin(Feature):
    r"""
    A Feature that extracts cost features (related to language model and acoustic model)

    Attributes:
        cost_diff : an array of cost margin values.
    """
    def __init__(self):
        self.cost_diff = []

    def augment(self, accost, lmcost):
        """
        Computes the metrics for the current samples and saves them in class attributes

        Parameters:
            accost: array-like of size nbest - nbest being the n-best possible hypothesis
            lmcost: array-like of size nbest - nbest being the n-best possible hypothesis
        """
        # Compute the score
        cost = [np.exp(s.astype(np.float128)) for s in -(0.1*accost + lmcost)]
        # Normalize by the sum
        cost_normalized = np.array([e/np.sum(e) for e in cost])
        cost_normalized = np.array([np.sort(x) for x in cost_normalized])
        self.cost_diff = np.array([(p[-1] - p[-2]) if len(p) >= 2 else
                                   p[-1] for p in cost_normalized])

    def get_features(self):
        """
        A getter for costmargin

        Returns:
            array-like of shape (n_samples, )
        """
        return np.array(np.expand_dims(self.cost_diff, axis=1).T)
