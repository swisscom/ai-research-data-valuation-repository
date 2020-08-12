"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from .feature import Feature
import numpy as np

NBEST = 5


class PurityFeats(Feature):
    r"""
    A Feature that extracts purity features (from the n-best hypothesis)

    Attributes:
        one_to_one : an array of avergae one-to-one value
    """
    def __init__(self):
        self.one_to_one = np.empty((0, NBEST))

    def augment(self, nbest_dict):
        """
        Computes the metrics for the current samples and saves them in class attributes

        Parameters:
            nbest_dict: dict-like object, keys are sentences ids, values of sentences.
        """
        one_on_one_l = []
        one_on_one_array = []
        for xid in nbest_dict:
            list_words = xid['1'].strip().split(' ')
            one_to_one_xid = []
            for key in xid:
                if key == '1':
                    one_to_one_xid.append([1] * len(list_words))
                else:
                    purity_per_sentence = []
                    for i in range(0, len(list_words)):
                        try:
                            purity_per_sentence.append(int(list_words[i] == xid[key].strip().split(' ')[i]))
                        except IndexError:
                            purity_per_sentence.append(0)
                    one_to_one_xid.append(purity_per_sentence)

            one_on_one_l.append(np.array(one_to_one_xid))
        one_on_one_array = np.array([np.mean(s, axis=1) for s in one_on_one_l])

        for i in range(0, len(one_on_one_array)):
            while len(one_on_one_array[i]) < NBEST:
                one_on_one_array[i] = np.append(one_on_one_array[i], np.array([1]), axis=0)
            one_on_one_array[i] = np.expand_dims(one_on_one_array[i], axis=1)
            self.one_to_one = np.concatenate([self.one_to_one, one_on_one_array[i].T], axis=0)

    def get_features(self):
        """
        A getter for purity_features

        Returns:
            array-like of shape (n_samples, )
        """
        return np.expand_dims(self.one_to_one, axis=0)
