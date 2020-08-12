"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import numpy as np
MAX_LEN = 180


class Parser:
    r"""
    A Helper that parses the output text files from KALDI and outputs a zero-padded
    logit array.

    Attributes:
        logit : an array of logits.
    """
    def __init__(self):
        self.logit = []

    def parse(self, logit_string):
        """
        Parses the logit_string

        Parameters:
            logit_string: string of logits
        """
        logit_init = []
        for xid in logit_string:
            for word in [xid]:
                S = word.strip().split('[')
                parsed_S = [''.join(s.strip().split(']')).strip().split(' ') for s in S]
                len_seq = []
                for prob_pairs in parsed_S[1:]:
                    seq = []
                    if len(prob_pairs) == 2:
                        seq.append(float(prob_pairs[1]))
                        seq.append(0)
                    elif len(prob_pairs) >= 4:
                        seq.append(float(prob_pairs[1]))
                        seq.append(float(prob_pairs[3]))
                    len_seq.append(np.array(seq))

            logit_init.append(np.array(len_seq))
        logit_init = np.expand_dims(logit_init, axis=1)
        self.zeropad(np.array(logit_init))

    def zeropad(self, logit_init):
        """
        Zero pads the logits if smaller than MAX_LEN, otherwise take first MAX_LEN values.
        """
        for logit in logit_init:
            if np.shape(logit[0])[0] > MAX_LEN:
                logit[0] = logit[0][:MAX_LEN]
        self.logit = np.array([np.pad(a[0], [(0, (MAX_LEN - a[0].shape[0])), (0, 0)],
                                      mode='constant', constant_values=0) for a in logit_init])

    def get_logits(self):
        """
        A getter for logits

        Returns
            array-like of shape (n_samples, maxlen)
        """
        return self.logit
