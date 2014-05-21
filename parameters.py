__author__ = 'jyl111'

import numpy as np


class Parameters:
    def __init__(self,
                 n=2**10,
                 k=10,
                 snr=np.inf,
                 location_loops=4,
                 estimation_loops=16,
                 ):
        """

        :param n: Signal size
        :param k: Signal sparsity
        :param snr: SNR ratio
        :param location_loops: Number of iterations to find the locations of large frequency bins
        :param estimation_loops: Number of iterations to find the values of large frequency bins
        """
        self.n = n
        self.k = k
        self.snr = snr
        self.location_loops = location_loops
        self.estimation_loops = estimation_loops

    @property
    def total_loops(self):
        return self.location_loops + self.estimation_loops
