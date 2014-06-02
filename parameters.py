__author__ = 'jyl111'

import numpy as np


class Parameters:
    def __init__(self,
                 n=2**10,
                 k=10,
                 snr=np.inf,
                 repeat=1,
                 B_k_location=2,
                 B_k_estimation=0.2,
                 location_loops=4,
                 estimation_loops=16,
                 threshold_loops=2,
                 tolerance_location=1e-6,
                 tolerance_estimation=1e-8
                 ):
        """

        :param n: Signal size
        :param k: Signal sparsity
        :param snr: SNR ratio
        :param repeat: Number of times to repeat the experiment and the result averaged
        :param B_k_location
        :param B_k_estimation
        :param location_loops: Number of iterations to find the locations of large frequency bins
        :param estimation_loops: Number of iterations to find the values of large frequency bins
        :param threshold_loops
        :param tolerance_location
        :param tolerance_estimation
        """
        self.n = n
        self.k = k
        self.snr = snr
        self.repeat = repeat
        self.B_k_location = B_k_location
        self.B_k_estimation = B_k_estimation
        self.location_loops = location_loops
        self.estimation_loops = estimation_loops
        self.threshold_loops = threshold_loops
        self.tolerance_location = tolerance_location
        self.tolerance_estimation = tolerance_estimation

    @property
    def total_loops(self):
        return self.location_loops + self.estimation_loops

    @property
    def B_location(self):
        return self.B_k_location * np.sqrt((self.n*self.k)/np.log(self.n))

    @property
    def B_estimation(self):
        return self.B_k_estimation * np.sqrt((self.n*self.k)/np.log(self.n))
