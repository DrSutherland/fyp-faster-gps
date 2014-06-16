__author__ = 'jyl111'

import numpy as np


class PerformanceCounter:
    def __init__(self):
        self.additions = 0
        self.multiplications= 0

    def increase(self, additions=0, multiplications=0):
        self.additions += additions
        self.multiplications += multiplications

    def fft(self, n):
        self.additions += int(n * np.log2(n))
        self.multiplications += int((n / 2) * np.log2(n))

    @property
    def flops(self):
        return self.additions + self.multiplications

    def __repr__(self):
        return 'additions = %s, multiplications = %s, flops (total) = %s' % (repr(self.additions), repr(self.multiplications), repr(self.flops))
