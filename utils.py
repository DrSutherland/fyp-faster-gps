__author__ = 'jyl111'

import fractions

import numpy as np

gcd = np.frompyfunc(fractions.gcd, 2, 1)
