__author__ = 'jyl111'

import utils

import numpy as np


def execute(params):
    outer_loop(params)

def outer_loop(params):
    for i in xrange(params.total_loops):
        a = 0
        b = 0

        # GCD test
        # http://en.wikipedia.org/wiki/GCD_test
        while utils.gcd(a, params.n) != 1:
            a = np.random.randint(params.n)
            print 'check', a, params.n, utils.gcd(a, params.n)

        print a
