__author__ = 'jyl111'

import utils

import numpy as np

from parameters import Parameters


def execute(params):
    print('sFFT filter parameters for n={0}, k={1}'.format(params.n, params.k))

    print('Location filter: (numlobes={numlobes}, tol={tol}, b={b}) B: {B_threshold}/{B}, loops: {threshold_loops}/{loops}'.format(
        numlobes=0.5/params.lobe_fraction_location,
        tol=params.tolerance_location,
        b=params.b_location,
        B_threshold=params.B_threshold,
        B=params.B_location,
        threshold_loops=params.threshold_loops,
        loops=params.location_loops
    ))

    print('Estimation filter: (numlobes={numlobes}, tol={tol}, b={b}) B: {B}, loops: {loops}'.format(
        numlobes=0.5/params.lobe_fraction_estimation,
        tol=params.tolerance_estimation,
        b=params.b_estimation,
        B=params.B_estimation,
        loops=params.estimation_loops
    ))

    outer_loop(params)


def outer_loop(params):
    for i in xrange(params.total_loops):
        a = 0
        b = 0

        # GCD test
        # http://en.wikipedia.org/wiki/GCD_test
        while utils.gcd(a, params.n) != 1:
            a = np.random.randint(params.n)
            # print 'check', a, params.n, utils.gcd(a, params.n)

        print a


def main():
    # ./experiment -N 1024 -K 1 -B 4 -E 2 -L 8 -l 5 -r 4 -t 1e-8 -e 1e-8
    params = Parameters(
        n=1024,
        k=1,
        B_k_location=4,
        B_k_estimation=2,
        estimation_loops=8,
        location_loops=5,
        threshold_loops=4,
        tolerance_location=1e-8,
        tolerance_estimation=1e-8
    )

    execute(params)


if __name__ == '__main__':
    main()
