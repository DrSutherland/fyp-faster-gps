__author__ = 'jyl111'

import utils

import numpy as np
from scipy.fftpack import fft, ifft, fftshift

import filters
from parameters import Parameters


def execute(params, x):
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

    assert params.B_threshold < params.B_location
    assert params.threshold_loops <= params.location_loops

    filter_location_t = filters.generate_dolph_chebyshev(
        lobe_fraction=params.lobe_fraction_location,
        tolerance=params.tolerance_location
    )
    w_location = filter_location_t['w']
    filter_location = filters.make_multiple(
        x=filter_location_t['x'],
        w=w_location,
        n=params.n,
        b=params.b_location
    )

    filter_estimation_t = filters.generate_dolph_chebyshev(
        lobe_fraction=params.lobe_fraction_estimation,
        tolerance=params.tolerance_estimation
    )
    w_estimation = filter_estimation_t['w']
    filter_estimation = filters.make_multiple(
        x=filter_estimation_t['x'],
        w=w_estimation,
        n=params.n,
        b=params.b_estimation
    )

    print('Window size: Location: {0}, Estimation: {1}'.format(w_location, w_estimation))

    filter_noise_location = 0
    filter_noise_estimation = 0

    for i in xrange(10):
        filter_noise_location = np.maximum(
            filter_noise_location,
            np.maximum(
                np.abs(filter_location['freq'][params.n/2+i]),
                np.abs(filter_location['freq'][params.n/2-i])
            )
        )

        filter_noise_estimation = np.maximum(
            filter_noise_estimation,
            np.maximum(
                np.abs(filter_estimation['freq'][params.n/2+i]),
                np.abs(filter_estimation['freq'][params.n/2-i])
            )
        )

    print('Noise in filter: Location: {0}, Estimation: {1}'.format(filter_noise_location, filter_noise_estimation))

    # todo repetitions
    outer_loop(
        params=params,
        x=x,
        filter_location=filter_location,
        filter_estimation=filter_estimation
    )


def outer_loop(params, x, filter_location, filter_estimation):
    permute = np.empty(params.total_loops)
    permute_b = np.empty(params.total_loops)
    x_samp = []

    for i in xrange(params.total_loops):
        if i < params.location_loops:
            x_samp.append(np.zeros(params.B_estimation, dtype=np.complex128))
        else:
            x_samp.append(np.zeros(params.B_location, dtype=np.complex128))

    hits_found = 0
    hits = None
    scores = None

    # Inner loop
    for i in xrange(params.total_loops):
        a = 0
        b = 0

        # GCD test
        # http://en.wikipedia.org/wiki/GCD_test
        while utils.gcd(a, params.n) != 1:
            a = np.random.randint(params.n)
            # print 'check', a, params.n, utils.gcd(a, params.n)
        ai = utils.mod_inverse(a, params.n)

        permute[i] = ai
        permute_b[i] = b

        perform_location = i < params.location_loops
        filter = filter_location if perform_location else filter_estimation
        current_B = params.B_location if perform_location else params.B_estimation

        inner_loop_locate_result = inner_loop_locate(
            x=x,
            filter=filter,
            B=current_B,
            B_threshold=params.B_threshold,
            a=a,
            ai=ai,
            b=b
        )

        assert inner_loop_locate_result['J'].size == params.B_threshold

        if perform_location:
            inner_loop_filter_result = inner_loop_filter(
                J=inner_loop_locate_result['J'],
                B=current_B,
                B_threshold=params.B_threshold,
                loop_threshold=params.threshold_loops,
                n=params.n,
                a=a
            )

            hits_found = inner_loop_filter_result['hits_found']
            hits = inner_loop_filter_result['hits']
            scores = inner_loop_filter_result['scores']


def inner_loop_locate(x, filter, B, B_threshold, a, ai, b):
    n = x.size

    if n % B:
        print('Warning: n is not divisible by B')

    x_t_samp = np.zeros(B, dtype=np.complex128)

    # Permutate and dot product
    idx = b
    for i in xrange(filter['size']):
        x_t_samp[i % B] += x[idx] * filter['time'][i]
        idx = (idx + ai) % n

    x_samp = fft(x_t_samp)

    samples = np.empty(B)
    for i in xrange(B):
        samples[i] = np.abs(x_samp[i])

    J = np.argsort(samples)[:B_threshold]

    return {
        'x_samp': x_samp,
        'J': J
    }


def inner_loop_filter(J, B, B_threshold, n, a, loop_threshold):
    hits_found = 0
    hits = np.empty(n)
    scores = np.zeros(n)

    for i in xrange(B_threshold):
        low = (int(np.ceil((J[i] - 0.5) * n / B)) + n) % n
        high = (int(np.ceil((J[i] + 0.5) * n / B)) + n) % n
        loc = (low * a) % n

        j = low
        while j != high:
            scores[loc] += 1

            if scores[loc] == loop_threshold:
                hits[hits_found] = loc
                hits_found += 1

            loc = (loc + a) % n

            j = (j + 1) % n

    return {
        'hits_found': hits_found,
        'hits': hits,
        'scores': scores
    }
  # for(int i = 0; i < num; i++){
  #   int low, high;
  #   low = (int(ceil((J[i] - 0.5) * n / B)) + n)%n;
  #   high = (int(ceil((J[i] + 0.5) * n / B)) + n)%n;
  #   int loc = timesmod(low, a, n);
  #   for(int j = low; j != high; j = (j + 1)%n) {
  #     score[loc]++;
  #     if(score[loc]==loop_threshold)
  #       hits[hits_found++]=loc;
  #     loc = (loc + a)%n;
  #   }
  # }


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

    from simulation import Simulation

    sim = Simulation(params=params)

    execute(params=params, x=sim.x)


if __name__ == '__main__':
    main()
