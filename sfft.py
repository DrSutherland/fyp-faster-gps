import utils

import matplotlib.pyplot as plt
import numpy as np

import filters
from fourier_transforms import fft, ifft
from parameters import Parameters

__author__ = 'jyl111'

def execute(params, x, simulation=None):
    print('sFFT filter parameters for n={0}, k={1}'.format(params.n, params.k))

    print('Location filter: (numlobes={numlobes}, tol={tol}, b={b}) B: {B_threshold}/{B}, loops: {loop_threshold}/{loops}'.format(
        numlobes=0.5/params.lobe_fraction_location,
        tol=params.tolerance_location,
        b=params.b_location,
        B_threshold=params.B_threshold,
        B=params.B_location,
        loop_threshold=params.loop_threshold,
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
    assert params.loop_threshold <= params.location_loops

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
    outer_loop_result = outer_loop(
        simulation=simulation,
        params=params,
        x_f=x,
        filter_location=filter_location,
        filter_estimation=filter_estimation
    )

    if simulation is not None:
        x_f = outer_loop_result['x_f']
        answers = outer_loop_result['answers']

        # calculate errors
        x_f_Large = np.zeros(params.n)
        ans_Large = np.zeros(params.n, dtype=np.complex128)

        for location, value in answers.iteritems():
            ans_Large[location] = answers[location]

        x_f_Large[simulation.indices] = simulation.x_f[simulation.indices]


        large_found = 0
        FOUND = 0
        for i in xrange(params.k):
            if simulation.indices[i] in answers.keys():
                FOUND += 1

            if ans_Large[simulation.indices[i]] != 0:
                large_found += 1

        ERROR = 0
        for i in xrange(params.n):
            ERROR += np.abs(ans_Large[i] - x_f_Large[i])

        print('K={k}; MISSED (estimation, result) = ({estimation}, {result}); L1 ERROR = {error} ({error_per} per large freq)'.format(
            k=params.k,
            estimation=params.k-FOUND,
            result=params.k-large_found,
            error=ERROR,
            error_per=ERROR/params.k
        ))

    return x_f


def outer_loop(params, x_f, filter_location, filter_estimation, simulation=None):
    permute = np.empty(params.total_loops)
    permute_b = np.empty(params.total_loops)
    x_samp = []

    # for i in xrange(params.total_loops):
    #     if i < params.location_loops:
    #         x_samp.append(np.zeros(params.B_location, dtype=np.complex128))
    #     else:
    #         x_samp.append(np.zeros(params.B_estimation, dtype=np.complex128))

    hits_found = 0
    hits = np.zeros(params.n)
    scores = np.zeros(params.n)

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

        if perform_location:
            current_filter = filter_location
            current_B = params.B_location
        else:
            current_filter = filter_estimation
            current_B = params.B_estimation

        inner_loop_locate_result = inner_loop_locate(
            x=x_f,
            n=params.n,
            filt=current_filter,
            B=current_B,
            B_threshold=params.B_threshold,
            a=a,
            ai=ai,
            b=b
        )

        x_samp.append(inner_loop_locate_result['x_samp'])
        # assert x_samp[i].shape == inner_loop_locate_result['x_samp'].shape
        # print i, x_samp[i].shape, inner_loop_locate_result['x_samp'].shape
        assert inner_loop_locate_result['J'].size == params.B_threshold

        if perform_location:
            inner_loop_filter_result = inner_loop_filter(
                J=inner_loop_locate_result['J'],
                B=current_B,
                B_threshold=params.B_threshold,
                loop_threshold=params.loop_threshold,
                n=params.n,
                a=a,
                hits_found=hits_found,
                hits=hits,
                scores=scores
            )

            hits_found = inner_loop_filter_result['hits_found']
            hits = inner_loop_filter_result['hits']
            scores = inner_loop_filter_result['scores']

        # print params.B_threshold, inner_loop_locate_result['J'][0], inner_loop_locate_result['J'][1], hits_found

    print('Number of candidates: {0}'.format(hits_found))

    # Estimate values
    answers = estimate_values(
        hits=hits,
        hits_found=hits_found,
        x_samp=x_samp,
        loops=params.total_loops,
        n=params.n,
        permute=permute,
        B_location=params.B_location,
        B_estimation=params.B_estimation,
        filter_location=filter_location,
        filter_estimation=filter_estimation,
        location_loops=params.location_loops
    )

    # Reconstructed signal
    x_f = np.zeros(params.n)
    for location, value in answers.iteritems():
        print 'got', int(location), np.abs(value)
        x_f[int(location)] = np.abs(value)

    # Counts
    xc = np.zeros(params.n)
    for i in xrange(params.n):
        xc[i] = scores[i] * 1./ params.total_loops

    # debug
    if simulation is not None:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(
            simulation.t, xc, '-x',
            simulation.t, simulation.x_f * params.n, '-x',
            simulation.t, x_f * params.n, '-.x',
        )
        ax.legend(
            (
                'counts',
                'true signal',
                'reconstruction'
            )
        )
        ax.set_xlim(right=simulation.t.shape[-1]-1)

    return {
        'x_f': x_f,
        'answers': answers
    }


def inner_loop_locate(x, n, filt, B, B_threshold, a, ai, b):
    assert n % B == 0

    x_t_samp = np.zeros(B, dtype=np.complex128)

    # Permutate and dot product
    index = b
    for i in xrange(filt['size']):
        x_t_samp[i % B] += x[index] * filt['time'][i]
        index = (index + ai) % n

    x_samp = fft(x_t_samp, n=B)

    # samples = np.empty(B)
    # for i in xrange(B):
    #     samples[i] = np.abs(x_samp[i])
    # np.testing.assert_array_equal(np.abs(x_samp), samples)

    samples = np.empty(B)
    for i in xrange(B):
        # samples[i] = np.square(np.abs(x_samp[i]))
        samples[i] = x_samp[i].real*x_samp[i].real + x_samp[i].imag*x_samp[i].imag

    J = np.argsort(samples)[::-1][:B_threshold]

    np.testing.assert_array_equal(samples.take(J), np.sort(samples)[::-1][:B_threshold])

    debug_inner_loop_locate(
        x=x,
        n=n,
        filt=filt,
        B_threshold=B_threshold,
        B=B,
        a=a,
        ai=ai,
        b=b,
        J=J,
        samples=samples
    )

    return {
        'x_samp': x_samp,
        'J': J
    }


def inner_loop_filter(J, B, B_threshold, n, a, loop_threshold, hits_found, hits, scores):
    for i in xrange(B_threshold):
        low = (int(np.ceil((J[i] - 0.5) * n / B)) + n) % n
        high = (int(np.ceil((J[i] + 0.5) * n / B)) + n) % n
        loc = (low * a) % n

        # print 'low', low, 'high', high, 'loc', loc

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


def debug_inner_loop_locate(x, n, filt, B_threshold, B, a, ai, b, J, samples):
    np.testing.assert_array_equal(samples.take(J), np.sort(samples)[::-1][:B_threshold])

    x_f = np.empty(n, dtype=np.complex128)
    pxdotg = np.zeros(n, dtype=np.complex128)
    # pxdotgn = np.empty(n, dtype=np.complex128)
    # pxdotgw = np.empty(n, dtype=np.complex128)

    fft_large = np.array([])
    fft_large_t = np.array([])

    index = b
    for i in xrange(n):
        x_f[i] = x[index]
        index = (index + ai) % n

    w = filt['size']
    pxdotg[:w] = x_f[:w]
    for i in xrange(w):
        pxdotg[i] *= filt['time'][i]

    print('Using {0}x ({1}^-1)'.format(a, ai))

    x_f = fft(x_f, n=n)
    pxdotgn = fft(pxdotg, n=n)
    pxdotgw = fft(pxdotg, n=w)

    t_n = np.linspace(0, 1, num=n, endpoint=False)
    t_w = np.linspace(0, 1, num=w, endpoint=False)
    t_B = np.linspace(0, 1, num=B, endpoint=False)

    for i in xrange(B_threshold):
        np.testing.assert_approx_equal(J[i] / float(B),  t_B[J[i]])

        if J[i]:
            fft_large_t = np.append(fft_large_t, (J[i] - 0.1) / float(B))
            fft_large = np.append(fft_large, 0)

        fft_large_t = np.append(fft_large_t, J[i] / float(B))
        # fft_large_t = np.append(fft_large_t, t_B[J[i]])
        # fft_large = np.append(fft_large, np.sqrt(samples[J[i]]))
        fft_large = np.append(fft_large, np.sqrt(samples[J[i]]) * n)
        # fft_large = np.append(fft_large, 1)

        if J[i] < B - 1:
            fft_large_t = np.append(fft_large_t, (J[i] + 0.1) / float(B))
            fft_large = np.append(fft_large, 0)

    print fft_large_t
    print fft_large

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(
        t_n, np.abs(pxdotgn) * n, '-x',
        t_w, np.abs(pxdotgw) * n, '-x',
        t_B, np.sqrt(samples) * n, '-x',
        fft_large_t, fft_large, '-x',
        t_n, np.abs(x_f), '-x'
    )
    ax.legend(
        (
            'n-dim convolved FFT',
            'w-dim convolved FFT',
            'sampled convolved FFT',
            'largest in sample',
            'true FFT'
        )
    )

    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(self.t, self.x.real, self.t, self.x.imag)
    # ax.legend(('Real', 'Imaginary'))


def estimate_values(hits, hits_found, x_samp, loops, n, permute, B_location, B_estimation, filter_location, filter_estimation, location_loops):
    answers = {}
    reals = np.zeros(loops)
    imags = np.zeros(loops)

    for i in xrange(hits_found):
        for j in xrange(loops):
            if j < location_loops:
                current_B = B_location
                current_filter = filter_location
            else:
                current_B = B_estimation
                current_filter = filter_estimation

            permuted_index = (permute[j] * hits[i]) % n
            hashed_to = permuted_index / (n / current_B)
            dist = permuted_index % (n / current_B)

            if dist > (n / current_B) / 2:
                hashed_to = (hashed_to + 1) % current_B
                dist -= n / current_B

            dist = (n - dist) % n
            filter_value = current_filter['freq'][dist]

            reals[j] = (x_samp[j][hashed_to] / filter_value).real
            imags[j] = (x_samp[j][hashed_to] / filter_value).imag

        location = (loops - 1) / 2

        # nth_element?
        print reals

        reals = np.sort(reals)
        imags = np.sort(imags)

        print reals

        answers[hits[i]] = reals[location] + 1j * imags[location]

    return answers


def main():
    # ./experiment -N 1024 -K 1 -B 4 -E 2 -L 8 -l 5 -r 4 -t 1e-8 -e 1e-8
    params = Parameters(
        n=1024,
        k=1,
        B_k_location=4,
        B_k_estimation=2,
        estimation_loops=8,
        location_loops=5,
        loop_threshold=4,
        tolerance_location=1e-8,
        tolerance_estimation=1e-8,
        snr=0.5
    )

    from simulation import Simulation

    sim = Simulation(params=params)
    sim.plot()

    execute(params=params, x=sim.x, simulation=sim)

    plt.show()


if __name__ == '__main__':
    main()
