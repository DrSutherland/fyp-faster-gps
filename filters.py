import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from fourier_transforms import fft, ifft, fftshift
from parameters import Parameters

__author__ = 'jyl111'


def generate_gaussian(lobe_fraction, tolerance):
    w = int((2 / np.pi) * (1 / lobe_fraction) * np.log(1 / tolerance))

    print('lobe_fraction = {0}, tolerance = {1}, w = {2}'.format(lobe_fraction, tolerance, w))

    std = (w / 2.0) / np.sqrt(2 * np.log(1 / tolerance))

    return signal.gaussian(
        M=w,
        std=std
    ).reshape((-1, 1))


def generate_gaussian_original(lobe_fraction, tolerance):
    w = int((2 / np.pi) * (1 / lobe_fraction) * np.log(1 / tolerance))

    print('lobe_fraction = {0}, tolerance = {1}, w = {2}'.format(lobe_fraction, tolerance, w))

    std = (w / 2.0) / np.sqrt(2 * np.log(1 / tolerance))
    center = w / 2.0

    output = np.empty((w, 1), dtype=np.complex128)

    for i in xrange(w):
        distance = np.abs(i - center)
        output[i] = np.exp(-(distance*distance) / (2 * std * std))

    return output


def generate_dolph_chebyshev(lobe_fraction, tolerance):
    def cheb(m, x):
        if np.abs(x) <= 1:
            return np.cos(m * np.arccos(x))
        else:
            return np.cosh(m * np.arccosh(np.abs(x))).real

    w = int((1 / np.pi) * (1 / lobe_fraction) * np.arccosh(1 / tolerance))

    if w % 2 == 0:
        w -= 1

    beta = np.cosh(np.arccosh(1 / tolerance) / float(w - 1))

    print('lobe_fraction = {0}, tolerance = {1}, w = {2}, beta = {3}'.format(lobe_fraction, tolerance, w, beta))

    x = np.empty(w, dtype=np.complex128)

    for i in xrange(w):
        x[i] = cheb(w - 1, beta * np.cos(np.pi * i / w)) * tolerance

    x = fft(x, n=w)
    x = fftshift(x)
    x = np.real(x)

    return {
        'x': x,
        'w': w
    }


def make_multiple(x, w, n, b):
    print('w = {0}, n = {1}, b = {2}'.format(w, n, b))

    assert b <= n
    assert w <= n

    g = np.zeros(n, dtype=np.complex128)
    h = np.zeros(n, dtype=np.complex128)

    g[0:w-(w/2)] = x[(w/2):]
    g[n-(w/2):] = x[:(w/2)]

    g = fft(g, n=n)

    s = 0
    for i in xrange(b):
        s += g[i]

    max = 0
    offset = int(b/2)


    for i in xrange(n):
        h[(i+n+offset)%n] = s
        max = np.maximum(max, np.abs(s))
        s += (g[(i+b)%n]-g[i])

    h /= max

    offsetc = 1
    step = np.exp(-2 * np.pi * 1j * (w / 2) / n)

    for i in xrange(n):
        h[i] *= offsetc
        offsetc *= step

    x = ifft(h, n=n)

    return {
        'time': x,
        'size': w,
        'freq': h
    }

def main():
    def display_gaussian():
        output = generate_gaussian(
            lobe_fraction=0.0125,
            tolerance=0.00000001
        )

        plt.figure()
        plt.plot(output)
        plt.title('Gaussian window')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample')

    def display_dolph_chebyshev():
        output = generate_dolph_chebyshev(
            lobe_fraction=0.0125,
            tolerance=0.00000001
        )

        plt.figure()
        plt.plot(output['x'])
        plt.title('Dolph-Chebyshev window')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample')

        output = make_multiple(
            x=output['x'],
            w=output['w'],
            n=1024,
            b=33
        )

        plt.figure()
        plt.plot(output['time'])
        plt.title('Dolph-Chebyshev window multiple')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample')

    # display_gaussian()
    display_dolph_chebyshev()
    plt.show()


if __name__ == '__main__':
    main()
