import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

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
            return np.cosh(m * np.arccosh(x)).real

    w = int((1 / np.pi) * (1 / lobe_fraction) * np.log(1 / tolerance))

    if w % 2 == 0:
        w -= 1

    beta = np.cosh(np.arccosh(1 / tolerance) / float(w - 1))

    print('lobe_fraction = {0}, tolerance = {1}, w = {2}, beta = {3}'.format(lobe_fraction, tolerance, w, beta))

    output = np.empty((w, 1), dtype=np.complex128)

    for i in xrange(w):
        output[i] = cheb(w - 1, beta * np.cos(np.pi * i / float(w))) * tolerance

    output = fft(output, overwrite_x=True)

    # TODO fixme

    # output = fftshift(output)
    # output = np.real(output)

    print output

    return output


def main():
    def display_gaussian():
        output = generate_gaussian(
            lobe_fraction=0.025,
            tolerance=0.00000001
        )

        plt.figure()
        plt.plot(output)
        plt.title('Gaussian window')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample')
        plt.show()

    def display_dolph_chebyshev():
        output = generate_dolph_chebyshev(
            lobe_fraction=0.025,
            tolerance=0.00000001
        )

        plt.figure()
        plt.plot(output)
        plt.title('Dolph-Chebyshev window')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample')
        plt.show()

    display_gaussian()
    # display_dolph_chebyshev()


if __name__ == '__main__':
    main()
