import numpy as np
from scipy import signal
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


def main():
    params = Parameters(
        n=65536,
        k=50
    )

    output = generate_gaussian(
        lobe_fraction=0.025,
        tolerance=0.00000001
    )

    plt.figure()
    plt.plot(output)
    plt.title(r'Gaussian window ($\sigma$=7)')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.show()


if __name__ == '__main__':
    main()
