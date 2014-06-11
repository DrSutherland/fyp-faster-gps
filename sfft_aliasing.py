__author__ = 'jyl111'

import matplotlib.pyplot as plt
import numpy as np


def execute(a, q):
    """

    :param a: The signal to be aliased
    :param q: The downsampling factor
    :return:
    """

    n = a.size
    n_aliased = n/q

    if n % q != 0:
        raise 'n is not divisible by q'

    output = np.zeros(n_aliased)

    for i in xrange(n):
        output[i % n_aliased] += a[i]

    return output


def main():
    # Generate a step sequence and then leveling off
    input_t = np.append(
        np.linspace(0, 1, 6),
        np.ones(4)
    )

    # Perform aliasing
    output_t = execute(input_t, q=2)

    # Calculate FFTs
    input_f = np.fft.fft(input_t)
    output_f = np.fft.fft(output_t)

    # FFT shift
    input_f = np.fft.fftshift(input_f)
    output_f = np.fft.fftshift(output_f)

    # Absolute FFT output
    input_f = np.abs(input_f)
    output_f = np.abs(output_f)

    plt.figure()

    plt.subplot(221)
    plt.plot(input_t, '-ob')
    for i, value in enumerate(input_t):
        plt.plot([i, i], [value, 0], '--b')
    plt.title('Original Input')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample index')

    plt.subplot(222)
    plt.plot(output_t, '-ob')
    for i, value in enumerate(output_t):
        plt.plot([i, i], [value, 0], '--b')
    plt.title('Aliased Input, $q=2$')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample index')
    plt.ylim(ymin=0)
    plt.xticks(np.arange(output_t.size))

    plt.subplot(223)
    plt.plot(input_f, '-ob')
    for i, value in enumerate(input_f):
        plt.plot([i, i], [value, 0], '--b')
    plt.title('FFT of Original Input')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency index')

    plt.subplot(224)
    plt.plot(output_f, '-ob')
    for i, value in enumerate(output_f):
        plt.plot([i, i], [value, 0], '--b')
    plt.title('FFT of Aliased Input, $q=2$')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency index')

    plt.show()


if __name__ == '__main__':
    main()
