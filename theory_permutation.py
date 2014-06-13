__author__ = 'jyl111'

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift


def main():
    freq_samp = 1000
    freq = 50
    length = 1000
    t = np.arange(length, dtype=float)/freq_samp
    f = freq_samp/2.0*np.linspace(-1, 1, length)

    x_t = np.sin(2*np.pi*freq*t)
    x_f = fft(x_t)

    # Permutate
    y_f = np.concatenate((x_f[100:200], x_f[0:100], x_f[200:800], x_f[900:1000], x_f[800:900]))
    y_t = np.real(ifft(y_f))

    plt.figure()
    plt.plot(
        t, x_t, 'b',
        t, y_t, 'g'
    )
    plt.xlim(0, 0.05)
    plt.ylim(-1, 1)
    plt.legend(('Original signal', 'Permuted signal'))
    plt.ylabel('Amplitude')
    plt.xlabel('Time')

    plt.figure()
    plt.plot(
        f, np.abs(fftshift(x_f)), 'b',
        f, np.abs(fftshift(y_f)), 'g'
    )
    plt.xlim(-300, 300)
    plt.legend(('Original signal', 'Permuted signal'))
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency ($Hz$)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()