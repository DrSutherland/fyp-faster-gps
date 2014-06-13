__author__ = 'jyl111'

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift

import ca_code


def main():
    # (a) Input
    code = ca_code.generate(prn=1, repeats=10)
    rx = np.roll(ca_code.generate(prn=1, repeats=10), 10230/4*3) | ca_code.generate(prn=2, repeats=10)

    noise_std = 1
    noise = np.random.normal(
        scale=noise_std,
        size=code.size,
    )

    rx = np.add(rx, noise)

    # (b) FFT
    code_f = fft(code, n=code.size)
    rx_f = fft(rx)

    # (c) Multiply
    code_rx_f = np.multiply(np.conjugate(code_f), rx_f)

    # (d) IFFT
    code_rx = np.real(ifft(code_rx_f))

    plt.figure('Local C/A code')
    plt.step(np.arange(code.size), code)
    plt.xlim(0, code.size-1)
    plt.title('Local C/A code')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.figure('Received signal')
    plt.plot(rx)
    plt.xlim(0, code.size-1)
    plt.title('Received signal')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.figure('FFT of Local C/A code')
    plt.plot(np.abs(fftshift(code_f)))
    plt.title('FFT of Local C/A code')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.figure('FFT of Received signal')
    plt.plot(np.abs(fftshift(rx_f)))
    plt.title('FFT of Received signal')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.figure('Multiply')
    plt.plot(np.abs(fftshift(code_rx_f)))
    plt.title('Multiply')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.figure('IFFT')
    plt.plot(code_rx)
    plt.title('IFFT')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()