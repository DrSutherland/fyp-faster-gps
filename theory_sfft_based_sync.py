__author__ = 'jyl111'

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift

import ca_code
import sfft_aliasing


def main():
    # (a) Input
    code = ca_code.generate(prn=1, repeats=10)
    # rx = np.roll(ca_code.generate(prn=1, repeats=10), -10230/6*1) | ca_code.generate(prn=2, repeats=10)
    rx = np.roll(ca_code.generate(prn=1, repeats=10), 1000) | ca_code.generate(prn=2, repeats=10)

    noise_std = 1
    noise = np.random.normal(
        scale=noise_std,
        size=code.size,
    )

    rx = np.add(rx, noise)

    # (b) Aliasing
    q = 2
    code_aliased = sfft_aliasing.execute(code, q)
    rx_aliased = sfft_aliasing.execute(rx, q)

    # (c) FFT
    code_f = fft(code_aliased)
    rx_f = fft(rx_aliased)

    # (d) Multiply
    code_rx_f = np.multiply(np.conjugate(code_f), rx_f)

    # (e) IFFT
    code_rx = np.real(ifft(code_rx_f))

    print 'arg max is', np.argmax(code_rx)

    plt.figure('Local C/A code')
    plt.step(np.arange(code.size), code)
    plt.xlim(0, code.size-1)
    plt.title('Local C/A code')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('Received signal')
    plt.plot(rx)
    plt.xlim(0, code.size-1)
    plt.title('Received signal')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('Aliased Local C/A code')
    plt.step(np.arange(code_aliased.size), code_aliased)
    plt.xlim(0, code_aliased.size-1)
    plt.title('Local C/A code (Aliased)')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('Aliased Received signal')
    plt.plot(rx_aliased)
    plt.xlim(0, rx_aliased.size-1)
    plt.title('Received signal (Aliased)')
    plt.xlabel('Time')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('FFT of Local CA code')
    plt.plot(np.abs(fftshift(code_f)))
    plt.title('FFT of Local C/A code')
    plt.xlabel('Frequency')
    plt.xlim(1000, 4000)
    plt.ylim(0, 500)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('FFT of Received signal')
    plt.plot(np.abs(fftshift(rx_f)))
    plt.title('FFT of Received signal')
    plt.xlabel('Frequency')
    plt.xlim(1000, 4000)
    plt.ylim(0, 500)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('Multiply')
    plt.plot(np.abs(fftshift(code_rx_f)))
    plt.title('Multiply')
    plt.xlabel('Frequency')
    plt.xlim(1500, 3500)
    plt.ylim(0, 100000)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.figure('IFFT')
    plt.plot(code_rx)
    plt.title('IFFT')
    plt.xlim(0, code_rx.size-1)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.show()


if __name__ == '__main__':
    main()