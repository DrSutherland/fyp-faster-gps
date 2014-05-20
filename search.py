from collections import namedtuple

import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft, ifft
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ca_code


PlotPoints = namedtuple('PlotPoints', 'x y z')


def fdcorr(a, b, frange):
    a = a.reshape((-1, 1))
    b = b.reshape((-1, 1))
    frange = frange.flatten(1)

    n = np.max([a.size, b.size])
    fft_a = fft(a, n, axis=0)

    freq_index = np.arange(-np.floor((n-1)/2.0), np.floor(n/2.0)+1)
    freq_axis = freq_index * 2 * np.pi / n

    freq_include = np.nonzero((freq_axis >= frange[0]) & (freq_axis <= frange[1]))[0]
    freq_axis = freq_axis[freq_include]
    freq_index = freq_index[freq_include]
    num_freqs = freq_include.size

    fft_idx = np.arange(1, n+1).reshape((-1, 1))

    fft_dopplers = np.tile(fft_idx, (1, num_freqs))
    fft_dopplers += np.tile(freq_index, (n, 1))

    # Generate Hankel matrix
    fft_dopplers = np.mod(fft_dopplers, n) - 1

    fft_dopplers = fft_a.take(fft_dopplers)

    fft_code_mtx = fft(b, n, axis=0)
    fft_code_mtx = np.tile(fft_code_mtx, (1, num_freqs))

    fdout = ifft(fft_code_mtx * np.conj(fft_dopplers), axis=0)

    return PlotPoints(
        x=freq_axis.reshape((1, num_freqs)),
        y=np.arange(1, n+1).reshape((n, 1)),
        z=np.abs(fdout).reshape((n, num_freqs)),
    )


def plot(plot_points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X=plot_points.x,
        Y=plot_points.y,
        Z=plot_points.z,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def main():
    data = loadmat('fdcorr_demo')

    plot_points = fdcorr(data['rx_seg'], ca_code.generate(prn=26, sampling_rate=2), data['frange'])

    plot(plot_points)


if __name__ == '__main__':
    main()
