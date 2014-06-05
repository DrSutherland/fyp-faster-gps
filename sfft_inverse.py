import matplotlib.pyplot as plt
import numpy as np

from parameters import Parameters
import sfft

__author__ = 'jyl111'

def execute(params, x):
    x = np.imag(x) + 1j * np.real(x)

    A = sfft.execute(params=params, x=x)

    B = np.imag(A) + 1j * np.real(A)

    return B


def main():
    params = Parameters(
        n=1024,
        k=1,
        B_k_location=4,
        B_k_estimation=2,
        estimation_loops=8,
        location_loops=5,
        loop_threshold=4,
        tolerance_location=1e-8,
        tolerance_estimation=1e-8
    )

    from simulation import Simulation

    sim = Simulation(params=params)

    x_f = execute(params=params, x=sim.x)
    x_f_actual = np.fft.ifft(sim.x)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(
        sim.t, x_f_actual.real, '-',
        sim.t, x_f_actual.imag, '-',
        sim.t, x_f.real, '--',
        sim.t, x_f.imag, '--',
    )
    ax.legend((
        'Real', 'Imaginary',
        'sFFT Real', 'sFFT Imaginary'
    ))
    ax.set_xlim(right=sim.t.shape[-1]-1)

    plt.show()


if __name__ == '__main__':
    main()