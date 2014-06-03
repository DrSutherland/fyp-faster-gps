from __future__ import division

__author__ = 'jyl111'

import matplotlib.pyplot as plt
import numpy as np

import fourier_transforms
from parameters import Parameters
import sfft


class Simulation:
    def __init__(self, params):
        self.params = params

        self.t = np.arange(self.params.n)
        self.x = np.zeros(self.params.n, dtype=np.complex128)
        self.x_f = np.zeros(self.params.n)
        self.y = np.zeros(self.params.n, dtype=np.complex128)

        self.generate_frequencies()

        self.generate_input()
        self.add_noise_to_input()

        self.x_f /= self.params.n

        # self.execute_sfft()

        # self.generate_output()

    def generate_frequencies(self):
        """Generate locations of k random frequencies"""

        # Generate k random indices
        indices = np.random.randint(self.params.n, size=self.params.k)

        # Set values of locations to 1.0
        self.x_f[indices] = 1.0

    def generate_input(self):
        """Generate time domain signal"""
        self.x = fourier_transforms.ifft(self.x_f)

    def add_noise_to_input(self):
        if np.isinf(self.params.snr):
            return

        # Calculate signal power
        signal_power = np.sum(np.square(np.absolute(self.x))) / self.x.shape[-1]

        # Calculate desired noise std
        noise_std = np.sqrt(signal_power / self.params.snr)

        # Generate AWGN
        noise = np.random.normal(
            scale=noise_std,
            size=self.params.n,
        )

        # Calculate noise power
        noise_power = np.sum(np.square(np.absolute(noise))) / noise.shape[-1]

        # Calculate SNR
        snr = signal_power / noise_power

        # Add noise to input
        self.x += noise

        print "SNR is {0} or {1} dB".format(snr, 10 * np.log10(snr))

    def execute_sfft(self):
        # todo do something with the result
        sfft.execute(params=self.params)

    def generate_output(self):
        self.y = fourier_transforms.fft(self.x)

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.t, self.x.real, 'b-', self.t, self.x.imag, 'r--')
        ax.legend(('Real', 'Imaginary'))
        ax.set_xlim(right=self.t.shape[-1]-1)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.t, self.x_f)
        ax.set_xlim(right=self.t.shape[-1]-1)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.t, self.y.real, 'b-', self.t, self.y.imag, 'r--')
        ax.legend(('Real', 'Imaginary'))
        ax.set_xlim(right=self.t.shape[-1]-1)


def main():
    params = Parameters(
        n=1024,
        k=10,
    )
    sim = Simulation(params=params)
    sim.plot()
    plt.show()


if __name__ == '__main__':
    main()
