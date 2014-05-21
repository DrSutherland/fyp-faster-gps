__author__ = 'jyl111'

import numpy as np
import fourier_transforms
from parameters import Parameters


class Simulation:
    def __init__(self, params):
        self.params = params

        self.t = np.arange(self.params.n)
        self.x = np.empty(self.params.n, dtype=np.complex128)
        self.x_f = np.zeros(self.params.n)

        self.generate_frequencies()
        self.generate_input()

    def generate_frequencies(self):
        """Generate locations of k random frequencies"""

        # Generate k random indices
        indices = np.random.randint(self.params.n, size=self.params.k)

        # Set values of locations to 1.0
        self.x_f[indices] = 1.0

    def generate_input(self):
        """Generate time domain signal"""
        self.x = fourier_transforms.ifft(self.x_f)


def main():
    params = Parameters(
        n=1024,
        k=10,
    )
    sim = Simulation(params=params)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(sim.t, sim.x.real, 'b-', sim.t, sim.x.imag, 'r--')
    ax.legend(('Real', 'Imaginary'))
    ax.set_xlim(right=sim.t.shape[-1]-1)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(sim.t, sim.x_f)
    ax.set_xlim(right=sim.t.shape[-1]-1)

    plt.show()


if __name__ == '__main__':
    main()
