import matplotlib.pyplot as plt
import numpy as np

import ca_code
import fourier_transforms
from parameters import Parameters
import sfft, sfft_inverse

__author__ = 'jyl111'

def main():
    code_1 = ca_code.generate(prn=1).reshape((-1,))
    code_2 = ca_code.generate(prn=1).reshape((-1,))

    # fftshift(ifft(fft(a,corrLength).*conj(fft(b,corrLength))))

    code_1_fft = fourier_transforms.fft(code_1)
    code_2_fft = fourier_transforms.fft(code_2)

    multiplied = np.multiply(code_1_fft, np.conjugate(code_2_fft))

    print multiplied.size

    params = Parameters(
        n=multiplied.size,
        k=1,
        B_k_location=2,
        B_k_estimation=2,
        estimation_loops=8,
        location_loops=5,
        loop_threshold=2,
        tolerance_location=1e-8,
        tolerance_estimation=1e-8
    )
    result = sfft_inverse.execute(params=params, x=multiplied)

    # result = fourier_transforms.ifft(multiplied)

    result = fourier_transforms.fftshift(result)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(
        result.real,
        result.imag
    )
    plt.show()


if __name__ == '__main__':
    main()
