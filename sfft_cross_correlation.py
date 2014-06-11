import matplotlib.pyplot as plt
import numpy as np

import ca_code
import fourier_transforms
from parameters import Parameters
import sfft, sfft_inverse
import sfft_aliasing

__author__ = 'jyl111'

def main():
    code_1 = ca_code.generate(prn=1).reshape((-1,))
    code_2 = ca_code.generate(prn=1).reshape((-1,))

    # code_2 = np.roll(code_2, 100)

    # fftshift(ifft(fft(a,corrLength).*conj(fft(b,corrLength))))

    q = 4
    code_1 = sfft_aliasing.execute(code_1, q)
    code_2 = sfft_aliasing.execute(code_2, q)

    code_1_fft = fourier_transforms.fft(code_1)
    code_2_fft = fourier_transforms.fft(code_2)

    multiplied = code_1_fft * np.conj(code_2_fft)

    print multiplied.size

    params = Parameters(
        n=multiplied.size/2,
        k=1,
        B_k_location=2,
        B_k_estimation=2,
        estimation_loops=8,
        location_loops=5,
        loop_threshold=4,
        tolerance_location=1e-8,
        tolerance_estimation=1e-8
    )
    result = sfft_inverse.execute(params=params, x=multiplied)
    result = fourier_transforms.fftshift(result)

    result_actual = fourier_transforms.ifft(multiplied)
    result_actual = fourier_transforms.fftshift(result_actual)

    print 'sfft size', result.size
    print 'original size', result_actual.size

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(np.linspace(-1, 1, result.size), np.abs(result))
    ax.plot(np.linspace(-1, 1, result_actual.size), np.abs(result_actual))
    plt.show()


if __name__ == '__main__':
    main()
