import numpy as np
import matplotlib.pyplot as plt

# Phase taps according to the ICD-GPS-200 specification
PHASE_TAPS = np.array([
    [2, 6],   # PRN 1
    [3, 7],   # PRN 2
    [4, 8],   # PRN 3
    [5, 9],   # PRN 4
    [1, 9],   # PRN 5
    [2, 10],  # PRN 6
    [1, 8],   # PRN 7
    [2, 9],   # PRN 8
    [3, 10],  # PRN 9
    [2, 3],   # PRN 10
    [3, 4],   # PRN 11
    [5, 6],   # PRN 12
    [6, 7],   # PRN 13
    [7, 8],   # PRN 14
    [8, 9],   # PRN 15
    [9, 10],  # PRN 16
    [1, 4],   # PRN 17
    [2, 5],   # PRN 18
    [3, 6],   # PRN 19
    [4, 7],   # PRN 20
    [5, 8],   # PRN 21
    [6, 9],   # PRN 22
    [1, 3],   # PRN 23
    [4, 6],   # PRN 24
    [5, 7],   # PRN 25
    [6, 8],   # PRN 26
    [7, 9],   # PRN 27
    [8, 10],  # PRN 28
    [1, 6],   # PRN 29
    [2, 7],   # PRN 30
    [3, 8],   # PRN 31
    [4, 9],   # PRN 32
]) - 1        # Minus 1 to take into account Python's zero-indexed arrays

# G1 = 1 + x3 + x10
POLYNOMIAL_1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

# G2 = 1 + x2 + x3 + x6 + x8 + x9 + x10
POLYNOMIAL_2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])

# Shift register size
SR_SIZE = 10


def get_phase_taps(prn):
    return PHASE_TAPS[prn-1]


def generate(prn, sampling_rate=1):
    """Generates the C/A gold code given a satellite's PRN number"""

    # Force numpy array
    prn = np.array(prn)

    # Allocate G1 (shift register)
    sr1 = np.ones([prn.size, POLYNOMIAL_1.size])

    # Allocate G2 (shift register)
    sr2 = np.ones([prn.size, POLYNOMIAL_2.size])

    # Get phase tap indices for selected PRN
    phase_taps = get_phase_taps(prn=prn)

    # Maximum-length shift register (MLSR)
    # Length is ((Length of 2^SR_SIZE) - 1) = 1023
    sr_length = (2 ** SR_SIZE) - 1

    # Allocate G3 (output)
    output = np.zeros([prn.size, sr_length])

    # Generate C/A code sequence
    for i in range(sr_length):
        # Take values from sr2 using the index specified by the PRN
        phase_tapped = np.mod(np.sum(sr2.take(phase_taps).reshape((-1, 2)), axis=1), 2)

        # Calculates the current C/A code sequence
        output[:, i] = np.mod(sr1[:, -1] + phase_tapped, 2)

        # Calculates last value in shift register sr1
        # BEFORE rotating to the right
        sr1[:, -1] = np.mod(np.sum(sr1 * POLYNOMIAL_1, axis=1), 2)
        # Once rotated, the last value is now the first value
        sr1 = np.roll(sr1, 1)

        # Do the same for shift register sr2
        sr2[:, -1] = np.mod(np.sum(sr2 * POLYNOMIAL_2, axis=1), 2)
        sr2 = np.roll(sr2, 1)

    return np.repeat(output, repeats=sampling_rate, axis=1)


def main():
    # Generate PRN values to generate
    prns = np.arange(1, PHASE_TAPS.shape[0] + 1)

    # Generate C/A codes given an array of PRN values
    ca_codes = generate(prn=prns)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    # Plots binary sequence as an image
    # Colormap reference: http://matplotlib.org/examples/color/colormaps_reference.html
    ax.imshow(
        ca_codes,
        cmap='binary',
        interpolation='nearest',
        origin='lower',
        extent=(0, ca_codes.shape[1], 1, ca_codes.shape[0]),
    )

    ax.set_title('C/A Codes with different PRNs')
    ax.set_ylabel('PRN')
    ax.set_xlabel('C/A Code')
    ax.tick_params(axis='y', labelsize=8)

    plt.show()


if __name__ == '__main__':
    main()
