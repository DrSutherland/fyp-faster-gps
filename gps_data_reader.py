__author__ = 'jyl111'

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def read(settings, plot_graphs=False, debug=False):
    # Calculate number of samples per spreading code (corresponding to 1ms of data)
    samples_per_code = int(round(settings['sampling_frequency'] * settings['code_length'] / float(settings['code_frequency'])))

    if debug:
        print 'There are %d samples per code' % samples_per_code

    # Milliseconds of data to plot
    ms_to_plot = 10

    samples_count = samples_per_code * ms_to_plot

    if debug:
        print 'Taking %d samples' % samples_count

    with open(settings['file_name'], 'rb') as f:
        if settings['byte_offset'] > 0:
            print 'Skipping first %d bytes' % settings['byte_offset']
            f.seek(settings['byte_offset'], os.SEEK_SET)

        if settings['load_all_data']:
            actual_count = -1
        else:
            actual_count = samples_count

        x = np.fromfile(f, dtype=settings['data_type'], count=actual_count)

    assert x.size >= samples_count, 'Not enough data'

    if plot_graphs:
        sampling_period = 1.0 / settings['sampling_frequency']

        t = np.arange(start=0, stop=5e-3, step=sampling_period)

        plt.figure()

        # Time domain plot
        plt.subplot(121)
        plt.plot(1000*t[0:np.round(samples_per_code/50)], x[0:np.round(samples_per_code/50)])
        plt.title('Time domain plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tight_layout()

        # Frequency domain plot
        plt.subplot(122)
        f, Pxx_den = signal.welch(x-np.mean(x), settings['sampling_frequency'])
        plt.plot(f, Pxx_den)
        plt.title('PSD estimation')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V**2/Hz)')
        plt.tight_layout()

        plt.show()

    return x


if __name__ == '__main__':
    settings = {
        'file_name': './GNSS_signal_records/GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin',
        'byte_offset': 600,
        'data_type': np.int8,
        'intermediate_frequency': 4130400,
        'sampling_frequency': 16367600,
        'code_frequency': 1023000,
        'code_length': 1023
    }

    x = read(settings, plot_graphs=True)
