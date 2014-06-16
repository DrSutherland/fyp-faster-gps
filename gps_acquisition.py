from __future__ import division

__author__ = 'jyl111'

import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.fft import fft, ifft
from scipy import signal

import ca_code


def acquisition(x, settings, plot_graphs=False):
    # Calculate number of samples per spreading code (corresponding to 1ms of data)
    samples_per_code = int(round(settings['sampling_frequency'] * settings['code_length'] / settings['code_frequency']))
    print 'samples_per_code = %s' % repr(samples_per_code)

    # Two consecutive 2ms reading
    x_1 = x[0:samples_per_code]
    x_2 = x[samples_per_code:2*samples_per_code]
    print 'x_1.shape = %s' % repr(x_1.shape)
    print 'x_2.shape = %s' % repr(x_2.shape)

    # x_0dc = x - np.mean(x)

    # Calculate sampling period
    sampling_period = 1.0 / settings['sampling_frequency']
    print 'sampling_period = %s' % repr(sampling_period)

    # Generate phase points of the local carrier
    phases = np.arange(0, samples_per_code) * 2 * np.pi * sampling_period
    print 'phases(%s) = %s' % (repr(phases.shape), repr(phases))

    # Calculate number of frequency bins depending on search frequency band and frequency step
    assert settings['acquisition_search_frequency_band'] % settings['acquisition_search_frequency_step'] == 0, \
        'acquisition_search_frequency_band should be divisible by acquisition_search_frequency_step'
    n_frequency_bins = int(settings['acquisition_search_frequency_band'] / settings['acquisition_search_frequency_step']) + 1
    print 'n_frequency_bins = %s' % repr(n_frequency_bins)

    # todo generate ca table

    # Allocate memory for the 2D search
    all_results = np.empty(shape=(n_frequency_bins, samples_per_code))

    # Generate all frequency bins
    frequency_bins = (
        settings['intermediate_frequency'] - \
        (settings['acquisition_search_frequency_band'] / 2) + \
        settings['acquisition_search_frequency_step'] * np.arange(n_frequency_bins)
    ).astype(int)
    print 'frequency_bins(%s) = %s' % (repr(frequency_bins.shape), repr(frequency_bins))

    # Allocate memory for output dictionary
    output = {
        'frequency_shifts': np.zeros(settings['satellites_total']),
        'code_shifts': np.zeros(settings['satellites_total']),
        'peak_ratios': np.zeros(settings['satellites_total']),
        'found': np.zeros(settings['satellites_total'], dtype=np.bool),
    }

    for prn in settings['satellites_to_search']:
        print '* searching PRN = %s' % (repr(prn),)

        # todo Move this somewhere else
        ca_code_time = ca_code.generate(prn=prn).repeat(samples_per_code/1023)
        ca_code_freq = np.conjugate(fft(ca_code_time))

        # Scan Doppler frequencies
        for freq_bin_i in xrange(n_frequency_bins):
            # Generate local sine and cosine carriers
            carrier_sin = np.sin(frequency_bins[freq_bin_i] * phases)
            carrier_cos = np.cos(frequency_bins[freq_bin_i] * phases)

            # Demodulation
            I1 = carrier_sin * x_1
            Q1 = carrier_cos * x_1
            I2 = carrier_sin * x_2
            Q2 = carrier_cos * x_2

            # Reconstruct baseband signal
            IQ1 = I1 + 1j*Q1
            IQ2 = I2 + 1j*Q2

            # Convert to frequency domain
            IQ1_freq = fft(IQ1)
            IQ2_freq = fft(IQ2)

            # Multiplication in the frequency domain corresponds to convolution in the time domain
            conv_code_IQ1 = IQ1_freq * ca_code_freq
            conv_code_IQ2 = IQ2_freq * ca_code_freq

            # IFFT to obtain correlation
            corr_result_1 = np.abs(ifft(conv_code_IQ1)) ** 2
            corr_result_2 = np.abs(ifft(conv_code_IQ2)) ** 2

            assert all_results[freq_bin_i, :].shape == corr_result_1.shape == corr_result_2.shape

            if np.max(corr_result_1) > np.max(corr_result_2):
                all_results[freq_bin_i, :] = corr_result_1
            else:
                all_results[freq_bin_i, :] = corr_result_2

        # Peak location for each Doppler shift
        peak_values = all_results.max(axis=1)

        assert all_results.max() in peak_values

        frequency_shift = peak_values.argmax()
        print 'Frequency shift is %d' % frequency_shift

        correct_frequency_result = all_results[frequency_shift]

        code_shift = correct_frequency_result.argmax()
        print 'Code shift is %d' % code_shift

        assert all_results.max() == all_results[frequency_shift][code_shift]
        peak_value = all_results[frequency_shift][code_shift]
        print 'Peak value = %f' % peak_value

        # if plot_graphs:
        #     print np.arange(1023*16).reshape((1, -1)).shape, frequency_bins.shape, all_results.shape
        #
        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        #     surf = ax.plot_surface(
        #         X=np.arange(1023*16).reshape((1, -1)),
        #         Y=frequency_bins.reshape((-1, 1)),
        #         Z=all_results,
        #         rstride=1,
        #         cstride=1,
        #         cmap=matplotlib.cm.coolwarm,
        #         linewidth=0,
        #         antialiased=False,
        #     )
        #     plt.show()

        # Calculate code phase range
        samples_per_code_chip = int(round(settings['sampling_frequency'] / settings['code_frequency']))
        excluded_range_1 = code_shift - samples_per_code_chip
        excluded_range_2 = code_shift + samples_per_code_chip
        print 'code_shift', code_shift
        print 'samples_per_code_chip', samples_per_code_chip
        print 'excluded_range_1', excluded_range_1
        print 'excluded_range_2', excluded_range_2

        if excluded_range_2 < 1:
            code_phase_range = np.arange(excluded_range_2, samples_per_code + excluded_range_1)
        elif excluded_range_2 >= samples_per_code:
            code_phase_range = np.arange(excluded_range_2 - samples_per_code, excluded_range_1)
        else:
            code_phase_range = np.concatenate((
                np.arange(0, excluded_range_1 - 1),
                np.arange(excluded_range_2, samples_per_code)
            ))

        print code_phase_range.shape[-1]

        # code_phase_range = np.arange(0, 20)
        # print code_phase_range
        assert code_shift not in code_phase_range

        second_peak_value = all_results[frequency_shift, code_phase_range].max()
        print 'Second peak value = %f' % second_peak_value

        peak_ratio = peak_value / float(second_peak_value)
        print 'peak_ratio = %f' % peak_ratio
        output['peak_ratios'][prn-1] = peak_ratio

        if peak_ratio > settings['acquisition_threshold']:
            output['found'][prn-1] = True
            print '* FOUND'
        else:
            output['found'][prn-1] = False
            print '* NOT FOUND'

    return output


if __name__ == '__main__':
    import gps_data_reader

    settings = {
        'file_name': './GNSS_signal_records/GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin',
        'byte_offset': 600,
        'data_type': np.int8,
        'intermediate_frequency': 4130400,
        'sampling_frequency': 16367600,
        'code_frequency': 1023000,
        'code_length': 1023,
        'satellites_total': 32,
        'satellites_to_search': np.arange(15, 23+1),
        'acquisition_search_frequency_band': 14000,
        'acquisition_search_frequency_step': 500,
        'acquisition_threshold': 2.5,
    }

    x = gps_data_reader.read(settings)

    results = acquisition(x, settings, plot_graphs=True)

    plt.figure()
    colors = ['r' if found else 'b' for found in results['found']]
    normal_bars = plt.bar(settings['satellites_to_search'], results['peak_ratios'], color=colors, align='center')
    plt.ylabel('Acquisition quality')
    plt.xlabel('PRN number')

    acquired_artist = plt.Rectangle((0, 0), 1, 1, fc="r")
    plt.legend((
        normal_bars,
        acquired_artist
    ), (
        'Signal not acquired',
        'Signal acquired'
    ))

    plt.show()
