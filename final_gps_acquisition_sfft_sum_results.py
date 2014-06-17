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
from performance_counter import PerformanceCounter
import sfft_aliasing


def acquisition(x, settings,
                plot_graphs=False,
                plot_3d_graphs=False,
                performance_counter=PerformanceCounter()):
    # Calculate number of samples per spreading code (corresponding to 1ms of data)
    samples_per_code = int(round(settings['sampling_frequency'] * settings['code_length'] / settings['code_frequency']))
    print 'samples_per_code = %s' % repr(samples_per_code)

    # p = int(np.round(np.sqrt(np.log(samples_per_code))))
    # print 'p = ', p
    p = settings['sfft_subsampling_factor']
    n__samples_to_alias = samples_per_code * p

    # print n__samples_to_alias
    # print x_1.shape


    # print 'x_1.shape = %s' % repr(x_1.shape)
    # print 'x_2.shape = %s' % repr(x_2.shape)

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

    # Generate and store C/A lookup table
    ca_codes__time = ca_code.generate_table(settings)

    # SFFT
    ca_codes__freq = np.conjugate(fft(ca_codes__time))
    print 'ca_codes__time(%s) = %s' % (repr(ca_codes__time.shape), repr(ca_codes__time))

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
    satellites_to_search__shape = settings['satellites_to_search'].shape
    output = {
        'frequency_shifts': np.zeros(satellites_to_search__shape),
        'code_shifts': np.zeros(satellites_to_search__shape),
        'peak_ratios': np.zeros(satellites_to_search__shape),
        'found': np.zeros(satellites_to_search__shape, dtype=np.bool),
    }

    for idx, prn in enumerate(settings['satellites_to_search']):
        print '* searching PRN = %s' % (repr(prn),)

        result_summation = np.zeros(samples_per_code)

        for sum_idx in xrange(settings['sfft_sum_results']):
            x_1 = x[sum_idx*n__samples_to_alias:(sum_idx*n__samples_to_alias+n__samples_to_alias)]
            x_1 = sfft_aliasing.execute(x_1, p)

            #
            # Scan all Doppler shifts
            #
            for freq_bin_i in xrange(n_frequency_bins):
                # Generate local sine and cosine carriers
                carrier_sin = np.sin(frequency_bins[freq_bin_i] * phases)
                carrier_cos = np.cos(frequency_bins[freq_bin_i] * phases)

                # Demodulation
                I1 = carrier_sin * x_1
                Q1 = carrier_cos * x_1

                # Reconstruct baseband signal
                IQ1 = I1 + 1j*Q1

                # if settings['use_sfft']:
                #     IQ1 = sfft_aliasing.execute(IQ1, settings['sfft_subsampling_factor'])
                #     IQ2 = sfft_aliasing.execute(IQ2, settings['sfft_subsampling_factor'])
                #     performance_counter.increase(additions=IQ1.size * settings['sfft_subsampling_factor'])
                #     performance_counter.increase(additions=IQ2.size * settings['sfft_subsampling_factor'])

                # Convert to frequency domain
                IQ1_freq = fft(IQ1)
                performance_counter.fft(IQ1.size)

                # Multiplication in the frequency domain corresponds to convolution in the time domain
                conv_code_IQ1 = IQ1_freq * ca_codes__freq[prn-1]
                performance_counter.increase(multiplications=IQ1_freq.size)

                # IFFT to obtain correlation
                corr_result_1 = np.abs(ifft(conv_code_IQ1)) ** 2

                # assert all_results[freq_bin_i, :].shape == corr_result_1.shape == corr_result_2.shape

                all_results[freq_bin_i, :] = corr_result_1

            # Get the peak location for every frequency bins
            peak_values = all_results.max(axis=1)
            assert all_results.max() in peak_values

            # Find the Doppler shift index
            frequency_shift_idx = peak_values.argmax()
            print 'frequency_shift_idx = %s' % repr(frequency_shift_idx)

            # Select the frequency bin that corresponds to this frequency shift index
            located_frequency_bin = all_results[frequency_shift_idx]

            result_summation += located_frequency_bin

        plt.figure()
        plt.plot(result_summation / result_summation.max())
        plt.ylabel('Normalised magnitude')
        plt.xlabel('Code shift (chips)')
        plt.title('Summing %d results' % settings['sfft_sum_results'])
        plt.show()
        exit()

        # Find the code shift in the correct frequency bin
        code_shift = result_summation.argmax()
        output['code_shifts'][idx] = code_shift
        print 'code_shift = %s' % repr(code_shift)

        peak_value = result_summation[code_shift]
        assert result_summation.max() == peak_value
        print 'peak_value = %s' % repr(peak_value)

        if plot_3d_graphs:
            fig = plt.figure()
            doppler_shifts__khz = (
                (settings['acquisition_search_frequency_step'] * np.arange(n_frequency_bins)) -
                (settings['acquisition_search_frequency_band'] / 2)
            ) / 1000

            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(
                X=np.arange(samples_per_code).reshape((1, -1)),
                Y=doppler_shifts__khz.reshape((-1, 1)),
                Z=all_results,
                rstride=1,
                cstride=1,
                cmap=matplotlib.cm.coolwarm,
                linewidth=0,
                antialiased=False,
            )
            ax.set_xlabel('Code shift')
            ax.set_ylabel('Doppler shift (kHz)')

        # Calculate code phase range
        samples_per_code_chip = int(round(settings['sampling_frequency'] / settings['code_frequency']))


        print 'samples_per_code_chip = %s' % repr(samples_per_code_chip)

        #
        # Get second largest peak value outside the chip where the maximum peak is located
        #

        # Calculate excluded range
        excluded_range_1 = code_shift - samples_per_code_chip
        excluded_range_2 = code_shift + samples_per_code_chip
        print 'excluded_range_1 = %s' % repr(excluded_range_1)
        print 'excluded_range_2 = %s' % repr(excluded_range_2)

        # Excluded range boundary correction
        if excluded_range_1 < 1:
            print 'excluded_range_1 < 1'
            code_phase_range = np.arange(excluded_range_2, samples_per_code + excluded_range_1)
        elif excluded_range_2 >= samples_per_code:
            print 'excluded_range_2 >= samples_per_code'
            code_phase_range = np.arange(excluded_range_2 - samples_per_code, excluded_range_1)
        else:
            code_phase_range = np.concatenate((
                np.arange(0, excluded_range_1),
                np.arange(excluded_range_2, samples_per_code)
            ))

        assert code_shift not in code_phase_range
        print 'code_phase_range(%s) = %s' % (repr(code_phase_range.shape), repr(code_phase_range))

        # Get second largest peak value
        second_peak_value = all_results[frequency_shift_idx, code_phase_range].max()
        print 'second_peak_value = %s' % repr(second_peak_value)

        # Calculate ratio between the largest peak value and the second largest peak value
        peak_ratio = peak_value / second_peak_value
        print 'peak_ratio = %s' % repr(peak_ratio)
        output['peak_ratios'][idx] = peak_ratio

        #
        # Thresholding
        #

        if peak_ratio > settings['acquisition_threshold']:
            output['found'][idx] = True
            print '-> %s FOUND' % repr(prn)
        else:
            output['found'][idx] = False
            print '-> %s NOT FOUND' % repr(prn)

    if plot_graphs:
        plt.figure()
        colors = ['r' if found else 'b' for found in output['found']]
        plt.bar(settings['satellites_to_search'], output['peak_ratios'], color=colors, align='center')
        plt.ylabel('Acquisition quality')
        plt.xlabel('PRN number')

        artist__not_acquired = plt.Rectangle((0, 0), 1, 1, fc='b')
        artist__acquired = plt.Rectangle((0, 0), 1, 1, fc='r')
        plt.legend((
            artist__not_acquired,
            artist__acquired
        ), (
            'Not acquired',
            'Acquired'
        ))
        plt.xlim(0, settings['satellites_total'] + 1)

        plt.grid()
        plt.tight_layout()

    return output, performance_counter


if __name__ == '__main__':
    import gps_data_reader

    settings = {
        'file_name': './GNSS_signal_records/GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin',
        'load_all_data': True,
        'byte_offset': 2000,
        'data_type': np.int8,
        'intermediate_frequency': 4130400,
        'sampling_frequency': 16367600,
        'code_frequency': 1023000,
        'code_length': 1023,
        'code_offset': 0,
        'satellites_total': 32,
        'satellites_to_search': np.array([3]),#np.arange(32)+1,#np.array([22]),
        'acquisition_search_frequency_band': 14000,
        'acquisition_search_frequency_step': 500,
        'acquisition_threshold': 2.5,
        'use_sfft': True,
        'sfft_subsampling_factor': 2,
        'sfft_sum_results': 20
    }

    x = gps_data_reader.read(settings)

    results, performance_counter = acquisition(x, settings, plot_graphs=True, plot_3d_graphs=False)

    for idx, found in enumerate(results['found']):
        if found:
            print '-> FOUND: prn = %s, shift = %s' % (repr(settings['satellites_to_search'][idx]), repr(results['code_shifts'][idx]))

    print repr(performance_counter)

    plt.show()
