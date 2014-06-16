__author__ = 'jyl111'

import numpy as np
import gps_acquisition
import gps_data_reader

n_code_offsets = 5
threshold_ranges = np.arange(300, 320)


def get_snr(x):
    max_value = x.max()
    max_idx = x.argmax()
    x__without_peak = np.delete(x, max_idx)

    assert max_value not in x__without_peak

    noise_var = (x__without_peak - x__without_peak.mean()).var()

    return max_value**2 / noise_var


corr_result__summed_magnitudes = np.zeros(16368)
total_runs = []
code_shifts = []
snrs = []

for threshold in threshold_ranges:
    code_offset = 0
    current_snr = None

    while True:
        settings = {
            'file_name': './GNSS_signal_records/GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin',
            'byte_offset': 0,
            'data_type': np.int8,
            'intermediate_frequency': 4130400,
            'sampling_frequency': 16367600,
            'code_frequency': 1023000,
            'code_length': 1023,
            'code_offset': code_offset,
            'satellites_total': 32,
            'satellites_to_search': np.array([3]),
            'acquisition_search_frequency_band': 14000,
            'acquisition_search_frequency_step': 500,
            'acquisition_threshold': 2.5,
            'use_sfft': False,
            'sfft_subsampling_factor': 2
        }
        x = gps_data_reader.read(settings)

        results, performance_counter = gps_acquisition.single_acquisition(x, prn=16, settings=settings, plot_3d_graphs=False)

        if not results:
            break

        corr_result__summed_magnitudes += np.abs(results['corr_result'])

        current_snr = get_snr(corr_result__summed_magnitudes)

        if current_snr > threshold:
            break

        # snrs.append(get_snr(corr_result__summed_magnitudes))

        # code_shifts[code_offset] = results['code_shift']

        # max_location = results['corr_result'].argmax()
        # corr_without_peak = np.delete(results['corr_result'], max_location)
        # assert results['corr_result'].max() not in corr_without_peak
        #
        # noise_var = corr_without_peak.var()
        # print 'noise_var = %s' % (repr(noise_var))

        # print results

        # Read the next n
        code_offset += 1

        # results, performance_counter = gps_acquisition.acquisition(x, settings=settings, plot_3d_graphs=False)
        # code_shifts[code_offset] = results['code_shifts'][0]

    total_runs.append(code_offset)
    snrs.append(current_snr)
    code_shifts.append(corr_result__summed_magnitudes.argmax())

print total_runs
print snrs
print code_shifts
