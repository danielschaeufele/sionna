#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from sionna.phy.ofdm import OFDMModulator
from sionna.phy.channel import PowerAmplifierNonlinearity, FreqDomainPowerAmplifierNonlinearity
from sionna.phy.mapping import QAMSource
from sionna.phy import config

import pytest
import numpy as np


def polynomial_nonlinearity(data, backoff_db, model_coefficients):
    normalization_factor = np.sqrt(np.mean(np.abs(data)**2))
    data *= 10. ** (-backoff_db / 20) / normalization_factor
    clip_mask = np.abs(data) > 1
    data[clip_mask] = data[clip_mask] / np.abs(data[clip_mask])
    result = np.zeros_like(data)
    for l, m, coeffs in model_coefficients:
        signal_term = np.roll(data, l, axis=-1)
        envelope_term = np.roll(np.abs(data), l + m, axis=-1)
        for k, coeff in enumerate(coeffs):
            result += coeff * signal_term * np.power(envelope_term, 2 * k)
    result *= 10. ** (backoff_db / 20) * normalization_factor
    return result


def freq_domain_polynomial_nonlinearity(data, backoff_db, model_coefficients, fft_size, zero_pads):
    if zero_pads is not None:
        data = np.pad(data, {-1: zero_pads})
    data = np.fft.ifftshift(data, axes=-1)
    data_time = np.fft.ifft(data, axis=-1) * np.sqrt(fft_size)
    data_time_nl = polynomial_nonlinearity(data_time, backoff_db, model_coefficients)
    data_freq_nl = np.fft.fft(data_time_nl, axis=-1) / np.sqrt(fft_size)
    data_freq_nl = np.fft.fftshift(data_freq_nl, axes=-1)
    if zero_pads is not None:
        data_freq_nl = data_freq_nl[..., zero_pads : -zero_pads]
    return data_freq_nl


class TestPowerAmplifierNonlinearity:
    def test_time_domain_power_amplifier_nonlinearity(self, subtests):
        config.seed = 1
        fft_size = 1024
        batch_size = 10
        num_ofdm_symbols = 14

        custom_model_coefficients = [
            (0, 0, [1.07 - 0.2j, -0.95 - 0.25j, 21.84 + 13.71j, -150.68 - 59.45j,
                    468.34 + 111.04j, -798.37 - 89.73j, 775 + 3.05j,
                    -402.67 + 38.38j, 87.02 - 16.45j])]

        qam_source = QAMSource(4)
        modulator = OFDMModulator()

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        for power_backoff_db in [0, 2, 4, 10, 20]:
            for model in ['GaAs_2GHz', 'GaN_2GHz', 'CMOS_28GHz', 'GaN_28GHz', 'GaAs_2GHz_GMP',
                          'GaN_2GHz_GMP', 'CMOS_28GHz_GMP', 'GaN_28GHz_GMP', 'custom']:
                with subtests.test(msg=f"{model} with {power_backoff_db}dB power backoff"):
                    if model == 'custom':
                        model = custom_model_coefficients
                    pa_nonlin = PowerAmplifierNonlinearity(power_backoff_db, model)
                    x_test = pa_nonlin(x_time.clone())
                    x_gt = polynomial_nonlinearity(x_time.cpu().numpy(), power_backoff_db,
                                                pa_nonlin._model_coefficients)

                    np.testing.assert_array_almost_equal(x_test.cpu().numpy(), x_gt, decimal=3)

    def test_freq_domain_power_amplifier_nonlinearity(self, subtests):
        config.seed = 1
        fft_size = 1024
        batch_size = 10
        num_ofdm_symbols = 14
        sample_rate = 122.88e6

        custom_model_coefficients = [
            (0, 0, [1.07 - 0.2j, -0.95 - 0.25j, 21.84 + 13.71j, -150.68 - 59.45j,
                    468.34 + 111.04j, -798.37 - 89.73j, 775 + 3.05j,
                    -402.67 + 38.38j, 87.02 - 16.45j])]

        qam_source = QAMSource(4)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])

        for power_backoff_db in [0, 4, 10]:
            for model in ['GaAs_2GHz', 'GaN_28GHz', 'GaN_2GHz_GMP', 'CMOS_28GHz_GMP', 'custom']:
                with subtests.test(msg=f"{model} with {power_backoff_db}dB power backoff"):
                    if model == 'custom':
                        model = custom_model_coefficients
                        custom_sample_rate = 200e6
                    else:
                        custom_sample_rate = None
                    pa_nonlin = PowerAmplifierNonlinearity(
                        power_backoff_db, model, sample_rate=custom_sample_rate)
                    freq_domain_nonlin = FreqDomainPowerAmplifierNonlinearity(
                        pa_nonlin, fft_size, sample_rate=sample_rate)

                    if pa_nonlin._sample_rate is not None:
                        upsampling_factor = round(pa_nonlin._sample_rate / sample_rate)
                        upsampled_fft_size = fft_size * upsampling_factor
                        zero_pads = (upsampling_factor - 1) * fft_size // 2
                        assert upsampling_factor == freq_domain_nonlin._upsampling_factor
                        assert upsampled_fft_size == freq_domain_nonlin._upsampled_fft_size
                        assert zero_pads == freq_domain_nonlin._zero_pads
                    else:
                        zero_pads = None

                    x_gt = freq_domain_polynomial_nonlinearity(
                        x.cpu().numpy(),
                        power_backoff_db,
                        pa_nonlin._model_coefficients,
                        fft_size,
                        zero_pads
                    )
                    x_test = freq_domain_nonlin(x)
                    np.testing.assert_array_almost_equal(x_test.cpu().numpy(), x_gt, decimal=3)
