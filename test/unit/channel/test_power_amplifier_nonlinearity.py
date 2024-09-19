#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.channel import PowerAmplifierNonlinearity
from sionna.utils import QAMSource

import pytest
import unittest
import numpy as np
from scipy.signal import resample, resample_poly
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


def memoryless_polynomial_nonlinearity(data, backoff_db, model_coefficients):
    normalization_factor = np.sqrt(np.mean(np.abs(data)**2))
    data *= 10. ** (-backoff_db / 20) / normalization_factor
    clip_mask = np.abs(data) > 1
    data[clip_mask] = data[clip_mask] / np.abs(data[clip_mask])
    result = np.zeros_like(data)
    for k, coeff in enumerate(model_coefficients):
        result += coeff * data * np.power(np.abs(data), 2 * k)
    result *= 10. ** (backoff_db / 20) * normalization_factor
    return result


class TestPowerAmplifierNonlinearity(unittest.TestCase):
    def test_against_reference(self):
        tf.random.set_seed(1)
        fft_size = 1024
        batch_size = 10
        num_ofdm_symbols = 14

        model_coefficients = {
            'GaAs_2GHz': [-0.618347 - 0.785905j, 2.0831 - 1.69506j, -14.7229 + 16.8335j, 61.6423 - 76.9171j,
                          -145.139 + 184.765j, 190.61 - 239.371j, -130.184 + 158.957j, 36.0047 - 42.5192j],
            'GaN_2GHz': [0.999952 - 0.00981788j, -0.0618171 + 0.118845j, -1.69917 - 0.464933j, 3.27962 + 0.829737j,
                         -1.80821 - 0.454331j],
            'CMOS_28GHz': [0.491576 + 0.870835j, -1.26213 + 0.242689j, 7.11693 + 5.14105j, -30.7048 - 53.4924j,
                           73.8814 + 169.146j, -96.7955 - 253.635j, 65.0665 + 185.434j, -17.5838 - 53.1786j],
            'GaN_28GHz': [-0.334697 - 0.942326j, 0.89015 - 0.72633j, -2.58056 + 4.81215j, 4.81548 - 9.54837j,
                          -4.41452 + 8.63164j, 1.54271 - 2.94034j],
            'custom': [1.07 - 0.2j, -0.95 - 0.25j, 21.84 + 13.71j, -150.68 - 59.45j, 468.34 + 111.04j,
                       -798.37 - 89.73j, 775 + 3.05j, -402.67 + 38.38j, 87.02 - 16.45j]
        }

        qam_source = QAMSource(4)
        modulator = OFDMModulator()

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        for power_backoff_db in [0, 2, 4, 10, 20]:
            for model in ['GaAs_2GHz', 'GaN_2GHz', 'CMOS_28GHz', 'GaN_28GHz', 'custom']:
                # print(f"Testing {model} with {power_backoff_db}dB power backoff")
                x_gt = memoryless_polynomial_nonlinearity(x_time.numpy(), power_backoff_db,
                                                          model_coefficients[model])

                if model == 'custom':
                    model = model_coefficients[model]
                pa_nonlin = PowerAmplifierNonlinearity(power_backoff_db, model)
                x_test = pa_nonlin(x_time)

                np.testing.assert_array_almost_equal(x_gt, x_test, decimal=3)
