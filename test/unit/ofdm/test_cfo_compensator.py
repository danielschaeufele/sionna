#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

from sionna.ofdm import CFOCompensator, OFDMModulator, OFDMDemodulator
from sionna.utils import QAMSource

import unittest
import numpy as np
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


class TestOFDMModulator(unittest.TestCase):
    def test_cfo_compensation_uniform_cyclic_prefix(self):
        sample_rate = 122.88e6
        batch_size = 10
        fft_size = 4096
        num_ofdm_symbols = 14
        cyclic_prefix_length = 288

        qam_source = QAMSource(4)
        modulator = OFDMModulator(cyclic_prefix_length)
        demodulator = OFDMDemodulator(fft_size, 0, cyclic_prefix_length)

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        cfo = np.arange(-500, 500, 100)
        cfo_vec = np.exp(1j * np.arange(x_time.shape[-1], dtype=np.complex64)
            * (np.reshape(cfo, [batch_size, 1]) / sample_rate * 2 * np.pi))
        data_with_cfo = x_time.numpy() * cfo_vec
        cfo_compensator = CFOCompensator(demodulator, return_cfo=True)
        x_compensated, cfo_est = cfo_compensator(data_with_cfo)
        np.testing.assert_array_almost_equal(cfo, cfo_est * sample_rate / (2 * np.pi))
        x_freq = demodulator(x_compensated)
        np.testing.assert_array_almost_equal(x, x_freq)

    def test_cfo_compensation_nonuniform_cyclic_prefix(self):
        sample_rate = 122.88e6
        batch_size = 10
        fft_size = 4096
        num_ofdm_symbols = 14
        cyclic_prefix_length = [352] + [288] * 13

        qam_source = QAMSource(4)
        modulator = OFDMModulator(cyclic_prefix_length)
        demodulator = OFDMDemodulator(fft_size, 0, cyclic_prefix_length)

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        cfo = np.arange(-500, 500, 100)
        cfo_vec = np.exp(1j * np.arange(x_time.shape[-1], dtype=np.complex64)
            * (np.reshape(cfo, [batch_size, 1]) / sample_rate * 2 * np.pi))
        data_with_cfo = x_time.numpy() * cfo_vec
        cfo_compensator = CFOCompensator(demodulator, return_cfo=True)
        x_compensated, cfo_est = cfo_compensator(data_with_cfo)
        np.testing.assert_array_almost_equal(cfo, cfo_est * sample_rate / (2 * np.pi))
        x_freq = demodulator(x_compensated)
        np.testing.assert_array_almost_equal(x, x_freq)
