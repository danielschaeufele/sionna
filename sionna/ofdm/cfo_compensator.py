#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the CFO Compensator"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class CFOCompensator(Layer):
    r"""
    CFOCompensator(self, ofdm_demodulator, return_cfo=False, **kwargs)

    Computes the carrier frequency offset of the OFDM-modulated time-domain
    input signal and compensates it.

    The CFO compensator exploits the fact that for a signal with CFO
    :math:`\theta`, sample rate :math:`f_\mathrm{s}` and FFT size :math:`L`
    the following property holds when :math:`n` indexes a sample in the cyclic
    prefix:

    .. math::

        x_{n+L} = x_n \exp\left(j 2\pi \theta \frac{L}{f_\mathrm{s}}\right)

        \hat{\theta} = \frac{\angle(x_n^* x_{n+L})}{2 \pi \frac{L}{f_\mathrm{s}}}

    For reduced noise this computation is done for all samples in the cyclic
    prefix and the result is averaged.

    The CFO compensated signal is computed as

     .. math::

        \hat{x}_n = x_n \exp\left(-j 2 \pi \hat{\theta} \frac{n}{f_\mathrm{s}}\right)

    This algorithm only removes the CFO when
    :math:`|\theta| < \frac{\Delta_f}{2}`.

    Parameters
    ----------
    ofdm_modulator : OFDMDemodulator
        OFDMModulator from which cyclic prefix length and FFT size will be
        extracted.

    return_cfo : bool
        If `True`, the estimated CFO is returned as additional output.
        Defaults to `False`.

    Input
    -----
    :[...,num_samples], tf.complex
        Tensor containing the time-domain signal along the last dimension.

    Output
    ------
    x_compensated : [...,num_samples], tf.complex
        Tensor containing the CFO-compensated time-domain signal along the
        last dimension.

    cfo : [...], tf.float
        Estimated CFO values in radians per sample. To convert to Hz, multiply
        with :math:`\frac{f_\mathrm{s}}{2\pi}`
    """
    def __init__(self, ofdm_demodulator, return_cfo=False, **kwargs):
        super().__init__(**kwargs)
        self._fft_size = ofdm_demodulator.fft_size
        self._cyclic_prefix_length = ofdm_demodulator.cyclic_prefix_length
        self._return_cfo = return_cfo

    def build(self, input_shape):
        num_samples = input_shape[-1]

        if isinstance(self._cyclic_prefix_length, int):
            self._cyclic_prefix_length = np.full(input_shape[-1] //
                                 (self._fft_size + self._cyclic_prefix_length),
                                 self._cyclic_prefix_length)
        self._num_ofdm_symbols = self._cyclic_prefix_length.shape[0]

        symbol_starts = tf.math.cumsum(self._cyclic_prefix_length +
                                       self._fft_size, exclusive=True)
        assert num_samples >= symbol_starts[-1],\
            "shape(inputs)[-1] must be larger or equal than samples per slot"

        cp_v1_idx, cp_v2_idx = [], []
        for i in range(self._num_ofdm_symbols):
            cp_range = tf.range(symbol_starts[i], symbol_starts[i] +
                                self._cyclic_prefix_length[i])
            cp_v1_idx.append(cp_range)
            cp_v2_idx.append(cp_range + self._fft_size)
        self._cp_v1_idx = tf.concat(cp_v1_idx, 0)
        self._cp_v2_idx = tf.concat(cp_v2_idx, 0)


    def call(self, inputs):
        batch_dims = tf.shape(inputs)[:-1]
        num_samples = tf.shape(inputs)[-1]

        cp_v1 = tf.gather(inputs, self._cp_v1_idx, axis=-1)
        cp_v2 = tf.gather(inputs, self._cp_v2_idx, axis=-1)

        cfo_values = tf.math.conj(cp_v1) * cp_v2
        cfo_sum = tf.math.reduce_sum(cfo_values, axis=[-1])
        cfo = tf.math.angle(cfo_sum) / self._fft_size
        new_shape = tf.concat([batch_dims, [1]], 0)
        cfo = tf.reshape(cfo, new_shape)

        cfo_compensation = tf.math.exp(-1j * tf.cast(tf.range(num_samples),
                dtype=inputs.dtype) * tf.cast(cfo, dtype=inputs.dtype))
        x = inputs * cfo_compensation

        if self._return_cfo:
            return x, tf.squeeze(cfo, -1)
        else:
            return x
