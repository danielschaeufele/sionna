#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the OFDM Modulator"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.signal import ifftshift
from sionna.utils import flatten_last_dims
from sionna.signal import ifft


class OFDMModulator(Layer):
    # pylint: disable=line-too-long
    """
    OFDMModulator(cyclic_prefix_length=0, **kwargs)

    Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix.

    Parameters
    ----------
    cyclic_prefix_length : int or list[int] or np.ndarray[int]
        Integer or vector of integers indicating the length of the cyclic
        prefix that it prepended to each OFDM symbol.

    Input
    -----
    : [...,num_ofdm_symbols,fft_size], tf.complex
        A resource grid in the frequency domain.

    Output
    ------
    : [...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], tf.complex
        Time-domain OFDM signal.
    """

    def __init__(self, cyclic_prefix_length=0, **kwargs):
        super().__init__(**kwargs)
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            assert (np.issubdtype(value.dtype, np.integer) and
                    value.ndim == 1 and np.all(value >= 0)),\
                ("`cyclic_prefix_length` must be a 1D array with"
                 " only nonnegative integers.")
        else:
            assert isinstance(value, int) and value >=0,\
                "`cyclic_prefix_length` must be a nonnegative integer."
        self._cyclic_prefix_length = value

    def build(self, input_shape):
        fft_size = input_shape[-1]
        num_ofdm_symbols = input_shape[-2]

        if isinstance(self.cyclic_prefix_length, int):
            self.cyclic_prefix_length = np.full(num_ofdm_symbols,
                                                self.cyclic_prefix_length)
        else:
            assert len(self.cyclic_prefix_length) == num_ofdm_symbols,\
                ("shape(inputs)[-2] must match size of lists of cyclic"
                 " prefix lengths")

        gather_idx = []
        for i, cp_len in enumerate(self.cyclic_prefix_length):
            assert cp_len <= fft_size, \
                ("shape(inputs)[-1] must not be smaller than"
                 " `cylic_prefix_length`")

            gather_idx.append(tf.range(fft_size * (i + 1) - cp_len,
                                       fft_size * (i + 1), dtype=tf.int32))
            gather_idx.append(tf.range(fft_size * i,
                                       fft_size * (i + 1), dtype=tf.int32))

        self._gather_idx = tf.concat(gather_idx, 0)

    def call(self, inputs):
        fft_size = tf.shape(inputs)[-1]
        num_ofdm_symbols = tf.shape(inputs)[-2]
        batch_dims = tf.shape(inputs)[:-2]

        # Shift DC subcarrier to first position
        inputs = ifftshift(inputs, axes=-1)

        # Compute IFFT along the last dimension
        x = ifft(inputs)

        # Reshape into slots
        new_shape = tf.concat([batch_dims, [num_ofdm_symbols * fft_size]], 0)
        x = tf.reshape(x, new_shape)

        # Add cyclic prefix
        x = tf.gather(x, self._gather_idx, axis=-1)

        return x
