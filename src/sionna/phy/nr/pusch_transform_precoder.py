#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definitions for PUSCH transform precoder and deprecoder"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.signal import fft, ifft


def _check_largest_prime_factor_not_larger_then_5(n):
    for p in [2, 3, 5]:
        while n % p == 0:
            n /= p
    if n > 1:
        raise ValueError("Number of subcarriers shouldn't have a prime factor > 5")


class PUSCHTransformPrecoder(Block):
    r"""
    Performs transform precoding of layer mapped symbols as defined in
    [3GPP38211]_ Sec. 6.3.1.4.
    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the FFT will be applied individually.
    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.
    Parameters
    ----------
        num_subcarriers: int
            Number of subcarriers. The largest prime factor must not be larger
            than 5.
        dtype : One of [tf.complex64, tf.complex128]
            Dtype of inputs and outputs. Defaults to tf.complex64.
    Input
    -----
        inputs: [...,n], tf.complex
            Tensor containing the sequence of symbols to be transform precoded.
    Output
    ------
        : [...,n], tf.complex
            Tensor containing the sequence of symbols that have been transform
            precoded.
    """

    def __init__(self, num_subcarriers, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = fft(y_reshaped, precision=self._precision)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result


class PUSCHTransformDeprecoder(Block):
    r"""
    Performs transform deprecoding of layer mapped symbols as defined in
    [3GPP38211]_ Sec. 6.3.1.4.
    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the IFFT will be applied individually.
    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.
    Parameters
    ----------
        num_subcarriers: int
            Number of subcarriers. The largest prime factor must not be larger
            than 5.
        dtype : One of [tf.complex64, tf.complex128]
            Dtype of inputs and outputs. Defaults to tf.complex64.
    Input
    -----
        y: [...,n,1], tf.complex
            Tensor containing the sequence of symbols after transform precoding.
        no_eff: [...,n,1], tf.complex
            Tensor containing the noise variance of symbols after transform precoding.
        return_cov: bool
            Indicates whether to return the covariance matrix (True) or the diagonal of covariance matrix (False, default).
    Output
    ------
        y : [...,n,1], tf.complex
            Tensor containing the sequence of symbols before transform precoding.
        no_eff : [...,n,1] or [...,n,n,1], tf.complex
            Tensor containing the noise variance of symbols before transform precoding.
    """

    def __init__(self, num_subcarriers, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y, no_eff, return_cov=False):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = ifft(y_reshaped, precision=self._precision)
        y_result = tf.reshape(y_transformed, orig_shape)

        if return_cov:
            no_eff_reshaped = tf.reshape(no_eff, [-1, self._num_subcarriers])
            no_eff_diag = tf.linalg.diag(no_eff_reshaped)
            no_eff = fft(
                ifft(no_eff_diag, precision=self._precision, axis=-2),
                precision=self._precision,
                axis=-1,
            )
            no_shape = tf.concat([orig_shape[:-2], [self._num_subcarriers, self._num_subcarriers]], 0)
            no_eff = tf.reshape(no_eff, no_shape)
        else:
            # Noise power is evenly spread over all subcarriers by IDFT transform
            no_eff = tf.ones(orig_shape) * tf.reduce_mean(no_eff, axis=-2, keepdims=True)
        return y_result, no_eff
