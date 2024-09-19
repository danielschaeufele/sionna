#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for power amplifier nonlinearities"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class PowerAmplifierNonlinearity(Layer):
    r"""PowerAmplifierNonlinearity(power_backoff_db, model="GaN_2GHz", **kwargs)

        This layer implements a memoryless power amplifier model according to
        3GPP document R4-165901.

        The weights are either taken from R4-165901, when the corresponding
        model name is specified in the `model` parameter or they can be
        supplied directly.

        The formula for the non-linearity is
        :math:`y(n) = \sum_{k \in K} a_k x(n) |x(n)|^{2k}`

        To get reproducible results, the input signal will be normalized to
        have unit power before applying the power backoff. Due to the fact that
        the formula for the non-linearity gives non-plausible results for
        samples with magnitude > 1, all samples with magnitude > 1 will be
        clipped before applying the non-linearity.

        Parameters
        ----------
        power_backoff_db : float
            Power backoff in dB

        model : str, one of ["'GaAs_2GHz", "GaN_2GHz", "CMOS_28GHz", "GaN_28GHz"] or list[complex]
            Either name of the model according to R4-165901 or vector of
            complex model coefficients

        Input
        -----
        x : [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex
            Time-domain input signal, any other dimensions will also work, as
            the non-linearity is applied element-wise.

        Output
        ------
        y : [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex
            Time-domain output signal
    """

    def __init__(self, power_backoff_db, model="GaN_2GHz", **kwargs):
        super().__init__(**kwargs)

        if isinstance(model, str):
            model_coefficients_dict = {
                'GaAs_2GHz': [-0.618347 - 0.785905j, 2.0831 - 1.69506j,
                              -14.7229 + 16.8335j, 61.6423 - 76.9171j,
                              -145.139 + 184.765j, 190.61 - 239.371j,
                              -130.184 + 158.957j, 36.0047 - 42.5192j],
                'GaN_2GHz': [0.999952 - 0.00981788j, -0.0618171 + 0.118845j,
                             -1.69917 - 0.464933j, 3.27962 + 0.829737j,
                             -1.80821 - 0.454331j],
                'CMOS_28GHz': [0.491576 + 0.870835j, -1.26213 + 0.242689j,
                               7.11693 + 5.14105j, -30.7048 - 53.4924j,
                               73.8814 + 169.146j, -96.7955 - 253.635j,
                               65.0665 + 185.434j, -17.5838 - 53.1786j],
                'GaN_28GHz': [-0.334697 - 0.942326j, 0.89015 - 0.72633j,
                              -2.58056 + 4.81215j, 4.81548 - 9.54837j,
                              -4.41452 + 8.63164j, 1.54271 - 2.94034j],
            }
            if model not in model_coefficients_dict:
                raise ValueError(
                    f"Invalid model {model}. Valid models are"
                    f"{', '.join(model_coefficients_dict.keys())}.")
            self._model_coefficients = model_coefficients_dict[model]
        else:
            self._model_coefficients = model

        self._power_backoff_db = float(power_backoff_db)

    def call(self, inputs):
        x = inputs
        # Normalize input power to 1
        scaling = tf.cast(tf.math.sqrt(tf.reduce_mean(tf.math.abs(x) ** 2)),
                     inputs.dtype)
        # Apply power backoff
        scaling *= 10. ** (self._power_backoff_db / 20)
        x /= scaling
        # Clip samples with magnitude > 1
        x = tf.where(tf.math.abs(x) > 1, x / tf.cast(tf.math.abs(x),
                                                     inputs.dtype), x)

        # Apply nonlinearity
        x_nonlinear = tf.zeros_like(x)
        for k, coeff in enumerate(self._model_coefficients):
            x_nonlinear += coeff * x * tf.cast(tf.math.pow(tf.math.abs(x),
                                                   2 * k), inputs.dtype)

        # Revert scaling
        x_nonlinear *= scaling
        return x_nonlinear
