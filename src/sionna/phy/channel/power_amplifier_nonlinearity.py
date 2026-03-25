#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for power amplifier nonlinearities"""

import math
from typing import Optional
import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator


class PowerAmplifierNonlinearity(Block):
    r"""
    This block implements the Generalized Memory Polynomial (GMP) model
    for power amplifiers according to 3GPP document R4-165901.
    The weights are either taken from R4-165901, when the corresponding
    model name is specified in the `model` parameter or they can be
    supplied directly.

    The weights should be specified as a list of tuples of the form
    (l, m, coeffs), where l is the signal term delay in samples, m is the
    envelope term delay in samples, and coeffs is a list of complex
    coefficients for the polynomial terms.

    The Models `GaAs_2GHz`, `GaN_2GHz`, `CMOS_28GHz`, `GaN_28GHz` are
    simplified memoryless polynomial models, which do not contain
    cross-terms (i.e., terms with m != 0 or l !=0 ). The models
    `GaAs_2GHz_GMP`, `GaN_2GHz_GMP`, `CMOS_28GHz_GMP`, `GaN_28GHz_GMP`
    are full GMP models that contain cross-terms up to order m = 3.

    For GMP models, the sample rate of the input signal should be close
    to the sample rate at which the model was measured for the non-linearity
    to be accurate. For the models from R4-165901, the sample rates are given
    in the `sample_rate` property of the corresponding model. For custom
    models, the sample rate should be specified when instantiating the model
    if it contains cross-terms.

    To get reproducible results, the input signal will be normalized to
    have unit power before applying the power backoff. Due to the fact that
    the formula for the non-linearity gives non-plausible results for
    samples with magnitude > 1, all samples with magnitude > 1 will be
    clipped before applying the non-linearity.
    
    :param power_backoff_db: Power backoff in dB
    :param model: 
        Either name of the model according to R4-165901 or list of model
        coefficients.
    :param sample_rate: Sample rate in Hz. If the model is specified by name,
        the sample rate is taken from R4-165901. If the model is specified by
        coefficients, this parameter must be specified if the model contains
        cross-terms (i.e., terms with m != 0) and should not be specified
        otherwise.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    
    :input x: [..., num_time_samples], tf.complex
        Time-domain input signal. Sample rate should be close to `sample_rate`
        for the non-linearity to be accurate, If `sample_rate` is None, the
        sample rate can be arbitrary.
    
    :output y: [..., num_time_samples], tf.complex
        Time-domain output signal.
    """

    def __init__(
            self,
            power_backoff_db: float,
            model: str | list[tuple[int, int, list[complex]]] = "GaN_2GHz",
            sample_rate: Optional[float] = None,
            precision: Optional[Precision] = None,
            device: Optional[str] = None,
            **kwargs):
        super().__init__(precision=precision, device=device, **kwargs)

        if isinstance(model, str):
            model_coefficients_dict = {
                'GaAs_2GHz': [(0, 0, [-0.618347 - 0.785905j, 2.0831 - 1.69506j,
                                      -14.7229 + 16.8335j, 61.6423 - 76.9171j,
                                      -145.139 + 184.765j, 190.61 - 239.371j,
                                      -130.184 + 158.957j, 36.0047 - 42.5192j])],
                'GaN_2GHz': [(0, 0, [0.999952 - 0.00981788j, -0.0618171 + 0.118845j,
                                     -1.69917 - 0.464933j, 3.27962 + 0.829737j,
                                     -1.80821 - 0.454331j])],
                'CMOS_28GHz': [(0, 0, [0.491576 + 0.870835j, -1.26213 + 0.242689j,
                                       7.11693 + 5.14105j, -30.7048 - 53.4924j,
                                       73.8814 + 169.146j, -96.7955 - 253.635j,
                                       65.0665 + 185.434j, -17.5838 - 53.1786j])],
                'GaN_28GHz': [(0, 0, [-0.334697 - 0.942326j, 0.89015 - 0.72633j,
                                      -2.58056 + 4.81215j, 4.81548 - 9.54837j,
                                      -4.41452 + 8.63164j, 1.54271 - 2.94034j])],
                 'GaAs_2GHz_GMP': [
                    (2, 0, [0.0145707 + 0.00223568j, 0.0166021 + 0.0884597j, -0.170987 - 0.889998j, 0.398012 + 4.25717j,
                            -0.922915 - 11.5296j, 1.51648 + 16.8822j, -1.31708 - 12.4992j, 0.443603 + 3.66282j]),
                    (1, 0, [-0.0730384 - 0.0608598j, 0.316437 - 0.130488j, -2.64289 + 1.95766j, 13.9617 - 8.92706j,
                            -35.9884 + 25.271j, 49.5323 - 38.9777j, -34.8388 + 30.1032j, 9.83576 - 9.12289j]),
                    (0, 0, [-0.369392 - 0.616894j, 0.582141 - 1.54129j, -4.2332 + 13.9746j, 15.4346 - 56.4738j,
                            -34.026 + 106.817j, 42.3779 - 83.2642j, -26.6004 + 5.86237j, 6.4982 + 15.2082j]),
                    (-1, 0, [-0.109009 - 0.0382752j, 1.34619 - 0.303139j, -7.57533 + 2.07457j, 30.8214 - 7.83883j,
                             -71.9119 + 13.7515j, 94.7172 - 10.8742j, -65.1891 + 3.05573j, 18.1882 + 0.14561j]),
                    (-2, 0, [0.0913878 + 0.029207j, -0.205695 - 0.0047561j, 0.436792 + 0.098933j, -0.0447736 + 0.802472j,
                            -1.91069 - 3.64271j, 3.53201 + 6.10853j, -2.64467 - 4.81807j, 0.741402 + 1.46945j]),
                    (0, 1, [-0.0732748 - 0.0617029j, 1.04861 + 0.216692j, -7.53774 - 2.85579j, 29.348 + 11.8762j,
                            -68.0727 - 19.8783j, 92.0079 + 4.93057j, -66.4247 + 17.5978j, 19.6384 - 12.2782j]),
                    (0, -1, [-0.108885 - 0.0392921j, -0.65351 - 0.122316j, 2.7747 + 3.26333j, -6.41902 - 23.391j,
                             9.68476 + 79.745j, -10.5191 - 141.613j, 6.89414 + 125.231j, -1.92908 - 43.441j])
                ],
                'GaN_2GHz_GMP': [
                    (2, 0, [-0.0625941 - 0.0142818j, 0.0956533 + 0.00900184j, -0.197256 - 0.0252242j,
                            0.235044 + 0.0097242j, -0.101881 + 0.00776414j]),
                    (1, 0, [0.176832 + 0.0265921j, -0.411554 - 0.0417628j, 0.795672 + 0.146965j,
                            -0.904609 - 0.134671j, 0.364885 + 0.0256412j]),
                    (0, 0, [0.930707 - 0.0506493j, -0.134627 + 0.195504j, -1.4589 - 0.410569j,
                            2.97014 + 0.552334j, -1.66244 - 0.229841j]),
                    (-1, 0, [-0.000408452 + 0.0188736j, 0.573671 - 0.0891485j, -1.43878 - 0.0446107j,
                            1.88831 + 0.11494j, -0.898231 - 0.0576903j]),
                    (-2, 0, [-0.114268 + 0.0207177j, -0.163861 - 0.0420654j, 0.454916 + 0.223106j,
                            -0.606208 - 0.294749j, 0.279233 + 0.126344j]),
                    (0, 3, [0.0946171 - 0.0134503j, -0.22721 + 0.102407j, 0.825701 - 0.485074j,
                            -1.35047 + 0.945727j, 0.754396 - 0.612916j]),
                    (0, -3, [-0.0238986 + 0.00753547j, 0.224223 - 0.0511775j, -0.811315 + 0.176395j,
                            1.31147 - 0.269401j, -0.699496 + 0.152096j])
                ],
                'CMOS_28GHz_GMP': [
                    (1, 0, [-0.0109821 + 0.00313982j, -0.00397658 - 0.0427409j, -0.171194 + 0.151692j, 0.879844 - 0.0235651j,
                            -1.97684 - 0.862044j, 2.32524 + 1.99694j, -1.34472 - 1.77602j, 0.289959 + 0.559338j]),
                    (0, 0, [0.473465 + 0.860276j, -0.953417 + 0.640666j, 1.9899 - 2.3847j, 7.5417 + 6.38381j,
                            -64.8415 - 60.8762j, 159.01 + 189.579j, -167.466 - 225.579j, 65.4247 + 92.5967j]),
                    (-1, 0, [0.0164844 + 0.00671299j, -0.0198519 + 0.177212j, 0.669594 - 0.543745j, -2.98038 - 0.279477j,
                            6.6717 + 4.50511j, -8.26935 - 9.04627j, 5.42365 + 7.52782j, -1.47259 - 2.32623j]),
                    (0, 1, [-0.000292543 - 0.0150556j, -0.122202 - 0.283752j, 2.56792 + 4.68957j, -18.4244 - 34.2816j,
                            66.3648 + 126.766j, -124.066 - 239.871j, 115.273 + 220.218j, -42.1527 - 77.6225j]),
                    (0, -1, [0.0163452 + 0.00969618j, -0.281971 - 0.188069j, 3.35025 + 3.60649j, -24.5434 - 31.1539j,
                            87.5451 + 124.093j, -157.821 - 243.086j, 139.85 + 227.416j, -48.6255 - 81.0794j])
                ],
                'GaN_28GHz_GMP': [
                    (2, 0, [0.023307 + 0.0467845j, -0.0257521 + 0.0511316j, 0.083841 - 0.334476j,
                            -0.168793 + 0.770187j, 0.161316 - 0.770897j, -0.0568524 + 0.279384j]),
                    (1, 0, [-0.045146 - 0.16848j, 0.131447 - 0.1201j, -0.320679 + 0.930956j,
                            0.604716 - 2.09601j, -0.594149 + 2.05955j, 0.22185 - 0.744002j]),
                    (0, 0, [-0.268916 - 0.707247j, 0.722109 - 0.647857j, -2.04126 + 3.97994j,
                            3.57012 - 7.51441j, -3.00197 + 6.42268j, 0.936088 - 2.05401j]),
                    (-1, 0, [-0.0539225 - 0.119444j, 0.081078 - 0.0363615j, -0.297265 + 0.246711j,
                            0.591961 - 0.510012j, -0.542816 + 0.502644j, 0.186803 - 0.187022j]),
                    (-2, 0, [0.022577 + 0.04227j, 0.0085171 - 0.00686566j, -0.0110846 + 0.0177386j,
                            0.0157497 - 0.00255606j, -0.0231175 - 0.0213148j, 0.012631 + 0.0109949j]),
                    (0, 3, [-0.00997684 - 0.0214876j, 0.04625 - 0.0124587j, -0.315178 + 0.16066j,
                            0.841832 - 0.395568j, -1.02048 + 0.442096j, 0.463711 - 0.197228j]),
                    (0, -3, [-0.0138413 - 0.0283711j, 0.0103081 - 0.0570896j, -0.0723643 + 0.440087j,
                            0.399287 - 1.24045j, -0.712003 + 1.50792j, 0.402778 - 0.661505j])
                ],
            }
            model_sample_rate_dict = {
                'GaAs_2GHz': None,
                'GaN_2GHz': None,
                'CMOS_28GHz': None,
                'GaN_28GHz': None,
                'GaAs_2GHz_GMP': 307.2e6,
                'GaN_2GHz_GMP': 200e6,
                'CMOS_28GHz_GMP': 2.281e9,
                'GaN_28GHz_GMP': 2.281e9,
            }
            if model not in model_coefficients_dict:
                raise ValueError(
                    f"Invalid model {model}. Valid models are"
                    f"{', '.join(model_coefficients_dict.keys())}.")
            self._model_coefficients = model_coefficients_dict[model]
            self._sample_rate = model_sample_rate_dict[model]
        else:
            self._model_coefficients = model
            self._sample_rate = sample_rate

        self._power_backoff_db = float(power_backoff_db)

    @property
    def sample_rate(self) -> Optional[float]:
        r"""
        Sample rate that this model was measured at in Hz. The time-domain
        input should be sampled at this rate for the non-linearity to be
        accurate. If `None`, the model is assumed to be independent of the
        sample rate because there are no cross-terms.
        """
        return self._sample_rate

    def call(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input power to 1
        scaling = torch.sqrt(torch.mean(torch.abs(x) ** 2))
        # Apply power backoff
        scaling *= 10. ** (self._power_backoff_db / 20)
        x /= scaling
        # Clip samples with magnitude > 1
        x_abs = torch.abs(x)
        x = torch.where(x_abs > 1, x / x_abs, x)
        x_abs = torch.abs(x)

        # Apply nonlinearity
        x_nonlinear = torch.zeros_like(x)
        for l, m, coeffs in self._model_coefficients:
            signal_term = torch.roll(x, shifts=l, dims=-1)
            envelope_term = torch.roll(x_abs, shifts=l + m, dims=-1)
            for k, coeff in enumerate(coeffs):
                x_nonlinear += coeff * signal_term * envelope_term.pow(2 * k)

        # Revert scaling
        x_nonlinear *= scaling
        return x_nonlinear


class FreqDomainPowerAmplifierNonlinearity(Block):
    r"""
    This block implements a frequency-domain wrapper around a power amplifier
    nonlinearity that works in time-domain.

    The specified input sample rate is compared to the sample rate
    specified in `pa.target_sample_rate` and the closest integer upsampling
    rate is chosen. The actual upsampling is then performed by zero-padding
    the input to the OFDM modulation.

    :param pa: Time-domain power amplifier nonlinearity.
    :param fft_size: FFT size of the OFDM modulation before upsampling.
    :param sample_rate: Sample rate of the input signal in Hz. This is used to
        determine the upsampling factor. If the PA model is independent of the
        sample rate, i.e. `pa.sample_rate` is `None`, this will be ignored.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., num_ofdm_symbols, fft_size], torch.complex
        Frequency-domain input signal.

    :output y: [..., num_ofdm_symbols, fft_size], torch.complex
        Frequency-domain output signal.
    """
    def __init__(
            self,
            pa: PowerAmplifierNonlinearity,
            fft_size: int,
            sample_rate: Optional[float] = None,
            precision: Optional[Precision] = None,
            device: Optional[str] = None,
            **kwargs
        ):
        super().__init__(precision=precision, device=device, **kwargs)
        if pa.sample_rate is not None:
            if sample_rate is None:
                raise ValueError("sample_rate must be specified when pa.sample_rate is not None.")
            self._upsampling_factor = round(pa.sample_rate / sample_rate)
        else:
            self._upsampling_factor = 1

        self._orig_fft_size = fft_size
        self._upsampled_fft_size = fft_size * self._upsampling_factor
        self._zero_pads = (self._upsampling_factor - 1) * fft_size // 2
        self._ofdm_modulator = OFDMModulator()
        self._ofdm_demodulator = OFDMDemodulator(self._upsampled_fft_size, 0)
        self._pa = pa

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == self._orig_fft_size, f"Last dimension of input must be {self._orig_fft_size}."
        x_freq = torch.nn.functional.pad(inputs, (self._zero_pads, self._zero_pads))
        x_time = self._ofdm_modulator(x_freq) * math.sqrt(self._upsampling_factor)
        time_shape = x_time.shape
        x_time = self._pa(x_time.reshape(-1, self._upsampled_fft_size)).reshape(time_shape)
        x_freq = self._ofdm_demodulator(x_time) / math.sqrt(self._upsampling_factor)
        if self._zero_pads > 0:
            x_freq = x_freq[..., self._zero_pads : -self._zero_pads]
        return x_freq
