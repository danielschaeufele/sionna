#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH transform precoder and deprecoder."""

from typing import Optional, Tuple
import torch

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
    This block performs transform precoding of layer mapped symbols as defined
    in :cite:p:`3GPP38211` Sec. 6.3.1.4.
    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the FFT will be applied individually.

    :param num_subcarriers: Number of subcarriers. The largest prime factor must
        not be larger than 5.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the default device is
        used.

    :input y: [...,n], `torch.complex`.
            Tensor containing the sequence of symbols to be transform precoded.
    :output: [...,n], `torch.complex`.
            Tensor containing the sequence of symbols that have been transform
            precoded.
    """

    def __init__(
            self,
            num_subcarriers: int,
            precision: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs
        ):
        super().__init__(precision=precision, device=device, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y: torch.Tensor) -> torch.Tensor:
        y_reshaped = y.reshape(-1, self._num_subcarriers)
        y_transformed = fft(y_reshaped, precision=self.precision)
        return y_transformed.reshape(y.shape)


class PUSCHTransformDeprecoder(Block):
    r"""
    Performs transform deprecoding of layer mapped symbols as defined in
    :cite:p:`3GPP38211` Sec. 6.3.1.4.
    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the IFFT will be applied individually.
    
    :param num_subcarriers: Number of subcarriers. The largest prime factor must
        not be larger than 5.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the default device is
        used.

    :input y: [...,n,1], `torch.complex`.
            Tensor containing the sequence of symbols after transform precoding.
    :input no_eff: [...,n,1], `torch.complex`.
            Tensor containing the noise variance of symbols after transform
            precoding.
    :input return_cov: bool
            Indicates whether to return the covariance matrix (True) or the
            diagonal of covariance matrix (False, default).
        
    :output y: [...,n,1], `torch.complex`
            Tensor containing the sequence of symbols before transform precoding.
    :output no_eff: [...,n,1] or [...,n,n,1], `torch.complex`
            Tensor containing the noise variance or covariance matrix of
            symbols before transform precoding.
    """

    def __init__(
            self,
            num_subcarriers: int,
            precision: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs
        ):
        super().__init__(precision=precision, device=device, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(
            self, y: torch.Tensor,
            no_eff: torch.Tensor,
            return_cov: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_reshaped = y.reshape(-1, self._num_subcarriers)
        y_transformed = ifft(y_reshaped, precision=self.precision)
        y_result = y_transformed.reshape(y.shape)

        if return_cov:
            no_eff_reshaped = no_eff.reshape(-1, self._num_subcarriers)
            no_eff_diag = torch.diag_embed(no_eff_reshaped)
            no_eff = fft(
                ifft(no_eff_diag, precision=self.precision, axis=-2),
                precision=self.precision,
                axis=-1,
            )
            no_eff = no_eff.reshape(*y.shape[:-2], self._num_subcarriers, self._num_subcarriers)
        else:
            # Noise power is evenly spread over all subcarriers by IDFT transform
            no_eff = torch.ones_like(no_eff) * no_eff.mean(dim=-2, keepdim=True)
        return y_result, no_eff
