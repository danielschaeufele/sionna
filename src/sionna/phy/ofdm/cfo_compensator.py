#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the CFO Compensator"""

from typing import Optional, Tuple
import torch

from sionna.phy import Block
from sionna.phy.ofdm import OFDMDemodulator


class CFOCompensator(Block):
    r"""
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

    :param ofdm_demodulator: OFDMDemodulator from which cyclic prefix length
        and FFT size will be extracted.
    :param return_cfo: If `True`, the estimated CFO is returned as additional
        output. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the default device is
        used.

    :input inputs: [...,num_samples], `torch.complex`
        Tensor containing the time-domain signal along the last dimension.
    :output x_compensated: [...,num_samples], `torch.complex`
        Tensor containing the CFO-compensated time-domain signal along the last dimension.
    :output cfo: [...], `torch.float`
        Estimated CFO values in radians per sample. To convert to Hz, multiply
        with :math:`\frac{f_\mathrm{s}}{2\pi}`.
    """
    def __init__(
            self,
            ofdm_demodulator: OFDMDemodulator,
            return_cfo:bool=False,
            precision: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs
        ):
        super().__init__(precision=precision, device=device, **kwargs)
        self._fft_size = ofdm_demodulator.fft_size
        self._cyclic_prefix_length = ofdm_demodulator.cyclic_prefix_length
        self._return_cfo = return_cfo

    def build(self, input_shape: tuple):
        num_samples = input_shape[-1]

        cp_length = self._cyclic_prefix_length
        if not isinstance(cp_length, torch.Tensor):
            cp_length = torch.as_tensor(cp_length, device=self.device)
        cp_length = cp_length.to(dtype=torch.int64, device=self.device)
        if cp_length.dim() == 0:
            cp_value = int(cp_length.item())
            cp_length = torch.full(
                (num_samples // (self._fft_size + cp_value),),
                cp_value,
                dtype=torch.int64,
                device=self.device,
            )
        self._num_ofdm_symbols = cp_length.shape[0]

        symbol_starts = torch.cumsum(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=self.device),
                    cp_length[:-1] + self._fft_size,
                ]
            ),
            dim=0,
        )
        if num_samples < symbol_starts[-1].item():
            raise ValueError("shape(inputs)[-1] must be larger or equal than samples per slot")

        cp_v1_idx, cp_v2_idx = [], []
        for i in range(self._num_ofdm_symbols):
            cp_range = torch.arange(
                symbol_starts[i],
                symbol_starts[i] + cp_length[i],
                dtype=torch.int64,
                device=self.device,
            )
            cp_v1_idx.append(cp_range)
            cp_v2_idx.append(cp_range + self._fft_size)
        self.register_buffer("_cp_v1_idx", torch.cat(cp_v1_idx, 0))
        self.register_buffer("_cp_v2_idx", torch.cat(cp_v2_idx, 0))


    def call(self, inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        num_samples = inputs.shape[-1]

        cp_v1 = torch.index_select(inputs, -1, self._cp_v1_idx)
        cp_v2 = torch.index_select(inputs, -1, self._cp_v2_idx)

        cfo_values = torch.conj(cp_v1) * cp_v2
        cfo_sum = torch.sum(cfo_values, dim=-1)
        cfo = (torch.angle(cfo_sum) / self._fft_size).unsqueeze(-1)

        sample_idx = torch.arange(num_samples, device=inputs.device)
        cfo_compensation = torch.exp(-1j * sample_idx * cfo)
        x = inputs * cfo_compensation

        if self._return_cfo:
            return x, cfo.squeeze(-1)
        else:
            return x
