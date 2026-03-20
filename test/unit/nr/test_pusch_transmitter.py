#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PUSCHTransmitter."""

from pathlib import Path

import pytest
import numpy as np
import torch

from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter


class TestPUSCHTransmitter:
    """Tests for PUSCHTransmitter."""

    def test_basic_initialization(self):
        """Test basic transmitter initialization."""
        config = PUSCHConfig()
        transmitter = PUSCHTransmitter(config)

        assert transmitter.resource_grid is not None
        assert transmitter.pilot_pattern is not None

    def test_frequency_domain_output(self):
        """Test transmitter with frequency domain output."""
        config = PUSCHConfig()
        config.n_size_bwp = 12

        transmitter = PUSCHTransmitter(config, output_domain="freq")

        x, b = transmitter(4)

        # Check output shapes
        assert x.dim() >= 4
        assert b.shape[0] == 4

    def test_time_domain_output(self):
        """Test transmitter with time domain output."""
        config = PUSCHConfig()
        config.n_size_bwp = 12

        transmitter = PUSCHTransmitter(config, output_domain="time")

        x, b = transmitter(4)

        # Time domain output has different shape
        assert x.dim() >= 3

    def test_return_bits_true(self):
        """Test transmitter with return_bits=True."""
        config = PUSCHConfig()
        config.n_size_bwp = 12

        transmitter = PUSCHTransmitter(config, return_bits=True)

        result = transmitter(4)

        assert len(result) == 2
        x, b = result
        assert x is not None
        assert b is not None

    def test_return_bits_false(self):
        """Test transmitter with return_bits=False (provide bits)."""
        config = PUSCHConfig()
        config.n_size_bwp = 12

        transmitter = PUSCHTransmitter(config, return_bits=False)

        # Generate input bits with correct shape
        tb_size = transmitter._tb_size
        num_tx = transmitter._num_tx
        bits = torch.randint(0, 2, (4, num_tx, tb_size), dtype=torch.float32)

        x = transmitter(bits)

        # Only signal is returned
        assert isinstance(x, torch.Tensor)

    def test_multiple_transmitters(self):
        """Test with multiple transmitter configurations."""
        config1 = PUSCHConfig()
        config1.n_size_bwp = 12
        config1.n_rnti = 1001

        config2 = PUSCHConfig()
        config2.n_size_bwp = 12
        config2.n_rnti = 1002

        transmitter = PUSCHTransmitter([config1, config2])

        x, b = transmitter(4)

        # Should have 2 transmitters
        assert transmitter._num_tx == 2

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_different_num_layers(self, num_layers):
        """Test with different number of layers."""
        config = PUSCHConfig()
        config.n_size_bwp = 12
        config.num_layers = num_layers
        config.num_antenna_ports = num_layers

        transmitter = PUSCHTransmitter(config)

        x, b = transmitter(2)

        assert x is not None

    def test_codebook_precoding(self):
        """Test with codebook precoding."""
        config = PUSCHConfig()
        config.n_size_bwp = 12
        config.num_layers = 1
        config.num_antenna_ports = 2
        config.precoding = "codebook"
        config.tpmi = 0

        transmitter = PUSCHTransmitter(config)

        x, b = transmitter(2)

        # Output should have num_antenna_ports antennas
        assert transmitter._precoder is not None

    def test_output_dtype(self):
        """Test that output has correct dtype."""
        config = PUSCHConfig()
        config.n_size_bwp = 12

        transmitter = PUSCHTransmitter(config)

        x, b = transmitter(2)

        assert x.is_complex()
        assert not b.is_complex()


class TestPUSCHTransmitterEdgeCases:
    """Edge case tests for PUSCHTransmitter."""

    def test_single_prb(self):
        """Test with single PRB."""
        config = PUSCHConfig()
        config.n_size_bwp = 1

        transmitter = PUSCHTransmitter(config)
        x, b = transmitter(1)

        assert x is not None

    def test_maximum_prbs(self):
        """Test with maximum number of PRBs."""
        config = PUSCHConfig()
        config.n_size_bwp = 275

        transmitter = PUSCHTransmitter(config)
        x, b = transmitter(1)

        assert x is not None

    def test_short_symbol_allocation(self):
        """Test with short symbol allocation."""
        config = PUSCHConfig()
        config.n_size_bwp = 12
        config.symbol_allocation = [0, 4]

        transmitter = PUSCHTransmitter(config)
        x, b = transmitter(1)

        assert x is not None

    def test_against_reference_transform_precoding(self):
        """Test PUSCHTransmitter output against reference MATLAB implementation
        with transform precoding enabled"""
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = 30
        pusch_config.carrier.n_size_grid = 273
        pusch_config.carrier.n_cell_id = 1
        pusch_config.n_rnti = 42
        pusch_config.tb.mcs_index = 9
        pusch_config.transform_precoding = True
        pusch_config.dmrs.n_sid = 3

        ref_data = np.load(
            Path(__file__).with_name("pusch_transmitter_transform_precoding.npz")
        )

        pusch_transmitter = PUSCHTransmitter(pusch_config, return_bits=False)
        x_grid = pusch_transmitter(ref_data["bits"]).cpu().numpy()

        np.testing.assert_array_almost_equal(x_grid, ref_data["grid"])

    def test_modulation_parameters(self):
        """Test FFT size and cyclic prefix derived from standard sampling."""
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = 30
        pusch_config.n_size_bwp = 273
        pusch_config.sample_rate = "standard"

        transmitter = PUSCHTransmitter(pusch_config, output_domain="time")
        np.testing.assert_equal(pusch_config.fft_size, 4096)
        np.testing.assert_almost_equal(pusch_config.sample_rate, 122.88e6)
        np.testing.assert_equal(transmitter.resource_grid.fft_size, 4096)
        np.testing.assert_equal(transmitter.resource_grid.num_effective_subcarriers, 3276)
        np.testing.assert_array_equal(
            transmitter._ofdm_modulator.cyclic_prefix_length.cpu().numpy(),
            [352] + [288] * 13,
        )

