#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PUSCHConfig and related configuration classes."""

from pathlib import Path
import pytest
import numpy as np

from sionna.phy.nr import (
    PUSCHConfig,
    CarrierConfig,
    PUSCHDMRSConfig,
    TBConfig,
    check_pusch_configs,
)


class TestPUSCHConfig:
    """Tests for PUSCHConfig."""

    def test_default_initialization(self):
        """Test default configuration creation."""
        config = PUSCHConfig()

        assert config.mapping_type == "A"
        assert config.num_layers == 1
        assert config.num_antenna_ports == 1
        assert config.symbol_allocation == [0, 14]

    def test_custom_mapping_type(self):
        """Test configuration with custom mapping type."""
        config = PUSCHConfig(mapping_type="B")
        # For mapping type B, symbol_allocation[0] doesn't have to be 0
        config.symbol_allocation = [2, 12]
        config.check_config()

        assert config.mapping_type == "B"

    def test_carrier_config(self):
        """Test carrier configuration integration."""
        carrier = CarrierConfig()
        carrier.n_cell_id = 42
        carrier.subcarrier_spacing = 30

        config = PUSCHConfig(carrier_config=carrier)

        assert config.carrier.n_cell_id == 42
        assert config.carrier.subcarrier_spacing == 30

    def test_dmrs_config(self):
        """Test DMRS configuration integration."""
        dmrs = PUSCHDMRSConfig()
        dmrs.config_type = 2
        dmrs.additional_position = 1

        config = PUSCHConfig(pusch_dmrs_config=dmrs)

        assert config.dmrs.config_type == 2
        assert config.dmrs.additional_position == 1

    def test_tb_config(self):
        """Test transport block configuration integration."""
        tb = TBConfig(channel_type="PUSCH")
        tb.mcs_index = 10

        config = PUSCHConfig(tb_config=tb)

        assert config.tb.mcs_index == 10

    def test_num_layers_validation(self):
        """Test num_layers validation."""
        config = PUSCHConfig()

        with pytest.raises(ValueError):
            config.num_layers = 5

    def test_num_antenna_ports_validation(self):
        """Test num_antenna_ports validation."""
        config = PUSCHConfig()

        with pytest.raises(ValueError):
            config.num_antenna_ports = 3

    def test_mapping_type_validation(self):
        """Test mapping_type validation."""
        config = PUSCHConfig()

        with pytest.raises(ValueError):
            config.mapping_type = "C"

    def test_symbol_allocation_for_mapping_a(self):
        """Test symbol allocation constraints for mapping type A."""
        config = PUSCHConfig(mapping_type="A")
        config.symbol_allocation = [0, 14]
        config.check_config()

        # For mapping type A, first element must be 0
        config.symbol_allocation = [1, 13]
        with pytest.raises(ValueError):
            config.check_config()

    @pytest.mark.parametrize("num_prbs", [1, 50, 100, 275])
    def test_num_resource_blocks(self, num_prbs):
        """Test num_resource_blocks property."""
        config = PUSCHConfig()
        config.n_size_bwp = num_prbs

        assert config.num_resource_blocks == num_prbs

    def test_num_subcarriers(self):
        """Test num_subcarriers property."""
        config = PUSCHConfig()
        config.n_size_bwp = 52

        assert config.num_subcarriers == 52 * 12

    def test_dmrs_symbol_indices(self):
        """Test DMRS symbol indices calculation."""
        config = PUSCHConfig()
        config.dmrs.additional_position = 0

        indices = config.dmrs_symbol_indices
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_dmrs_mask_shape(self):
        """Test DMRS mask shape."""
        config = PUSCHConfig()
        config.n_size_bwp = 10

        mask = config.dmrs_mask

        assert mask.shape[0] == 10 * 12  # num_subcarriers
        assert mask.shape[1] == config.carrier.num_symbols_per_slot
        assert mask.dtype == bool

    def test_dmrs_grid_shape(self):
        """Test DMRS grid shape."""
        config = PUSCHConfig()
        config.n_size_bwp = 10
        config.dmrs.dmrs_port_set = [0]

        grid = config.dmrs_grid

        assert grid.shape[0] == 1  # num_dmrs_ports
        assert grid.shape[1] == 10 * 12  # num_subcarriers
        assert grid.dtype == complex

    def test_phase_correction_sequence(self):
        """Generate carrier signal for upconversion and compare phase shift at
        the start of each symbol with the generated phase correction sequence"""
        for subcarrier_spacing in [15, 30, 60, 120, 240]:
            pusch_config = PUSCHConfig(subcarrier_spacing=subcarrier_spacing,
                                       sample_rate="standard")
            for carrier_frequency in np.arange(1e9, 10e9, .5e9):
                pusch_config.carrier.carrier_frequency = carrier_frequency

                fft_size = pusch_config.fft_size
                cp_lengths = np.round(pusch_config.carrier.cyclic_prefix_length *
                                      pusch_config.sample_rate).astype(int)

                sample_idx = np.arange(-cp_lengths[0], np.sum(cp_lengths[1:])
                                       + fft_size * len(cp_lengths))
                upconversion_vector = np.exp(2.j*np.pi*carrier_frequency*
                                             sample_idx/pusch_config.sample_rate)

                symbol_end_idx = np.cumsum(cp_lengths + fft_size)
                phase_rotations = []
                for idx in symbol_end_idx:
                    phase_rotations.append(upconversion_vector[idx - fft_size])

                np.testing.assert_array_almost_equal(phase_rotations,
                                     pusch_config.phase_correction_sequence)


class TestPUSCHDMRS:
    """Tests for the PUSCHDMRS Class"""

    def test_against_reference_1(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load(Path(__file__).with_name("reference_dmrs_1.npy"))
        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 1
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = (4,4)
        p = []
        for n_cell_id in [0,1,10,24,99,1006]:
            for slot_number in [0,1,5,9]:
                for port_set in [0,3,4,9,11]:
                    pusch_config.carrier.n_cell_id = n_cell_id
                    pusch_config.carrier.slot_number=slot_number
                    pusch_config.dmrs.dmrs_port_set = [port_set]
                    a = pusch_config.dmrs_grid
                    pilots = np.concatenate([a[0,:,2], a[0,:,3], a[0,:,10], a[0,:,11]])
                    pilots = pilots[np.where(pilots)]/np.sqrt(3)
                    p.append(pilots)
        pilots = np.transpose(np.array(p))
        assert np.allclose(pilots, reference_dmrs)

    def test_against_reference_2(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load(Path(__file__).with_name("reference_dmrs_2.npy"))
        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = (4,4)
        p = []
        for n_cell_id in [0,1,10,24,99,1006]:
            for slot_number in [0,1,5,9]:
                for port_set in [0,3,4,9,11]:
                    pusch_config.carrier.n_cell_id = n_cell_id
                    pusch_config.carrier.slot_number=slot_number
                    pusch_config.dmrs.dmrs_port_set = [port_set]
                    a = pusch_config.dmrs_grid
                    pilots = np.concatenate([a[0,:,2], a[0,:,3], a[0,:,10], a[0,:,11]])
                    pilots = pilots[np.where(pilots)]/np.sqrt(3)
                    p.append(pilots)
        pilots = np.transpose(np.array(p))
        assert np.allclose(pilots, reference_dmrs)

    def  test_against_reference_transform_precoding(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load(Path(__file__).with_name("reference_dmrs_transform_precoding.npy"))
        pusch_config = PUSCHConfig()
        pusch_config.transform_precoding = True
        pusch_config.carrier.subcarrier_spacing = 30
        pusch_config.carrier.n_size_grid = 273
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 0
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        p = []
        for n_cell_id in [0, 1, 10, 24, 99, 1006]:
            pusch_config.carrier.n_cell_id = n_cell_id
            a = pusch_config.dmrs_grid
            pilots = np.concatenate([a[0, :, 2], a[0, :, 3], a[0, :, 10], a[0, :, 11]])
            pilots = pilots[np.where(pilots)]/np.sqrt(2)
            p.append(pilots)
        pilots = np.transpose(np.array(p))
        assert np.allclose(pilots, reference_dmrs)


class TestCheckPuschConfigs:
    """Tests for check_pusch_configs function."""

    def test_single_config(self):
        """Test with single configuration."""
        config = PUSCHConfig()
        params = check_pusch_configs([config])

        assert params["num_tx"] == 1
        assert params["num_layers"] == config.num_layers
        assert params["num_subcarriers"] == config.num_subcarriers

    def test_multiple_configs(self):
        """Test with multiple configurations."""
        config1 = PUSCHConfig()
        config2 = PUSCHConfig()

        params = check_pusch_configs([config1, config2])

        assert params["num_tx"] == 2

    def test_invalid_input_type(self):
        """Test that non-list input raises error."""
        config = PUSCHConfig()

        with pytest.raises(TypeError):
            check_pusch_configs(config)

    def test_invalid_element_type(self):
        """Test that non-PUSCHConfig elements raise error."""
        with pytest.raises(TypeError):
            check_pusch_configs([{"invalid": "config"}])


class TestCarrierConfig:
    """Tests for CarrierConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CarrierConfig()

        assert config.n_cell_id == 1
        assert config.subcarrier_spacing == 15
        assert config.cyclic_prefix == "normal"

    def test_subcarrier_spacing_validation(self):
        """Test subcarrier spacing validation."""
        config = CarrierConfig()

        with pytest.raises(ValueError):
            config.subcarrier_spacing = 45

    def test_num_symbols_per_slot(self):
        """Test num_symbols_per_slot property."""
        config = CarrierConfig()
        config.cyclic_prefix = "normal"

        assert config.num_symbols_per_slot == 14

        config.cyclic_prefix = "extended"
        assert config.num_symbols_per_slot == 12

    def test_cyclic_prefix_length(self):
        carrier_config = CarrierConfig(subcarrier_spacing=15, slot_number=0)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, ([5.2] + [4.69]*6)*2, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=15, slot_number=1)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, ([5.2] + [4.69]*6)*2, decimal=2)

        carrier_config = CarrierConfig(subcarrier_spacing=30, slot_number=0)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [2.86] + [2.34]*13, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=30, slot_number=1)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [2.86] + [2.34]*13, decimal=2)

        carrier_config = CarrierConfig(subcarrier_spacing=60, slot_number=0)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [1.69] + [1.17]*13, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=60, slot_number=1)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [1.17] * 14, decimal=2)

        carrier_config = CarrierConfig(subcarrier_spacing=60, slot_number=0, cyclic_prefix='extended')
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [4.17] * 12, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=60, slot_number=1, cyclic_prefix='extended')
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [4.17] * 12, decimal=2)

        carrier_config = CarrierConfig(subcarrier_spacing=120, slot_number=0)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [1.11] + [0.59] * 13, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=120, slot_number=1)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [0.59] * 14, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=120, slot_number=2)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [0.59] * 14, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=120, slot_number=3)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [0.59] * 14, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=120, slot_number=4)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [1.11] + [0.59] * 13, decimal=2)

        carrier_config = CarrierConfig(subcarrier_spacing=240, slot_number=0)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [0.81] + [0.29] * 13, decimal=2)
        carrier_config = CarrierConfig(subcarrier_spacing=240, slot_number=1)
        np.testing.assert_array_almost_equal(carrier_config.cyclic_prefix_length * 1e6, [0.29] * 14, decimal=2)


class TestPUSCHDMRSConfig:
    """Tests for PUSCHDMRSConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PUSCHDMRSConfig()

        assert config.config_type == 1
        assert config.length == 1
        assert config.additional_position == 0

    def test_config_type_validation(self):
        """Test config_type validation."""
        config = PUSCHDMRSConfig()

        with pytest.raises(ValueError):
            config.config_type = 3

    def test_length_validation(self):
        """Test length validation."""
        config = PUSCHDMRSConfig()

        with pytest.raises(ValueError):
            config.length = 3


class TestTBConfig:
    """Tests for TBConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TBConfig(channel_type="PUSCH")

        assert config.channel_type == "PUSCH"
        assert config.mcs_index == 14  # Default is 14 (16-QAM, r=0.54)
        assert config.mcs_table == 1

    def test_mcs_index_validation(self):
        """Test mcs_index validation."""
        config = TBConfig(channel_type="PUSCH")

        with pytest.raises(ValueError):
            config.mcs_index = 30

    def test_channel_type_validation(self):
        """Test channel_type validation."""
        with pytest.raises(ValueError):
            TBConfig(channel_type="INVALID")

    def test_num_bits_per_symbol_from_mcs(self):
        """Test num_bits_per_symbol derived from MCS index."""
        config = TBConfig(channel_type="PUSCH")
        config.mcs_index = 10  # Should give modulation order 4 (16QAM)

        assert config.num_bits_per_symbol == 4

