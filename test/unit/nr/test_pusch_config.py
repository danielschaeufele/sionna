#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

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

from sionna.nr import PUSCHConfig, CarrierConfig

class TestPUSCHDMRS(unittest.TestCase):
    """Tests for the PUSCHDMRS Class"""

    def test_against_reference_1(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load("unit/nr/reference_dmrs_1.npy")
        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 1
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [4,4]
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
        self.assertTrue(np.allclose(pilots, reference_dmrs))

    def test_against_reference_2(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load("unit/nr/reference_dmrs_2.npy")
        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [4,4]
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
        self.assertTrue(np.allclose(pilots, reference_dmrs))

    def  test_against_reference_transform_precoding(self):
        """Test that DMRS patterns match a reference implementation"""
        reference_dmrs = np.load("unit/nr/reference_dmrs_transform_precoding.npy")
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
        self.assertTrue(np.allclose(pilots, reference_dmrs))

    def test_orthogonality_over_resource_grid(self):
        """Test that DMRS for different ports are orthogonal
           across a resource grid by computing the LS estimate
           on a noise less block-constant channel
        """
        def ls_estimate(pusch_config):
            """Assigns a random channel coefficient to each port
               and computes the LS estimate
            """
            a = pusch_config.dmrs_grid
            channel = np.random.rand(a.shape[0], 1, 1)
            y = np.sum(channel*a, axis=0)
            for i, port in enumerate(a):
                ind = np.where(port)
                port = port[ind]

                # LS Estimate
                z = y[ind]*np.conj(port)/np.abs(port)**2

                # Time-domain averaging of CDMs for DMRSLength=2
                if pusch_config.dmrs.length==2: 
                    l = len(pusch_config.dmrs_symbol_indices)
                    z = np.reshape(z, [-1, l])
                    z = np.reshape(z, [-1, l//2, 2])
                    z = np.mean(z, axis=-1)

                # Frequency-domain averaging of CDMs
                if pusch_config.dmrs.config_type==1:
                    num_freq_pilots = 6 * pusch_config.carrier.n_size_grid
                else:
                    num_freq_pilots = 4 * pusch_config.carrier.n_size_grid
                z = np.reshape(z, [num_freq_pilots//2,2,-1])
                z = np.mean(z, axis=1)

                return np.allclose(z-channel[i], 0)

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.dmrs_port_set = [1,2,5,11]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.port_set = [2,3,4,5]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.port_set = [0,1,2,3]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))


    def test_precoding_against_reference(self):
        """Test precoded DMRS against reference implementation"""

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 1
        pusch_config.carrier.slot_number = 1
        pusch_config.dmrs.additional_position = 0
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [8,8]
        pusch_config.precoding = "codebook"

        # 1-Layer 2-Antenna Ports
        pusch_config.num_layers = 1
        pusch_config.num_antenna_ports = 2
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(6):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 1-Layer 4-Antenna Ports
        pusch_config.num_layers = 1
        pusch_config.num_antenna_ports = 4
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(28):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 2-Layer 2-Antenna Ports
        pusch_config.num_layers = 2
        pusch_config.num_antenna_ports = 2
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(3):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 2-Layer 4-Antenna Ports
        pusch_config.num_layers = 2
        pusch_config.num_antenna_ports = 4
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(22):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 3-Layer 4-Antenna Ports
        pusch_config.num_layers = 3
        pusch_config.num_antenna_ports = 4
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(7):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 4-Layer 4-Antenna Ports
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        ref = np.load(f"unit/nr/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(5):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))


class TestCarrierConfig(unittest.TestCase):
    """Tests for the CarrierConfig Class"""

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


class TestPUSCHConfig(unittest.TestCase):
    """Tests for the PUSCHConfig class"""

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
