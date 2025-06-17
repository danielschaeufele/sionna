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

from sionna.phy.nr import PUSCHTransformPrecoder, PUSCHTransformDeprecoder


class TestPUSCHTransformPrecoder(unittest.TestCase):
    """Test PUSCHTransformPrecoder and PUSCHTransformDeprecoder"""

    def test_precoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = np.load(f"unit/nr/pusch_transform_precoding_{prbs}_prbs.npz")
            tp = PUSCHTransformPrecoder(num_subcarriers=12 * prbs)
            x_transform_precoded = tp(ref_data["x_layer_mapped"])
            np.testing.assert_array_almost_equal(x_transform_precoded,
                                                 ref_data["x_transform_precoded"])

    def test_deprecoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = np.load(f"unit/nr/pusch_transform_precoding_{prbs}_prbs.npz")
            no_eff = np.random.uniform(size=ref_data["x_transform_precoded"].shape)
            tp = PUSCHTransformDeprecoder(num_subcarriers=12 * prbs)
            x_layer_mapped, no_eff_mapped = tp(ref_data["x_transform_precoded"], no_eff)
            np.testing.assert_array_almost_equal(x_layer_mapped,
                                                 ref_data["x_layer_mapped"])
            np.testing.assert_array_almost_equal(no_eff_mapped,
                                                 np.full(no_eff.shape, np.mean(no_eff)))

    def test_invalid_subcarrier_count(self):
        with self.assertRaises(ValueError):
            PUSCHTransformPrecoder(num_subcarriers=273 * 12)
        with self.assertRaises(ValueError):
            PUSCHTransformDeprecoder(num_subcarriers=273 * 12)
