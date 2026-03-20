#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import sys

sys.path.insert(0, "../")


try:
    import sionna
except ImportError as e:
    import sys

    sys.path.append("../")

import unittest
import numpy as np
import tensorflow as tf

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
                                                 ref_data["x_layer_mapped"], decimal=5)
            np.testing.assert_array_almost_equal(no_eff_mapped,
                                                 np.full(no_eff.shape, np.mean(no_eff)))

    def test_deprecoder_cov_matrix_against_reference(self):
        for prbs in [2, 270]:
            ref_data = np.load(f"unit/nr/pusch_transform_precoding_{prbs}_prbs.npz")
            x_transform_precoded = ref_data["x_transform_precoded"].reshape([1, 1, -1, prbs * 12, 1])
            x_layer_mapped_ref = ref_data["x_layer_mapped"].reshape([1, 1, -1, prbs * 12, 1])
            no_eff = np.random.uniform(size=x_transform_precoded.shape)
            no_eff_mat = tf.linalg.diag(no_eff.reshape([-1, prbs * 12])).numpy()
            ifft_mat = np.fft.ifft(np.eye(prbs * 12), norm="ortho")
            full_cov_ref = ifft_mat @ no_eff_mat @ ifft_mat.T.conj()

            tp = PUSCHTransformDeprecoder(num_subcarriers=prbs * 12)
            x_layer_mapped, full_cov = tp(x_transform_precoded, no_eff, True)
            full_cov = tf.reshape(full_cov, [-1, prbs * 12, prbs * 12])

            np.testing.assert_array_almost_equal(x_layer_mapped, x_layer_mapped_ref, decimal=5)
            np.testing.assert_array_almost_equal(full_cov, full_cov_ref, decimal=5)

    def test_invalid_subcarrier_count(self):
        with self.assertRaises(ValueError):
            PUSCHTransformPrecoder(num_subcarriers=273 * 12)
        with self.assertRaises(ValueError):
            PUSCHTransformDeprecoder(num_subcarriers=273 * 12)
