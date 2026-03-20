#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch

from sionna.phy.nr import PUSCHTransformPrecoder, PUSCHTransformDeprecoder


class TestPUSCHTransformPrecoder:
    """Test PUSCHTransformPrecoder and PUSCHTransformDeprecoder"""

    @staticmethod
    def _load_reference(prbs):
        return np.load(
            Path(__file__).with_name(f"pusch_transform_precoding_{prbs}_prbs.npz")
        )

    def test_precoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = self._load_reference(prbs)
            tp = PUSCHTransformPrecoder(num_subcarriers=12 * prbs)
            x_transform_precoded = tp(
                torch.as_tensor(ref_data["x_layer_mapped"])
            ).cpu().numpy()
            np.testing.assert_array_almost_equal(x_transform_precoded,
                                                 ref_data["x_transform_precoded"])

    def test_deprecoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = self._load_reference(prbs)
            no_eff = np.random.uniform(size=ref_data["x_transform_precoded"].shape)
            tp = PUSCHTransformDeprecoder(num_subcarriers=12 * prbs)
            x_layer_mapped, no_eff_mapped = tp(
                torch.as_tensor(ref_data["x_transform_precoded"]),
                torch.as_tensor(no_eff),
            )
            np.testing.assert_array_almost_equal(x_layer_mapped.cpu().numpy(),
                                                 ref_data["x_layer_mapped"], decimal=5)
            np.testing.assert_array_almost_equal(no_eff_mapped.cpu().numpy(),
                                                 np.full(no_eff.shape, np.mean(no_eff)))

    def test_deprecoder_cov_matrix_against_reference(self):
        for prbs in [2, 270]:
            ref_data = self._load_reference(prbs)
            x_transform_precoded = torch.as_tensor(
                ref_data["x_transform_precoded"].reshape([1, 1, -1, prbs * 12, 1])
            )
            x_layer_mapped_ref = ref_data["x_layer_mapped"].reshape([1, 1, -1, prbs * 12, 1])
            no_eff = np.random.uniform(size=x_transform_precoded.shape)
            no_eff_mat = torch.diag_embed(torch.as_tensor(no_eff.reshape([-1, prbs * 12]))).numpy()
            ifft_mat = np.fft.ifft(np.eye(prbs * 12), norm="ortho")
            full_cov_ref = ifft_mat @ no_eff_mat @ ifft_mat.T.conj()

            tp = PUSCHTransformDeprecoder(num_subcarriers=prbs * 12)
            x_layer_mapped, full_cov = tp(x_transform_precoded, torch.as_tensor(no_eff), True)
            full_cov = full_cov.reshape(-1, prbs * 12, prbs * 12).cpu().numpy()

            np.testing.assert_array_almost_equal(
                x_layer_mapped.cpu().numpy(), x_layer_mapped_ref, decimal=5
            )
            np.testing.assert_array_almost_equal(full_cov, full_cov_ref, decimal=5)

    def test_invalid_subcarrier_count(self):
        with pytest.raises(ValueError):
            PUSCHTransformPrecoder(num_subcarriers=273 * 12)
        with pytest.raises(ValueError):
            PUSCHTransformDeprecoder(num_subcarriers=273 * 12)
