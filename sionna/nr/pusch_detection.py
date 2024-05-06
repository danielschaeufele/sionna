import tensorflow as tf

import sionna
from sionna.utils import split_dim
from .pusch_transform_precoder import PUSCHTransformDeprecoder


class LinearTransformPrecodingMimoDetector(sionna.mimo.detection.LinearDetector):
    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 num_subcarriers,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(equalizer, output, demapping_method, constellation_type, num_bits_per_symbol, constellation,
                         hard_out, dtype, **kwargs)
        self._transform_deprecoder = PUSCHTransformDeprecoder(num_subcarriers, dtype)

    def call(self, inputs):
        x_hat, no_eff = self._equalizer(*inputs)
        x_transform_deprecoded = self._transform_deprecoder(x_hat)
        z = self._demapper([x_transform_deprecoded, no_eff])

        # Reshape to the expected output shape
        num_streams = tf.shape(inputs[1])[-1]
        if self._output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = split_dim(z, [num_streams, num_bits_per_symbol], tf.rank(z) - 1)

        return z


class LinearTransformPrecodingDetector(sionna.ofdm.detection.OFDMDetector):
    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        # Instantiate the linear detector
        detector = LinearTransformPrecodingMimoDetector(equalizer=equalizer,
                                                        output=output,
                                                        demapping_method=demapping_method,
                                                        num_subcarriers=resource_grid.num_effective_subcarriers,
                                                        constellation_type=constellation_type,
                                                        num_bits_per_symbol=num_bits_per_symbol,
                                                        constellation=constellation,
                                                        hard_out=hard_out,
                                                        dtype=dtype,
                                                        **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)
