import tensorflow as tf
from tensorflow.keras.layers import Layer


def _check_largest_prime_factor_not_larger_then_5(n):
    for p in [2, 3, 5]:
        while n % p == 0:
            n /= p
    if n > 1:
        raise ValueError("Number of subcarriers shouldn't have a prime factor > 5")


class PUSCHTransformPrecoder(Layer):
    def __init__(self,
                 num_subcarriers,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = tf.cast(tf.sqrt(1 / self._num_subcarriers), self._dtype) * tf.signal.fft(y_reshaped)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result


class PUSCHTransformDeprecoder(Layer):
    def __init__(self,
                 num_subcarriers,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = tf.cast(tf.sqrt(float(self._num_subcarriers)), self._dtype) * tf.signal.ifft(y_reshaped)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result
