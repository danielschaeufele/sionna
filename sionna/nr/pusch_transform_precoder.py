import tensorflow as tf
from tensorflow.keras.layers import Layer


class PUSCHTransformPrecoder(Layer):
    def __init__(self,
                 num_subcarriers,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
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
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = tf.cast(tf.sqrt(float(self._num_subcarriers)), self._dtype) * tf.signal.ifft(y_reshaped)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result
