import tensorflow as tf
import tensorflow_addons as tfa


class StandardizeRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return (inputs - 128) / 64


class SNConv2D(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        conv = tf.keras.layers.Conv2D(*args, **kwargs)
        super().__init__(name=conv.name)
        self.sn_conv = tfa.layers.SpectralNormalization(conv, name=conv.name)

    def call(self, inputs, *args, **kwargs):
        return self.sn_conv(inputs, *args, **kwargs)


class NoBatchNorm(tf.keras.layers.Activation):
    def __init__(self, *args, **kwargs):
        super().__init__('linear')
