import tensorflow as tf
import tensorflow_addons as tfa


class StandardizeRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return (inputs - 128) / 64


class SNConv2D(tfa.layers.SpectralNormalization):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        conv = tf.keras.layers.Conv2D(**config)
        return cls(conv, name=conv.name)


class NoBatchNorm(tf.keras.layers.Activation):
    def __init__(self, *args, **kwargs):
        super().__init__('linear')
