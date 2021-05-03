import tensorflow as tf
import tensorflow_addons as tfa


class StandardizeRGB(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return (inputs - 128) / 64


class SNConv2D(tfa.layers.SpectralNormalization):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        conv = tf.keras.layers.Conv2D(**config)
        return cls(conv, name=conv.name)


class NoBatchNorm(tf.keras.layers.Activation):
    def __init__(self, **kwargs):
        super().__init__('linear')


class StandardizeFeats(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        super().__init__(scale=False, center=False, momentum=0)


class MeasureFeats(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        norms = tf.norm(inputs, axis=-1)
        mean, std = tf.nn.moments(inputs, axes=None)
        self.add_metric(tf.reduce_mean(norms), f'{self.name}.norm')
        self.add_metric(mean, f'{self.name}.mean')
        self.add_metric(std, f'{self.name}.std')
        return inputs
