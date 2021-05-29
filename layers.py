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


class NoisyConv(tf.keras.layers.Conv2D):
    def call(self, inputs):
        x = super().call(inputs)
        return x + tf.random.normal(tf.shape(x), stddev=self.std)


class NoisyConvOne(NoisyConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = 1


class NoisyConvHalf(NoisyConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = 0.5


class NoisyConvQuarter(NoisyConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = 0.25


class NoisyConvTenth(NoisyConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = 0.1


class NoBatchNorm(tf.keras.layers.Activation):
    def __init__(self, **kwargs):
        super().__init__('linear')


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, preprocess_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess = preprocess_fn

    def call(self, inputs, **kwargs):
        return self.preprocess(inputs)


class MeasureFeats(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        norms = tf.norm(inputs, axis=-1)
        mean, std = tf.nn.moments(inputs, axes=None)
        self.add_metric(tf.reduce_mean(norms), f'{self.name}.norm')
        self.add_metric(mean, f'{self.name}.mean')
        self.add_metric(std, f'{self.name}.std')
        return inputs
