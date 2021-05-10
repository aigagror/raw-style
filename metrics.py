import tensorflow as tf
import tensorflow_probability as tfp
from absl import logging

from generator import PixelImageGenerator
from layers import Preprocess
from style import StyleModel
from utils import load_style_image, load_resize_batch_image


def get_moments(feats, epsilon):
    m1 = [tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True) for feats in feats]
    com2 = [tfp.stats.covariance(feats, sample_axis=[0, 1, 2]) for feats in feats]
    var = [tf.math.reduce_variance(feats, axis=[0, 1, 2], keepdims=True) for feats in feats]
    norm_feats = [(f - m) * tf.math.rsqrt(v + epsilon) for f, m, v in zip(feats, m1, var)]
    m3 = [tf.reduce_mean(z ** 3, axis=[0, 1, 2]) for z in norm_feats]
    m1 = [tf.squeeze(m1, [0, 1, 2]) for m1 in m1]
    return m1, com2, m3


class StyleMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, style_image, feat_model='vgg19'):
        super().__init__()
        self.epsilon = 1e-7
        input_shape = tf.shape(style_image)[1:]
        input, outputs = tf.keras.Input(input_shape), []
        if feat_model == 'vgg19':
            x = Preprocess(tf.keras.applications.vgg19.preprocess_input)(input)
            feat_model = tf.keras.applications.VGG19(include_top=False, input_tensor=x)
            for layer in [f'block{i}_conv1' for i in range(1, 6)]:
                batch_norm = tf.keras.layers.BatchNormalization(momentum=0, epsilon=self.epsilon, scale=False,
                                                                center=False)
                outputs.append(batch_norm(feat_model.get_layer(layer).output))

        elif feat_model == 'debug':
            x = tf.keras.layers.Conv2D(2, 1, kernel_initializer='ones', bias_initializer='zeros')(input)
            x = tf.keras.layers.BatchNormalization(momentum=0, epsilon=self.epsilon, scale=False, center=False)(x)
            outputs.append(x)
        else:
            raise ValueError(f'unknown feature model: {feat_model}')

        self.feat_model = tf.keras.Model(input, outputs)
        self.style_feats = self.feat_model(style_image, training=True)
        if isinstance(self.style_feats, tf.Tensor):
            self.style_feats = [self.style_feats]

        self.m1, self.m2, self.m3 = get_moments(self.style_feats, self.epsilon)

    def get_style_metrics(self):
        gen_image = self.model.generator(None)
        gen_feats = self.feat_model(gen_image, training=False)
        if isinstance(gen_feats, tf.Tensor):
            gen_feats = [gen_feats]

        gen_m1, gen_m2, gen_m3 = get_moments(gen_feats, self.epsilon)

        q1 = tf.reduce_mean([tf.norm(g - s, ord=1) for g, s in zip(gen_m1, self.m1)])
        # Covariance has basically double redundancies, so we divide by 2
        q2 = tf.reduce_mean([tf.norm(g - s, ord=1) / 2 for g, s in zip(gen_m2, self.m2)])
        q3 = tf.reduce_mean([tf.norm(g - s, ord=1) for g, s in zip(gen_m3, self.m3)])
        return q1, q2, q3

    def on_epoch_end(self, epoch, logs=None):
        q1, q2, q3 = self.get_style_metrics()

        if logs is not None:
            logs['q1'] = q1
            logs['q2'] = q2
            logs['q3'] = q3

        logging.info(f'{q1:.3} q1, {q2:.3} q2, {q3:.3} q3')


def assess_gen_style(gen_path, image_size):
    style_image = load_style_image()
    if gen_path is None:
        raise ValueError("gen_path must be specified under the 'assess' command")
    logging.info(f"assessing style quality from '{gen_path}'")
    gen_image = load_resize_batch_image(gen_path, image_size)
    generator = PixelImageGenerator(gen_image)
    style_model = StyleModel(None, generator)
    sm_cbk = StyleMetricCallback(style_image)
    sm_cbk.set_model(style_model)
    q1, q2, q3 = sm_cbk.get_style_metrics()
    logging.info('=' * 50)
    logging.info(f'{q1:.3} mean, {q2:.3} covariance, {q3:.3} skewness')
    logging.info('=' * 50)
