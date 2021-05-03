from functools import partial

import tensorflow as tf
from absl import logging

from utils import load_resize_batch_image


class ImageGenerator(tf.keras.layers.Layer):
    def clip_rgb(self):
        raise NotImplementedError()

    def get_gen_image(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()


class PixelImageGenerator(ImageGenerator):
    def __init__(self, gen_image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_image = tf.Variable(gen_image, trainable=True, name='gen_image')

    def call(self, inputs, *args, **kwargs):
        return self.gen_image

    def clip_rgb(self):
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_image, tf.uint8))

    def summary(self):
        print(self.__class__.__name__)


class DeepImageGenerator(ImageGenerator):
    def __init__(self, seed, gen_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.gen_model = gen_model

    def call(self, inputs, *args, **kwargs):
        return self.gen_model(self.seed, *args, **kwargs)

    def clip_rgb(self):
        pass  # RGB [0, 255] should already be enforced within self.gen_model

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_model(self.seed, training=False), tf.uint8))

    def summary(self):
        return self.gen_model.summary()


def make_karras_generator(output_shape, start_hdim=15, max_dim=512, dropout=0, lrelu=0.2):
    input_shape = [1, 1, 1, min(2 ** start_hdim, max_dim)]
    seed = tf.Variable(tf.random.normal(input_shape), trainable=True, name='seed')

    input = tf.keras.Input(input_shape[1:])
    x = input
    for i, j in enumerate(range(start_hdim, 1, -1)):
        h1, h2 = min(2 ** j, max_dim), min(2 ** (j - 1), max_dim)
        first_kernel_size = 4 if i == 0 else 3
        x = tf.keras.layers.Conv2D(h1, first_kernel_size, padding='same', name=f'gen_block{i + 1}_conv1')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'gen_block{i + 1}_lrelu1')(x)

        x = tf.keras.layers.Conv2D(h2, 3, padding='same', name=f'gen_block{i + 1}_conv2')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'gen_block{i + 1}_lrelu2')(x)

        if x.shape[1] == output_shape[1]:
            x = tf.keras.layers.Conv2D(3, 1, name='to_rgb', activation='sigmoid')(x)
            x = x * 255
            break
        else:
            x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)

    return seed, tf.keras.Model(input, x, name='generator')


def make_generator(output_shape, gen_path=None, gen_model=None, dropout=0, lrelu=0.2):
    if gen_model is None:
        if gen_path is not None:
            gen_image = load_resize_batch_image(gen_path, int(output_shape[1]))
            logging.info(f"loaded generated image from '{gen_path}'")
        else:
            gen_image = tf.random.uniform(output_shape, maxval=256, dtype=tf.float32)
        gen_image = tf.cast(gen_image, tf.float32)
        logging.info(f"generated image initialized as random uniform")
        generator = PixelImageGenerator(gen_image)
    else:
        assert gen_path is None
        gen_model_fn_dict = {
            'KarrasGen': partial(make_karras_generator, output_shape, dropout=dropout, lrelu=lrelu),
            'SmallKarrasGen': partial(make_karras_generator, output_shape, max_dim=128, dropout=dropout, lrelu=lrelu),
            'TinyKarrasGen': partial(make_karras_generator, output_shape, max_dim=64, dropout=dropout, lrelu=lrelu),
        }

        seed, gen_model = gen_model_fn_dict[gen_model]()
        generator = DeepImageGenerator(seed, gen_model)

    return generator
