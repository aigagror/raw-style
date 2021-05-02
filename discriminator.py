from functools import partial

import tensorflow as tf
from absl import flags
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.utils import layer_utils

from layers import SNConv2D, StandardizeRGB, NoBatchNorm, MeasureFeats

FLAGS = flags.FLAGS


def make_resnet152v2(input_tensor):
    resnet.layers = tf.keras.layers
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(input_tensor)
    x = resnet.stack2(x, 64, 3, name='conv2')
    x = resnet.stack2(x, 128, 8, name='conv3')
    x = resnet.stack2(x, 256, 36, name='conv4')
    x = resnet.stack2(x, 512, 3, stride1=1, name='conv5')
    return tf.keras.Model(layer_utils.get_source_inputs(input_tensor), x)


def make_karras_discriminator(input_tensor, start_hdim=4, dropout=0, lrelu=0.2):
    x = tf.keras.layers.Conv2D(min(2 ** start_hdim, 512), 1, name='conv0')(input_tensor)
    x = MeasureFeats(name='conv0_out')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LeakyReLU(lrelu, name=f'lrelu0')(x)

    for i, j in enumerate(range(start_hdim, start_hdim + 8)):
        h1, h2 = min(2 ** j, 512), min(2 ** (j + 1), 512)
        x = tf.keras.layers.Conv2D(h1, 3, padding='same', name=f'block{i + 1}_conv1')(x)
        x = MeasureFeats(name=f'block{i+1}_conv1_out')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'block{i + 1}_lrelu1')(x)

        x = tf.keras.layers.Conv2D(h2, 3, padding='same', name=f'block{i + 1}_conv2')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'block{i + 1}_lrelu2')(x)

        x = tf.keras.layers.AveragePooling2D()(x)

        if x.shape[1] <= 1:
            break

    return tf.keras.Model(layer_utils.get_source_inputs(input_tensor), x)


def attach_disc_head(input, nlayers, dropout, lrelu):
    x = input

    # Hidden layers
    for _ in range(nlayers):
        x = tf.keras.layers.Conv2D(256, 1)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU(lrelu)(x)

    # Last layer
    x = tf.keras.layers.Conv2D(1, 1)(x)

    return x


def make_discriminator(input_shape, backbone, layers, apply_spectral_norm=True, dropout=0, lrelu=0.2):
    input = tf.keras.Input(input_shape)
    x = StandardizeRGB()(input)

    backbone_fn_dict = {
        'KarrasDisc': partial(make_karras_discriminator, dropout=dropout, lrelu=lrelu),
        'BigKarrasDisc': partial(make_karras_discriminator, start_hdim=6, dropout=dropout, lrelu=lrelu),
        'ResNet152V2': make_resnet152v2,
        'VGG19': partial(tf.keras.applications.VGG19, weights=None)
    }

    backbone_fn = backbone_fn_dict[backbone]
    backbone = backbone_fn(input_tensor=x)

    # Get layer outputs
    nlayers = 0
    outputs = [attach_disc_head(backbone.get_layer(layer).output, nlayers, dropout, lrelu)
               for layer in layers]
    discriminator = tf.keras.Model(backbone.input, outputs, name='discriminator')

    # Apply spectral norm to linear layers
    if apply_spectral_norm:
        discriminator = tf.keras.Model().from_config(discriminator.get_config(),
                                                     custom_objects={'Conv2D': SNConv2D,
                                                                     'StandardizeRGB': StandardizeRGB,
                                                                     'BatchNormalization': NoBatchNorm,
                                                                     'MeasureFeats': MeasureFeats})

    return discriminator
