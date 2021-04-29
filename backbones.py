import tensorflow as tf
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.utils import layer_utils


def make_resnet152v2(input_tensor):
    resnet.layers = tf.keras.layers
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(input_tensor)
    x = resnet.stack2(x, 64, 3, name='conv2')
    x = resnet.stack2(x, 128, 8, name='conv3')
    x = resnet.stack2(x, 256, 36, name='conv4')
    x = resnet.stack2(x, 512, 3, stride1=1, name='conv5')
    return tf.keras.Model(layer_utils.get_source_inputs(input_tensor), x)


def make_karras(input_tensor, lrelu=0.2):
    x = tf.keras.layers.Conv2D(16, 1, name='conv0')(input_tensor)
    x = tf.keras.layers.LeakyReLU(lrelu, name=f'lrelu0')(x)

    for i, (h1, h2) in enumerate([(16, 32), (32, 64), (64, 128), (128, 256),
                                  (256, 512), (512, 512), (512, 512), (512, 512)]):
        x = tf.keras.layers.Conv2D(h1, 3, padding='same', name=f'block{i + 1}_conv1')(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'block{i + 1}_lrelu1')(x)

        x = tf.keras.layers.Conv2D(h2, 3, padding='same', name=f'block{i + 1}_conv2')(x)
        x = tf.keras.layers.LeakyReLU(lrelu, name=f'block{i + 1}_lrelu2')(x)

        x = tf.keras.layers.AveragePooling2D()(x)

        if x.shape[1] <= 1:
            break

    return tf.keras.Model(layer_utils.get_source_inputs(input_tensor), x)
