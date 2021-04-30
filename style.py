from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags, logging

from backbones import make_karras, make_resnet152v2
from layers import SNConv2D, StandardizeRGB, NoBatchNorm

FLAGS = flags.FLAGS

flags.DEFINE_enum('backbone', 'Karras', ['Karras', 'VGG19', 'ResNet152V2'], 'backbone of the discriminator model')
flags.DEFINE_list('layers', [f'block{i}_lrelu1' for i in range(1, 4)],
                  'names of the layers to use as output for the style features')
flags.DEFINE_integer('steps_exec', None, 'steps per execution')

flags.DEFINE_float('disc_lr', 1e-2, 'discriminator learning rate')
flags.DEFINE_float('gen_lr', 1e-2, 'generated image learning rate')

flags.DEFINE_bool('spectral_norm', True, 'apply spectral normalization to all linear layers in the model')


class StyleModel(tf.keras.Model):
    def __init__(self, discriminator, gen_image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.gen_image = tf.Variable(gen_image, trainable=True, name='gen_image', dtype=self.dtype)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compile(self, disc_opt, gen_opt, *args, **kwargs):
        super().compile(gen_opt, *args, **kwargs)
        self.disc_opt = self._get_optimizer(disc_opt)
        logging.info(f'discriminator optimizer: {disc_opt.__class__.__name__}')
        logging.info(f'generator optimizer: {self.optimizer.__class__.__name__}')

    def call(self, image, training=None, mask=None):
        return self.discriminator(image, training=training)

    def train_step(self, style_image):
        with tf.GradientTape(persistent=True) as tape:
            images = tf.concat([style_image, self.gen_image], axis=0)
            logits = self(images, training=True)

            # Discriminator
            if isinstance(logits, list):
                d_acc = [self._disc_bce_acc(l) for l in logits]
                d_loss = [self._disc_bce_loss(l) for l in logits]

                d_acc = tf.reduce_mean(d_acc)
                d_loss = tf.reduce_sum(d_loss)
            else:
                d_loss = self._disc_bce_loss(logits)
                d_acc = self._disc_bce_acc(logits)

            # Generation loss
            if isinstance(logits, list):
                g_loss = [self._gen_bce_loss(l) for l in logits]
                g_loss = tf.reduce_sum(g_loss)
            else:
                g_loss = self._gen_bce_loss(logits)

        # Optimize generated image
        grad = tape.gradient(g_loss, [self.gen_image])
        self.optimizer.apply_gradients(zip(grad, [self.gen_image]))

        # Clip to RGB range
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

        # Optimize feature model
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        return {'d_acc': d_acc, 'd_loss': d_loss, 'g_loss': g_loss}

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_image, tf.uint8))

    def _gen_bce_loss(self, logits):
        g_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(logits[1]), logits[1]))
        return g_loss

    def _disc_bce_acc(self, logits):
        style_logits, gen_logits = logits[0], logits[1]
        real_acc = tf.reduce_mean(
            tf.keras.metrics.binary_accuracy(tf.ones_like(style_logits), style_logits, threshold=0))
        gen_acc = tf.reduce_mean(
            tf.keras.metrics.binary_accuracy(tf.zeros_like(gen_logits), gen_logits, threshold=0))
        d_acc = 0.5 * real_acc + 0.5 * gen_acc
        return d_acc

    def _disc_bce_loss(self, logits):
        style_logits, gen_logits = logits[0], logits[1]
        real_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(style_logits), style_logits))
        gen_loss = tf.reduce_mean(self.bce_loss(tf.zeros_like(gen_logits), gen_logits))
        d_loss = real_loss + gen_loss
        return d_loss


def attach_disc_head(input, nlayers, lrelu=0.2):
    x = input
    hdim = input.shape[-1]

    # Hidden layers
    for _ in range(nlayers):
        x = tf.keras.layers.Conv2D(256, 1)(x)
        x = tf.keras.layers.LeakyReLU(lrelu)(x)

    # Last layer
    x = tf.keras.layers.Conv2D(1, 1)(x)

    return x


def make_discriminator(backbone, layers, apply_spectral_norm):
    # Get layer outputs
    outputs = [attach_disc_head(backbone.get_layer(layer).output, nlayers=0) for layer in layers]
    discriminator = tf.keras.Model(backbone.input, outputs)

    # Apply spectral norm to linear layers
    if apply_spectral_norm:
        discriminator = tf.keras.Model().from_config(discriminator.get_config(),
                                                     custom_objects={'Conv2D': SNConv2D,
                                                                     'StandardizeRGB': StandardizeRGB,
                                                                     'BatchNormalization': NoBatchNorm})

    return discriminator


backbone_fn_dict = {
    'Karras': make_karras,
    'ResNet152V2': make_resnet152v2,
    'VGG19': partial(tf.keras.applications.VGG19, weights=None)
}


def make_and_compile_style_model(gen_image):
    input = tf.keras.Input(tf.shape(gen_image)[1:])
    x = StandardizeRGB()(input)

    backbone_fn = backbone_fn_dict[FLAGS.backbone]
    backbone = backbone_fn(input_tensor=x)

    discriminator = make_discriminator(backbone, FLAGS.layers, apply_spectral_norm=FLAGS.spectral_norm)

    style_model = StyleModel(discriminator, gen_image)

    disc_opt = tfa.optimizers.LAMB(FLAGS.disc_lr)
    gen_opt = tf.optimizers.Adam(FLAGS.gen_lr)
    style_model.compile(disc_opt, gen_opt, steps_per_execution=FLAGS.steps_exec)

    return style_model
