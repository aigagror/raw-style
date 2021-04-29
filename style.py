from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags

from backbones import make_karras, make_resnet152v2
from layers import SNConv2D, StandardizeRGB, NoBatchNorm

FLAGS = flags.FLAGS

flags.DEFINE_enum('backbone', 'Karras', ['Karras', 'VGG19', 'ResNet152V2'], 'backbone of the discriminator model')
flags.DEFINE_list('layers', [f'block{i}_lrelu1' for i in range(1, 4)],
                  'names of the layers to use as output for the style features')
flags.DEFINE_integer('steps_exec', None, 'steps per execution')

flags.DEFINE_float('disc_lr', 1e-2, 'discriminator learning rate')
flags.DEFINE_float('gen_lr', 1e-2, 'generated image learning rate')


class StyleModel(tf.keras.Model):
    def __init__(self, discriminator, gen_init, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.gen_init = gen_init
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def configure(self, style_image):
        # Call to build gen image
        self(style_image)

    def build(self, input_shape):
        if self.gen_init == 'rand':
            initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=255)
        else:
            assert self.gen_init == 'black'
            initializer = tf.keras.initializers.Zeros()
        print(f'initialzed gen image with {initializer.__class__.__name__}')
        self.gen_image = self.add_weight('gen_image', [1] + input_shape[1:], initializer=initializer)

    def compile(self, disc_opt, gen_opt, *args, **kwargs):
        super().compile(gen_opt, *args, **kwargs)
        self.disc_opt = self._get_optimizer(disc_opt)
        print(f'discriminator optimizer: {disc_opt.__class__.__name__}')
        print(f'generator optimizer: {self.optimizer.__class__.__name__}')

    def reinit_gen_image(self):
        self.gen_image.assign(tf.random.uniform(self.gen_image.shape, maxval=255, dtype=self.gen_image.dtype))

    def call(self, image, training=None, mask=None):
        return self.discriminator(image, training=training)

    def train_step(self, style_image):
        with tf.GradientTape(persistent=True) as tape:
            images = tf.concat([style_image, self.gen_image], axis=0)
            logits = self(images, training=True)

            # Discriminator
            if isinstance(logits, list):
                style_logits = [l[0] for l in logits]
                gen_logits = [l[1] for l in logits]

                d_loss, d_acc = 0, 0
                for rl, gl in zip(style_logits, gen_logits):
                    d_loss += tf.reduce_mean(self.bce_loss(tf.ones_like(rl), rl) + self.bce_loss(tf.zeros_like(gl), gl))
                    d_acc += tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(rl), rl, threshold=0))
                    d_acc += tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(gl), gl, threshold=0))
                d_acc /= (len(style_logits) * 2)
            else:
                style_logits, gen_logits = logits[0], logits[1]

                real_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(style_logits), style_logits))
                gen_loss = tf.reduce_mean(self.bce_loss(tf.zeros_like(gen_logits), gen_logits))
                d_loss = real_loss + gen_loss

                real_acc = tf.reduce_mean(
                    tf.keras.metrics.binary_accuracy(tf.ones_like(style_logits), style_logits, threshold=0))
                gen_acc = tf.reduce_mean(
                    tf.keras.metrics.binary_accuracy(tf.zeros_like(gen_logits), gen_logits, threshold=0))
                d_acc = 0.5 * real_acc + 0.5 * gen_acc

            # Generation loss
            if isinstance(gen_logits, list):
                g_loss = [tf.reduce_mean(self.bce_loss(tf.ones_like(logits), logits)) for logits in gen_logits]
                g_loss = tf.reduce_sum(g_loss)
            else:
                g_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(gen_logits), gen_logits))

        # Optimize generated image
        grad = tape.gradient(g_loss, [self.gen_image])
        self.optimizer.apply_gradients(zip(grad, [self.gen_image]))

        # Clip to RGB range
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

        # Optimize feature model
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        return {'d_loss': d_loss, 'd_acc': d_acc}

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_image, tf.uint8))


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


def make_style_model(style_image):
    input = tf.keras.Input(tf.shape(style_image)[1:])
    x = StandardizeRGB()(input)
    backbone_fn = backbone_fn_dict[FLAGS.backbone]
    backbone = backbone_fn(input_tensor=x)
    discriminator = make_discriminator(backbone, FLAGS.layers, apply_spectral_norm=True)
    discriminator.summary()
    style_model = StyleModel(discriminator, gen_init='rand')
    disc_opt = tfa.optimizers.LAMB(FLAGS.disc_lr)
    gen_opt = tf.optimizers.Adam(FLAGS.gen_lr)
    style_model.compile(disc_opt, gen_opt, steps_per_execution=FLAGS.steps_exec)
    style_model.configure(style_image)
    return style_model
