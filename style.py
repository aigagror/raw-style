import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags, logging

from discriminator import make_discriminator

FLAGS = flags.FLAGS
flags.DEFINE_enum('backbone', 'Karras', ['Karras', 'VGG19', 'ResNet152V2'], 'backbone of the discriminator model')
flags.DEFINE_list('layers', [f'block{i}_lrelu1' for i in range(1, 4)],
                  'names of the layers to use as output for the style features')
flags.DEFINE_float('dropout', 0, 'probability that a feature is zero-ed out. only the Karras backbone is affected')
flags.DEFINE_float('lrelu', 0, 'Leaky ReLU parameter')

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
            gen_logits = self(self.gen_image, training=False)
            if isinstance(gen_logits, list):
                g_loss = [self._gen_bce_loss(l) for l in gen_logits]
                g_loss = tf.reduce_sum(g_loss)
            else:
                g_loss = self._gen_bce_loss(gen_logits)

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

    def _gen_bce_loss(self, gen_logits):
        g_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(gen_logits), gen_logits))
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


def make_and_compile_style_model(gen_image):
    input_shape = tf.shape(gen_image)[1:]

    discriminator = make_discriminator(input_shape, FLAGS.backbone, FLAGS.layers,
                                       FLAGS.spectral_norm, FLAGS.dropout, FLAGS.lrelu)

    style_model = StyleModel(discriminator, gen_image)

    disc_opt = tfa.optimizers.LAMB(FLAGS.disc_lr)
    gen_opt = tf.optimizers.Adam(FLAGS.gen_lr)
    style_model.compile(disc_opt, gen_opt, steps_per_execution=FLAGS.steps_exec)

    return style_model
