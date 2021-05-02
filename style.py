import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging


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
        g_grad = tape.gradient(g_loss, [self.gen_image])
        self.optimizer.apply_gradients(zip(g_grad, [self.gen_image]))
        avg_pixel_change = tf.reduce_mean(tf.abs(g_grad))

        # Clip to RGB range
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

        # Optimize feature model
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        return {'d_acc': d_acc, 'd_loss': d_loss, 'g_loss': g_loss, 'avg_pixel_change': avg_pixel_change}

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


def make_and_compile_style_model(gen_image, discriminator, disc_lr, disc_wd, gen_lr, gen_delay, steps_exec=None):
    # Style model
    style_model = StyleModel(discriminator, gen_image)

    # Discriminator optimizer
    disc_opt = tfa.optimizers.LAMB(disc_lr, weight_decay_rate=disc_wd)

    # Generator optimizer
    gen_lr_schedule = make_gen_lr_schedule(gen_lr, gen_delay)
    gen_opt = tf.optimizers.Adam(gen_lr_schedule)

    # Compile
    style_model.compile(disc_opt, gen_opt, steps_per_execution=steps_exec)

    return style_model


def make_gen_lr_schedule(gen_lr, gen_delay):
    boundaries, values = [gen_delay], [0.0, gen_lr]
    gen_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return gen_lr_schedule
