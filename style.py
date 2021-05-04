import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging

from generator import PixelImageGenerator


class StyleModel(tf.keras.Model):
    def __init__(self, discriminator, generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compile(self, disc_opt, gen_opt, *args, **kwargs):
        super().compile(gen_opt, *args, **kwargs)
        self.disc_opt = self._get_optimizer(disc_opt)
        logging.info(f'discriminator optimizer: {disc_opt.__class__.__name__}')
        logging.info(f'generator optimizer: {self.optimizer.__class__.__name__}')

    def call(self, style_image, training=None, mask=None):
        return self.generator(style_image, training=training)  # Image is just a filler argument to the generator

    def train_step(self, style_image):
        with tf.GradientTape(persistent=True) as tape:
            gen_image = self.generator(style_image, training=True)
            images = tf.concat([style_image, gen_image], axis=0)
            logits = self.discriminator(images, training=True)

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

        # Metrics
        metrics = {'d_acc': d_acc, 'd_loss': d_loss, 'g_loss': g_loss}

        # Optimize generator
        g_grad = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(g_grad, self.generator.trainable_weights))
        if isinstance(self.generator, PixelImageGenerator):
            metrics['avg_pixel_grad'] = tf.reduce_mean(tf.abs(g_grad))

        # Clip to RGB range
        self.generator.clip_rgb()

        # Optimize discriminator
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        for m in self.metrics:
            metrics[m.name] = m.result()

        return metrics

    def _gen_bce_loss(self, logits):
        gen_logits = logits[1]
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


def make_and_compile_style_model(discriminator, generator, disc_lr, disc_wd, gen_lr, gen_wd, gen_start, gen_decay,
                                 steps_exec=None):
    # Style model
    style_model = StyleModel(discriminator, generator)

    # Discriminator optimizer
    disc_opt = tfa.optimizers.LAMB(disc_lr, weight_decay_rate=disc_wd)

    # Generator optimizer
    gen_lr_schedule = make_gen_lr_schedule(gen_lr, gen_start, gen_decay)
    if isinstance(generator, PixelImageGenerator):
        gen_opt = tf.optimizers.Adam(gen_lr_schedule)
    else:
        gen_opt = tfa.optimizers.LAMB(gen_lr_schedule, weight_decay_rate=gen_wd)

    # Compile
    style_model.compile(disc_opt, gen_opt, steps_per_execution=steps_exec)

    return style_model


def make_gen_lr_schedule(gen_lr, gen_start, gen_decay=None):
    boundaries, values = [gen_start], [0.0, gen_lr]
    if gen_decay is not None:
        for i, decay_step in enumerate(gen_decay, 1):
            boundaries.append(decay_step)
            values.append(gen_lr * 10 ** -i)
    gen_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return gen_lr_schedule
