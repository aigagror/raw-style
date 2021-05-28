import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging

from generator import PixelImageGenerator
from utils import add_noise


class StyleModel(tf.keras.Model):
    def __init__(self, discriminator, generator, noise=0, debug_g_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.noise = noise
        self.debug_g_grad = debug_g_grad
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
            style_logits, gen_logits = self.compute_logits(style_image)

            # Discriminator accuracy and loss
            d_accs = [self._disc_bce_acc(sl, gl) for sl, gl in zip(style_logits, gen_logits)]
            d_loss = [self._disc_bce_loss(sl, gl) for sl, gl in zip(style_logits, gen_logits)]

            d_loss = tf.reduce_sum(d_loss)

            # Generation loss
            g_loss = [self._gen_bce_loss(gl) for gl in gen_logits]
            g_loss = tf.reduce_sum(g_loss)

        # Metrics
        metrics = {'d_loss': d_loss, 'g_loss': g_loss}
        for i, (real_acc, gen_acc) in enumerate(d_accs, 1):
            metrics[f'd{i}_acc'] = 0.5 * real_acc + 0.5 * gen_acc
            metrics[f'd{i}_real'] = real_acc
            metrics[f'd{i}_gen'] = gen_acc

        # Optimize generator
        g_grad = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(g_grad, self.generator.trainable_weights))
        if isinstance(self.generator, PixelImageGenerator):
            metrics['avg_gm.pixel'] = tf.reduce_mean(tf.abs(g_grad))
            metrics['max_gm.pixel'] = tf.reduce_max(tf.abs(g_grad))

        if self.debug_g_grad:
            d_g_grad = tape.gradient(g_loss, self.discriminator.trainable_weights)
            for g in d_g_grad:
                metrics[f"avg_gm.{g.name.split('/')[-1]}"] = tf.reduce_mean(tf.abs(g))

        # Clip to RGB range
        self.generator.clip_rgb()

        # Optimize discriminator
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        for m in self.metrics:
            metrics[m.name] = m.result()

        return metrics

    def compute_logits(self, style_image):
        gen_image = self.generator(style_image, training=True)
        images = tf.concat([style_image, gen_image], axis=0)
        images = add_noise(images, self.noise)
        logits = self.discriminator(images, training=True)
        if not isinstance(logits, list):
            logits = [logits]
        style_logits, gen_logits = self._split_logits(logits, style_batch_size=tf.shape(style_image)[0])
        return style_logits, gen_logits

    def _split_logits(self, logits, style_batch_size):
        logits = [(l[:style_batch_size], l[style_batch_size:]) for l in logits]
        style_logits, gen_logits = zip(*logits)
        return list(style_logits), list(gen_logits)

    def _gen_bce_loss(self, gen_logits):
        g_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(gen_logits), gen_logits))
        return g_loss

    def _disc_bce_acc(self, style_logits, gen_logits):
        real_acc = tf.reduce_mean(
            tf.keras.metrics.binary_accuracy(tf.ones_like(style_logits), style_logits, threshold=0))
        gen_acc = tf.reduce_mean(
            tf.keras.metrics.binary_accuracy(tf.zeros_like(gen_logits), gen_logits, threshold=0))
        return real_acc, gen_acc

    def _disc_bce_loss(self, style_logits, gen_logits):
        real_loss = tf.reduce_mean(self.bce_loss(tf.ones_like(style_logits), style_logits))
        gen_loss = tf.reduce_mean(self.bce_loss(tf.zeros_like(gen_logits), gen_logits))
        d_loss = real_loss + gen_loss
        return d_loss


def make_and_compile_style_model(discriminator, generator, noise, debug_g_grad,
                                 disc_opt, disc_lr, disc_wd, gen_lr,
                                 gen_wd, gen_start, gen_decay,
                                 steps_exec=None):
    # Style model
    style_model = StyleModel(discriminator, generator, noise, debug_g_grad)

    # Discriminator optimizer
    disc_opt_map = {'sgd': tf.optimizers.SGD(disc_lr), 'lamb': tfa.optimizers.LAMB(disc_lr, weight_decay_rate=disc_wd)}
    disc_opt = disc_opt_map[disc_opt]

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
