import tensorflow as tf
from absl.testing import absltest

from generator import PixelImageGenerator, make_karras_generator, DeepImageGenerator
from style import StyleModel


class TestTrainSteps(absltest.TestCase):
    def make_style_model(self, gen_image, gen_type='pixel'):
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, 1, kernel_initializer='ones', bias_initializer=tf.initializers.Constant(-0.5))
        ])
        if gen_type == 'pixel':
            generator = PixelImageGenerator(gen_image)
        else:
            assert gen_type == 'deep'
            seed, gen_model = make_karras_generator(tf.shape(gen_image), start_hdim=8, max_dim=8)
            generator = DeepImageGenerator(seed, gen_model)
        style_model = StyleModel(discriminator, generator)
        style_model.compile('sgd', 'sgd')
        return style_model

    def test_style_and_gen_logits(self):
        for gen_type in ['pixel', 'deep']:
            style_image, gen_image = tf.ones([3, 8, 8, 1]), tf.zeros([4, 8, 8, 1])
            style_model = self.make_style_model(gen_image, gen_type)
            style_logits, gen_logits = style_model.compute_logits(style_image)
            self.assertIsInstance(style_logits, list)
            self.assertIsInstance(gen_logits, list)

            self.assertEqual(len(style_logits), 1)
            self.assertEqual(len(gen_logits), 1)

            tf.debugging.assert_equal(style_logits[0], 0.5 * tf.ones_like(style_logits[0]))

            if gen_type == 'pixel':
                tf.debugging.assert_equal(gen_logits[0], -0.5 * tf.ones_like(gen_logits[0]))

            tf.debugging.assert_shapes([
                (style_logits[0], [3, 8, 8, 1]),
                (gen_logits[0], [4, 8, 8, 1]),
            ])

    def test_perfect_acc(self):
        gen_image, style_image = tf.zeros([1, 8, 8, 1]), tf.ones([1, 8, 8, 1])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d1_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, tf.ones_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(0.5)) - tf.math.log(1 - tf.sigmoid(-0.5)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(-0.5)))

    def test_mid_acc(self):
        gen_image, style_image = 0.5 * tf.ones([1, 8, 8, 1]), 0.5 * tf.ones([1, 8, 8, 1])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d1_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, 0.5 * tf.ones_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(0.0)) - tf.math.log(1 - tf.sigmoid(-0.0)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(0.0)))

    def test_worst_acc(self):
        gen_image, style_image = tf.ones([1, 8, 8, 1]), tf.zeros([1, 8, 8, 1])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d1_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, tf.zeros_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(-0.5)) - tf.math.log(1 - tf.sigmoid(0.5)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(0.5)))


if __name__ == '__main__':
    absltest.main()
