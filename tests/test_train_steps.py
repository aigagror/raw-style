from absl.testing import absltest
from style import StyleModel
import tensorflow as tf

class TestTrainSteps(absltest.TestCase):
    def make_style_model(self, gen_image):
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, 1, kernel_initializer='ones', bias_initializer='zeros')
        ])
        style_model = StyleModel(discriminator, gen_image)
        style_model.compile('sgd', 'sgd')
        return style_model

    def test_perfect_acc(self):
        gen_image, style_image = -tf.ones([1, 8, 8, 3]), tf.ones([1, 8, 8, 3])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, tf.ones_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(3.0)) - tf.math.log(1 - tf.sigmoid(-3.0)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(-3.0)))

    def test_mid_acc(self):
        gen_image, style_image = tf.zeros([1, 8, 8, 3]), tf.zeros([1, 8, 8, 3])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, 0.5 * tf.ones_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(0.0)) - tf.math.log(1 - tf.sigmoid(-0.0)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(0.0)))

    def test_worst_acc(self):
        gen_image, style_image = tf.ones([1, 8, 8, 3]), -tf.ones([1, 8, 8, 3])
        style_model = self.make_style_model(gen_image)

        metrics = style_model.train_step(style_image)
        d_acc, d_loss, g_loss = metrics['d_acc'], metrics['d_loss'], metrics['g_loss']
        tf.debugging.assert_equal(d_acc, tf.zeros_like(d_acc))
        tf.debugging.assert_near(d_loss, -tf.math.log(tf.sigmoid(-3.0)) - tf.math.log(1 - tf.sigmoid(3.0)))
        tf.debugging.assert_near(g_loss, -tf.math.log(tf.sigmoid(3.0)))


if __name__ == '__main__':
    absltest.main()