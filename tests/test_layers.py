import tensorflow as tf
from absl import flags
from absl.testing import absltest

from layers import StandardizeRGB, MeasureFeats

FLAGS = flags.FLAGS


class TestLayer(absltest.TestCase):
    def test_standardize_rgb(self):
        image = tf.random.uniform([1, 32, 32, 3], seed=0, maxval=256, dtype=tf.int32)
        model = tf.keras.Sequential([StandardizeRGB()])
        out = model(image)
        mean, std = tf.nn.moments(out, axes=[0, 1, 2])
        tf.debugging.assert_near(mean, tf.zeros_like(mean), atol=0.1)
        tf.debugging.assert_greater(std, tf.ones_like(std))
        tf.debugging.assert_less(std, 2 * tf.ones_like(std))

    def test_measure_feats(self):
        image = tf.random.normal([1, 32, 32, 3], seed=0)
        model = tf.keras.Sequential([MeasureFeats()])
        model(image)
        norm = model.metrics[0].result()
        mean = model.metrics[1].result()
        std = model.metrics[2].result()
        tf.debugging.assert_near(norm, (3 ** 0.5) * tf.ones_like(norm), rtol=0.5)  # There is precision loss for this
        tf.debugging.assert_near(mean, tf.zeros_like(mean), atol=0.01)
        tf.debugging.assert_near(std, tf.ones_like(std), atol=0.05)


if __name__ == '__main__':
    absltest.main()
