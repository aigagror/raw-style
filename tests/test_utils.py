import unittest

import tensorflow as tf

from generator import PixelImageGenerator
from metrics import get_moments, StyleMetricCallback
from style import StyleModel
from utils import add_noise


class TestUtils(unittest.TestCase):
    def test_no_add_noise(self):
        image = tf.random.uniform([8, 8, 3], maxval=255)
        noise_image = add_noise(image, magnitude=0)
        tf.debugging.assert_equal(image, noise_image)

    def test_add_little_noise(self):
        image = tf.random.uniform([8, 8, 3], maxval=255, seed=0)
        noise_image = add_noise(image, magnitude=1)
        tf.debugging.assert_near(image, noise_image, atol=1)

    def test_add_big_noise(self):
        image = tf.random.uniform([8, 8, 3], maxval=255, seed=0)
        noise_image = add_noise(image, magnitude=10)
        tf.debugging.assert_near(image, noise_image, atol=10)
        delta = tf.abs(noise_image - image)
        avg_change = tf.reduce_mean(delta)
        tf.debugging.assert_greater(avg_change, 4 * tf.ones_like(avg_change))

    def test_no_noise_black_image(self):
        image = tf.zeros([8, 8, 3])
        noise_image = add_noise(image, magnitude=0)
        tf.debugging.assert_equal(image, noise_image)

    def test_style_metric_standardize_style_feats(self):
        style_image = tf.random.uniform([1, 128, 128, 3], maxval=255, seed=0)  # Mean 0.5, Variance 1/12, Skewness 0
        gen_image = tf.random.uniform([1, 128, 128, 3], maxval=255, seed=1)  # Mean 0.5, Variance 1/12, Skewness 0

        sm_cbk = StyleMetricCallback(style_image, feat_model='debug')

        generator = PixelImageGenerator(gen_image)
        style_model = StyleModel(None, generator)
        sm_cbk.set_model(style_model)

        # 0 Mean
        for mean in sm_cbk.m1:
            tf.debugging.assert_shapes([(mean, [2])])
            tf.debugging.assert_near(mean, tf.zeros_like(mean), atol=1e-4)

        # Unit variance
        for covar in sm_cbk.m2:
            tf.debugging.assert_shapes([(covar, [2, 2])])
            style_var = tf.linalg.diag_part(covar)
            tf.debugging.assert_near(style_var, tf.ones_like(style_var), atol=1e-4)

        # 0 Skewness
        for skewness in sm_cbk.m3:
            tf.debugging.assert_shapes([(skewness, [2])])
            tf.debugging.assert_near(skewness, tf.zeros_like(skewness), atol=2e-2)

    def _assert_moments(self, feats, m1, m2, m3, atol):
        true_m1, true_m2, true_m3 = get_moments([feats], epsilon=1e-3)
        true_m1, true_m2, true_m3 = true_m1[0], true_m2[0], true_m3[0]
        tf.debugging.assert_near(true_m1, m1 * tf.ones_like(true_m1), atol=atol)
        tf.debugging.assert_near(true_m2, m2 * tf.ones_like(true_m2), atol=atol)
        tf.debugging.assert_near(true_m3, m3 * tf.ones_like(true_m3), atol=atol)

    def test_style_metric_near_perfect_qualities(self):
        style_image = tf.random.normal([1, 128, 128, 1], seed=0)
        gen_image = tf.random.normal([1, 128, 128, 1], seed=1)
        self._assert_moments(style_image, 0, 1, 0, atol=5e-2)
        self._assert_moments(gen_image, 0, 1, 0, atol=5e-2)

        sm_cbk = StyleMetricCallback(style_image, feat_model='debug')

        generator = PixelImageGenerator(gen_image)
        style_model = StyleModel(None, generator)
        sm_cbk.set_model(style_model)

        # Near perfect matching
        q1, q2, q3 = sm_cbk.get_style_metrics()
        tf.debugging.assert_near(q1, tf.zeros_like(q1), atol=1e-2)
        tf.debugging.assert_near(q2, tf.zeros_like(q2), atol=1e-1)
        tf.debugging.assert_near(q3, tf.zeros_like(q3), atol=4e-2)

    def test_style_metric_bad_mean_var_skew(self):
        style_image = tf.random.normal([1, 128, 128, 1], seed=0)
        gen_image = tf.random.gamma([1, 128, 128, 1], alpha=2, seed=1)  # Mean 2, variance 2, skewness ~1.4

        self._assert_moments(style_image, 0, 1, 0, atol=5e-2)
        self._assert_moments(gen_image, 2, 2, 1.4, atol=5e-2)

        sm_cbk = StyleMetricCallback(style_image, feat_model='debug')

        generator = PixelImageGenerator(gen_image)
        style_model = StyleModel(None, generator)
        sm_cbk.set_model(style_model)

        # Metrics
        q1, q2, q3 = sm_cbk.get_style_metrics()

        # Bad mean
        tf.debugging.assert_greater(q1, 2 * tf.ones_like(q1))

        # Bad variance
        tf.debugging.assert_greater(q2, 1.5 * tf.ones_like(q2))

        # Bad skew
        tf.debugging.assert_greater(q3, tf.ones_like(q3))


if __name__ == '__main__':
    unittest.main()
