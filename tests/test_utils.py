import unittest

import tensorflow as tf

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


if __name__ == '__main__':
    unittest.main()
