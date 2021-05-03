import tensorflow as tf
from absl import flags
from absl.testing import absltest

from generator import make_generator, PixelImageGenerator, DeepImageGenerator
from utils import load_resize_batch_image

FLAGS = flags.FLAGS


class TestGenerator(absltest.TestCase):
    def test_random_pixel_generator(self):
        generator = make_generator([1, 8, 8, 3], gen_path=None, gen_model=None)
        self.assertIsInstance(generator, PixelImageGenerator)
        tf.debugging.assert_shapes([(generator.get_gen_image(), [1, 8, 8, 3])])

    def test_load_image_pixel_generator(self):
        generator = make_generator([1, 8, 8, 3], gen_path='../images/starry_night.jpg', gen_model=None)
        self.assertIsInstance(generator, PixelImageGenerator)
        tf.debugging.assert_shapes([(generator.get_gen_image(), [1, 8, 8, 3])])
        starry_night = load_resize_batch_image('../images/starry_night.jpg', 8)
        tf.debugging.assert_equal(tf.cast(starry_night, tf.uint8), generator.get_gen_image())

    def test_deep_image_generator(self):
        generator = make_generator([1, 8, 8, 3], gen_path=None, gen_model='KarrasGen')
        generator.summary()
        self.assertIsInstance(generator, DeepImageGenerator)
        tf.debugging.assert_shapes([(generator.get_gen_image(), [1, 8, 8, 3])])

        # Check that it satisfies RGB range
        gen_image = generator.get_gen_image()
        tf.debugging.assert_greater_equal(gen_image, tf.zeros_like(gen_image))
        tf.debugging.assert_less_equal(gen_image, 255 * tf.ones_like(gen_image))

    def test_bad_gen_args(self):
        self.assertRaises(AssertionError, make_generator, [1, 8, 8, 3], gen_path='../images/starry_night.jpg',
                          gen_model='KarrasGen')


if __name__ == '__main__':
    absltest.main()
