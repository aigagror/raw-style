import tensorflow as tf
from absl import flags
from absl.testing import absltest

from discriminator import make_karras, make_resnet152v2

FLAGS = flags.FLAGS


class TestDiscriminator(absltest.TestCase):
    def test_make_model_fns(self):
        for make_model_fn in [make_karras, make_resnet152v2]:
            input = tf.keras.Input([32, 32, 3])
            model = make_model_fn(input)
            self.assertIsInstance(model, tf.keras.Model)


if __name__ == '__main__':
    absltest.main()
