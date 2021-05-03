import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
from absl.testing import absltest

from discriminator import make_karras_discriminator, make_resnet152v2, make_discriminator
from layers import NoBatchNorm

FLAGS = flags.FLAGS


class TestDiscriminator(absltest.TestCase):
    def test_make_model_fns(self):
        for make_model_fn in [make_karras_discriminator, make_resnet152v2]:
            input = tf.keras.Input([32, 32, 3])
            model = make_model_fn(input)
            self.assertIsInstance(model, tf.keras.Model)

    def test_spectral_norm(self):
        input_shape = [1, 32, 32, 3]
        disc_model = 'KarrasDisc'
        layers = ['conv0']
        for spectral_norm in [True, False]:
            discriminator = make_discriminator(input_shape, disc_model, layers, spectral_norm)
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    if spectral_norm:
                        self.assertIsInstance(layer, tfa.layers.SpectralNormalization)
                    else:
                        self.assertNotIsInstance(layer, tfa.layers.SpectralNormalization)

    def test_dropout(self):
        input_shape = [1, 32, 32, 3]
        disc_model = 'KarrasDisc'
        layers = ['block1_lrelu1']
        for dropout in [0, 0.5, 1]:
            discriminator = make_discriminator(input_shape, disc_model, layers, dropout=dropout)
            found_dropout = False
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    found_dropout = True
                    self.assertEqual(layer.rate, dropout)

            self.assertTrue(found_dropout)

    def test_lrelu(self):
        input_shape = [1, 32, 32, 3]
        disc_model = 'KarrasDisc'
        layers = ['block1_lrelu1']
        for lrelu in [0, 0.5, 1]:
            discriminator = make_discriminator(input_shape, disc_model, layers, lrelu=lrelu)
            found_lrelu = False
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.LeakyReLU):
                    found_lrelu = True
                    self.assertEqual(layer.alpha, lrelu)

            self.assertTrue(found_lrelu)

    def test_no_batch_norm(self):
        input_shape = [1, 224, 224, 3]
        disc_model = 'MobileNetV3Small'
        layers = ['re_lu']
        for spectral_norm in [False, True]:
            tf.keras.backend.clear_session()
            discriminator = make_discriminator(input_shape, disc_model, layers, spectral_norm)

            num_batch_norms, num_no_batch_norms = 0, 0
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    num_batch_norms += 1
                if isinstance(layer, NoBatchNorm):
                    num_no_batch_norms += 1

            if spectral_norm:
                self.assertEqual(num_batch_norms, 0)
                self.assertGreater(num_no_batch_norms, 0)
            else:
                self.assertGreater(num_batch_norms, 0)
                self.assertEqual(num_no_batch_norms, 0)


if __name__ == '__main__':
    absltest.main()
