import tensorflow as tf
import tensorflow_addons as tfa
from absl.testing import absltest

from discriminator import make_discriminator
from layers import NoBatchNorm, NoisyConv


class TestDiscriminator(absltest.TestCase):

    def test_spectral_norm(self):
        input_shape = [1, 8, 8, 3]
        disc_model = 'KarrasDisc'
        layers = ['conv0']
        for conv_mod in ['spectral_norm', None]:
            discriminator = make_discriminator(input_shape, disc_model, layers, conv_mod=conv_mod)
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    if conv_mod == 'spectral_norm':
                        self.assertIsInstance(layer, tfa.layers.SpectralNormalization)
                    else:
                        self.assertNotIsInstance(layer, tfa.layers.SpectralNormalization)

    def test_noisy_convs(self):
        input_shape = [1, 8, 8, 3]
        disc_model = 'KarrasDisc'
        layers = ['conv0']
        for conv_mod in ['noisy_conv_1', 'noisy_conv_0.5', 'noisy_conv_0.25', 'noisy_conv_0.1', 'noisy_conv_0.01']:
            discriminator = make_discriminator(input_shape, disc_model, layers, conv_mod=conv_mod)
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    self.assertIsInstance(layer, NoisyConv)
                    std = float(conv_mod.split('_')[-1])
                    self.assertEqual(std, layer.std)

    def test_dropout(self):
        input_shape = [1, 8, 8, 3]
        disc_model = 'KarrasDisc'
        layers = ['block1_lrelu1']
        for dropout in [0.5, 1]:
            discriminator = make_discriminator(input_shape, disc_model, layers, dropout=dropout)
            found_dropout = False
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    found_dropout = True
                    self.assertEqual(layer.rate, dropout)

            self.assertTrue(found_dropout)

    def test_no_dropout(self):
        input_shape = [1, 8, 8, 3]
        disc_model = 'KarrasDisc'
        layers = ['block1_lrelu1']
        discriminator = make_discriminator(input_shape, disc_model, layers, dropout=0)
        found_dropout = False
        for layer in discriminator.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                found_dropout = True

        self.assertFalse(found_dropout)

    def test_lrelu(self):
        input_shape = [1, 8, 8, 3]
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
        input_shape = [1, 32, 32, 3]
        disc_model = 'MobileNetV3Small'
        layers = ['re_lu']
        for conv_mod in [None, 'spectral_norm']:
            tf.keras.backend.clear_session()
            discriminator = make_discriminator(input_shape, disc_model, layers, conv_mod=conv_mod)

            num_batch_norms, num_no_batch_norms = 0, 0
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    num_batch_norms += 1
                if isinstance(layer, NoBatchNorm):
                    num_no_batch_norms += 1

            if conv_mod:
                self.assertEqual(num_batch_norms, 0)
                self.assertGreater(num_no_batch_norms, 0)
            else:
                self.assertGreater(num_batch_norms, 0)
                self.assertEqual(num_no_batch_norms, 0)


if __name__ == '__main__':
    absltest.main()
