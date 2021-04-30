import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
from absl.testing import absltest
from tensorflow import python as tf_python

from discriminator import make_discriminator
from layers import NoBatchNorm

FLAGS = flags.FLAGS


class TestStyle(absltest.TestCase):
    def test_spectral_norm(self):
        input_shape = [32, 32, 3]
        backbone = 'Karras'
        layers = ['conv0']
        for spectral_norm in [True, False]:
            discriminator = make_discriminator(input_shape, backbone, layers, spectral_norm)
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    if spectral_norm:
                        self.assertIsInstance(layer, tfa.layers.SpectralNormalization)
                    else:
                        self.assertNotIsInstance(layer, tfa.layers.SpectralNormalization)

    def test_no_batch_norm(self):
        input_shape = [224, 224, 3]
        backbone = 'ResNet152V2'
        layers = ['conv2_block1_out']
        for spectral_norm in [False, True]:
            discriminator = make_discriminator(input_shape, backbone, layers, spectral_norm)
            discriminator.summary()

            num_batch_norms, num_no_batch_norms = 0, 0
            for layer in discriminator.layers:
                if isinstance(layers, tf.keras.layers.BatchNormalization) or \
                        isinstance(layer, tf_python.keras.layers.normalization_v2.BatchNormalization):
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
