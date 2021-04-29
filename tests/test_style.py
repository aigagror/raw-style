import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
from absl.testing import absltest

from backbones import make_karras
from style import make_discriminator

FLAGS = flags.FLAGS


class TestStyle(absltest.TestCase):
    def test_spectral_norm(self):
        backbone = make_karras(tf.keras.Input([32, 32, 3]))
        layers = ['conv0']
        for spectral_norm in [True, False]:
            discriminator = make_discriminator(backbone, layers, apply_spectral_norm=spectral_norm)
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    if spectral_norm:
                        self.assertIsInstance(layer, tfa.layers.SpectralNormalization)
                    else:
                        self.assertNotIsInstance(layer, tfa.layers.SpectralNormalization)


if __name__ == '__main__':
    absltest.main()
