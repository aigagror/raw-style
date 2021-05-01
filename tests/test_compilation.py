import tensorflow as tf
from absl.testing import absltest

from style import make_gen_lr_schedule


class TestCompilation(absltest.TestCase):
    def test_no_gen_delay(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_delay=0)
        tf.debugging.assert_equal(lr_schedule(1), tf.ones_like(lr_schedule(1)))
        tf.debugging.assert_equal(lr_schedule(2), tf.ones_like(lr_schedule(2)))

    def test_gen_delay(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_delay=10)
        for i in range(1, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.zeros_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(11), tf.ones_like(lr_schedule(11)))


if __name__ == '__main__':
    absltest.main()
