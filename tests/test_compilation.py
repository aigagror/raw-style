import tensorflow as tf
from absl.testing import absltest

from style import make_gen_lr_schedule


class TestCompilation(absltest.TestCase):
    def test_no_gen_start(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=0)
        tf.debugging.assert_equal(lr_schedule(1), tf.ones_like(lr_schedule(1)))
        tf.debugging.assert_equal(lr_schedule(2), tf.ones_like(lr_schedule(2)))

    def test_gen_start(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=10)
        for i in range(1, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.zeros_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(11), tf.ones_like(lr_schedule(11)))

    def test_gen_decay(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=0, gen_decay=[10])
        for i in range(1, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.ones_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(11), 0.1 * tf.ones_like(lr_schedule(11)))

    def test_gen_start_and_gen_decay(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=5, gen_decay=[10])
        for i in range(1, 6):
            tf.debugging.assert_equal(lr_schedule(i), tf.zeros_like(lr_schedule(i)))
        for i in range(6, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.ones_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(11), 0.1 * tf.ones_like(lr_schedule(11)))

    def test_gen_start_and_gen_decay_on_same_step(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=10, gen_decay=[10])
        for i in range(1, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.zeros_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(11), 0.1 * tf.ones_like(lr_schedule(11)))

    def test_gen_multi_decay(self):
        lr_schedule = make_gen_lr_schedule(gen_lr=1, gen_start=0, gen_decay=[10, 20])
        for i in range(1, 11):
            tf.debugging.assert_equal(lr_schedule(i), tf.ones_like(lr_schedule(i)))
        for i in range(11, 21):
            tf.debugging.assert_equal(lr_schedule(i), 0.1 * tf.ones_like(lr_schedule(i)))
        tf.debugging.assert_equal(lr_schedule(21), 0.01 * tf.ones_like(lr_schedule(21)))


if __name__ == '__main__':
    absltest.main()
