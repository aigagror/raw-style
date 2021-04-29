import tensorflow as tf
from IPython import display
from PIL import Image
from absl import flags
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('image_path', 'images/starry_night.jpg', 'file path to the style image')
flags.DEFINE_integer('image_size', 512, 'image size')


def plot_history(history):
    for key, val in history.history.items():
        plt.figure()
        plt.plot(val, label=key)
        plt.legend()
    plt.show()


class DisplayGenImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_image(self):
        new_image = tf.squeeze(self.model.get_gen_image())
        self.prev_image = self.curr_image
        self.curr_image = new_image

        output.clear(output_tags='gen_image')
        with output.use_tags('gen_image'):
            a = tf.cast(self.prev_image, tf.float32)
            b = tf.cast(self.curr_image, tf.float32)
            delta = tf.reduce_mean(tf.math.abs(b - a), axis=-1)
            avg_change = tf.reduce_mean(delta)
            print(f"Mean pixel change: {float(avg_change):.3f}")
            delta_image = Image.fromarray(tf.cast(delta, tf.uint8).numpy())
            curr_image = Image.fromarray(self.curr_image.numpy())
            display.display(delta_image, curr_image)
        return avg_change

    def on_train_begin(self, logs=None):
        self.prev_image = tf.squeeze(self.model.get_gen_image())
        self.curr_image = tf.squeeze(self.model.get_gen_image())
        self.update_image()

    def on_epoch_end(self, epoch, logs=None):
        avg_change = self.update_image()
        if logs is not None:
            logs['delta'] = avg_change


def load_style_image():
    style_image = tf.image.decode_image(tf.io.read_file(FLAGS.image_path))
    style_image = tf.keras.preprocessing.image.smart_resize(style_image, [FLAGS.image_size, FLAGS.image_size])
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    style_image = tf.expand_dims(style_image, 0)
    return style_image
