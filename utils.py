import tensorflow as tf
from absl import flags, logging
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('style_path', 'images/starry_night.jpg', 'file path to the style image')
flags.DEFINE_string('gen_path', None,
                    'file path to the generated image. If None, a uniform randomly generated image is used.')
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
        try:
            from google.colab import output
            self.output = output
        except:
            logging.info('Colab not available. Generated image will not display during training.')

    def update_image(self):
        new_image = tf.squeeze(self.model.get_gen_image())
        self.prev_image = self.curr_image
        self.curr_image = new_image

        a = tf.cast(self.prev_image, tf.float32)
        b = tf.cast(self.curr_image, tf.float32)
        delta = tf.reduce_mean(tf.math.abs(b - a), axis=-1)
        avg_change = tf.reduce_mean(delta)

        if hasattr(self, 'output'):
            self.output.clear(output_tags='gen_image')
            with self.output.use_tags('gen_image'):
                logging.info(f"Mean pixel change: {float(avg_change):.3f}")
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(tf.cast(delta, tf.uint8))
                ax[1].imshow(self.curr_image)
                plt.show()

        return avg_change

    def on_train_begin(self, logs=None):
        self.prev_image = tf.squeeze(self.model.get_gen_image())
        self.curr_image = tf.squeeze(self.model.get_gen_image())
        self.update_image()

    def on_epoch_end(self, epoch, logs=None):
        avg_change = self.update_image()
        if logs is not None:
            logs['delta'] = avg_change


def load_and_resize_image(image_path, image_size):
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.keras.preprocessing.image.smart_resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image


def load_style_and_gen_images():
    style_image = load_and_resize_image(FLAGS.style_path, FLAGS.image_size)
    logging.info(f"loaded style image from '{FLAGS.style_path}'")

    if FLAGS.gen_path is not None:
        gen_image = load_and_resize_image(FLAGS.gen_path, FLAGS.image_size)
        logging.info(f"loaded generated image from '{FLAGS.gen_path}'")
    else:
        gen_image = tf.random.uniform([1, FLAGS.image_size, FLAGS.image_size, 3], maxval=256, dtype=tf.int32)
        gen_image = tf.cast(gen_image, tf.float32)
        logging.info(f"generated image initialized as random uniform")

    return style_image, gen_image
