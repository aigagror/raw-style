import tensorflow as tf
from absl import flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string('style_path', 'images/starry_night.jpg', 'file path to the style image')
flags.DEFINE_string('gen_path', None,
                    'file path to the generated image. If None, a uniform randomly generated image is used.')
flags.DEFINE_integer('image_size', 512, 'image size')


def load_resize_batch_image(image_path, image_size):
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.keras.preprocessing.image.smart_resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image


def load_style_image():
    style_image = load_resize_batch_image(FLAGS.style_path, FLAGS.image_size)
    logging.info(f"loaded style image from '{FLAGS.style_path}'")

    return style_image


def add_noise(image, magnitude):
    return tf.clip_by_value(image + magnitude * tf.random.uniform(tf.shape(image), minval=-1, maxval=1), 0, 255)
