import shutil

import tensorflow as tf
from PIL import Image
from absl import logging, flags, app

from discriminator import make_discriminator
from style import make_and_compile_style_model
from utils import ImageChangeCallback, load_style_and_gen_images

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'epochs')
flags.DEFINE_integer('steps_per_epoch', 1000, 'steps_per_epoch')
flags.DEFINE_integer('steps_exec', None, 'steps per execution')

flags.DEFINE_enum('disc_model', 'KarrasDisc', ['KarrasDisc', 'BigKarrasDisc', 'VGG19', 'ResNet152V2'],
                  'discriminator model')
flags.DEFINE_list('layers', [f'block{i}_lrelu1' for i in range(1, 4)],
                  'names of the layers to use as output for the style features')

flags.DEFINE_bool('spectral_norm', True, 'apply spectral normalization to all linear layers in the model')
flags.DEFINE_float('dropout', 0, 'probability that a feature is zero-ed out. only the KarrasDisc disc_model is affected')
flags.DEFINE_float('lrelu', 0, 'Leaky ReLU parameter')

flags.DEFINE_float('disc_lr', 1e-2, 'discriminator learning rate')
flags.DEFINE_float('disc_wd', 1e-2, 'discriminator weight decay')

flags.DEFINE_integer('gen_delay', 0, 'delay the optimization of the generated image by [gen_delay] epochs')
flags.DEFINE_float('gen_lr', 1e-2, 'generated image learning rate')


def train(sc_model, style_image):
    logging.info('started training')

    shutil.rmtree('out/logs', ignore_errors=True)
    callbacks = [ImageChangeCallback(),
                 tf.keras.callbacks.TensorBoard(log_dir='out/logs', histogram_freq=1, write_graph=False)]
    ds = tf.data.Dataset.from_tensor_slices([style_image])

    try:
        history = sc_model.fit(ds.cache().prefetch(tf.data.AUTOTUNE).repeat(),
                               epochs=FLAGS.epochs, steps_per_epoch=FLAGS.steps_per_epoch,
                               callbacks=callbacks)
    except KeyboardInterrupt:
        history = None
        logging.info('caught keyboard interrupt. ended training early.')

    logging.info('finished training')
    return history


def main(argv):
    del argv  # Unused
    logging.set_verbosity('info')

    # Load the images
    style_image, gen_image = load_style_and_gen_images()

    # Make the style model
    input_shape = tf.shape(gen_image)[1:]
    discriminator = make_discriminator(input_shape, FLAGS.disc_model, FLAGS.layers, FLAGS.spectral_norm, FLAGS.dropout,
                                       FLAGS.lrelu)
    discriminator.summary()
    style_model = make_and_compile_style_model(gen_image, discriminator, FLAGS.disc_lr, FLAGS.disc_wd,
                                               FLAGS.gen_lr, FLAGS.gen_delay, FLAGS.steps_exec)

    # Train the style model
    train(style_model, style_image)

    # Save the generated image
    out_path = 'out/gen.jpg'
    Image.fromarray(tf.squeeze(style_model.get_gen_image()).numpy()).save(out_path)
    logging.info(f"saved generated image to '{out_path}'")


if __name__ == '__main__':
    app.run(main)
