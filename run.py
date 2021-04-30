import shutil

import tensorflow as tf
from PIL import Image
from absl import logging, flags, app
import matplotlib.pyplot as plt

from style import make_and_compile_style_model
from utils import plot_history, DisplayGenImageCallback, load_style_and_gen_images

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'epochs')
flags.DEFINE_integer('steps_per_epoch', 1000, 'steps_per_epoch')


def train(sc_model, style_image):
    logging.info('started training')

    shutil.rmtree('out/logs', ignore_errors=True)
    callbacks = [DisplayGenImageCallback(),
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
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(tf.squeeze(tf.cast(gen_image, tf.uint8)))
    ax[1].imshow(tf.squeeze(tf.cast(style_image, tf.uint8)))
    plt.show()

    # Make the style model
    style_model = make_and_compile_style_model(gen_image)
    style_model.discriminator.summary()

    # Train the style model
    history = train(style_model, style_image)

    # Save the generated image
    out_path = 'out/gen.jpg'
    Image.fromarray(tf.squeeze(style_model.get_gen_image()).numpy()).save(out_path)
    logging.info(f"saved generated image to '{out_path}'")

    # Plots results
    if history is not None:
        plot_history(history)


if __name__ == '__main__':
    app.run(main)
