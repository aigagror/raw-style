import tensorflow as tf
from IPython import display
from PIL import Image
from absl import logging, flags, app

from style import make_style_model
from utils import plot_history, DisplayGenImageCallback, load_style_image

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'epochs')
flags.DEFINE_integer('steps_per_epoch', 1000, 'steps_per_epoch')

flags.DEFINE_bool('plot', True, 'plot the training history')


def train(sc_model, style_image):
    logging.info('started training')

    ds = tf.data.Dataset.from_tensor_slices([style_image])
    try:
        history = sc_model.fit(ds.cache().prefetch(tf.data.AUTOTUNE).repeat(),
                               epochs=FLAGS.epochs, steps_per_epoch=FLAGS.steps_per_epoch,
                               callbacks=DisplayGenImageCallback())
    except KeyboardInterrupt:
        history = None
    logging.info('finished training')
    return history


def main(argv):
    del argv  # Unused

    logging.set_verbosity('info')

    # Load the image
    style_image = load_style_image()
    display.display(Image.fromarray(tf.squeeze(tf.cast(style_image, tf.uint8)).numpy()))

    # Make the style model
    style_model = make_style_model(style_image)
    style_model.discriminator.summary()

    # Train the style model
    history = train(style_model, style_image)

    # Save the generated image
    out_path = 'out/gen.jpg'
    Image.fromarray(tf.squeeze(style_model.get_gen_image()).numpy()).save(out_path)
    logging.info(f"saved generated image to '{out_path}'")

    # Plots results
    if FLAGS.plot and history is not None:
        plot_history(history)


if __name__ == '__main__':
    app.run(main)
