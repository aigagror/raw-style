import shutil

import tensorflow as tf
from PIL import Image
from absl import logging, flags, app

from discriminator import make_discriminator
from generator import make_generator
from metrics import assess_gen_style, StyleMetricCallback, ImageChangeCallback
from style import make_and_compile_style_model
from utils import load_style_image

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('epochs', 10, 'epochs')
flags.DEFINE_integer('steps_per_epoch', 1000, 'steps_per_epoch')
flags.DEFINE_integer('steps_exec', None, 'steps per execution')

flags.DEFINE_enum('disc_model', 'KarrasDisc', ['KarrasDisc', '256KarrasDisc', 'VGG19', 'ResNet152V2'],
                  'discriminator model')
flags.DEFINE_list('disc_layers', [f'block{i}_lrelu1' for i in range(1, 4)],
                  'names of the layers to use as output for the style features')
flags.DEFINE_integer('head_layers', 0, 'number of hidden layers for each discriminator head')
flags.DEFINE_enum('disc_opt', 'lamb', ['sgd', 'lamb'], 'discriminator optimizer')
flags.DEFINE_float('disc_lr', 1e-2, 'discriminator learning rate')
flags.DEFINE_float('disc_wd', 1e-2, 'discriminator weight decay')

flags.DEFINE_enum('gen_model', None, ['KarrasGen', 'SmallKarrasGen', 'TinyKarrasGen'], 'generator model')
flags.DEFINE_float('gen_lr', 1e-2, 'generated image learning rate')
flags.DEFINE_float('gen_wd', 0, 'generator weight decay. should only be used if using DeepImageGenerator')
flags.DEFINE_integer('gen_start', 0, 'delay the optimization of the generated image by [gen_start] epochs')
flags.DEFINE_multi_integer('gen_decay', [],
                           'decay the learning rate of the generation optimizer by 0.1 at the given step')
flags.DEFINE_bool('debug_g_grad', False,
                  'debug the gradient of the generated image by examining the average gradients of the generator loss w.r.t to the discriminator layers')

flags.DEFINE_enum('conv_mod', None, ['spectral_norm', 'noisy_conv_1', 'noisy_conv_0.5', 'noisy_conv_0.25', 'noisy_conv_0.1', 'noisy_conv_0.01'],
                  'modification to the convolution layers')
flags.DEFINE_float('dropout', 0, 'probability that a feature is zero-ed out. only the Karras models are affected')
flags.DEFINE_float('image_noise', 0, 'the amount of uniform noise to add to the images during train to prevent overfitting')
flags.DEFINE_float('lrelu', 0.2, 'Leaky ReLU parameter')


def train(sc_model, style_image):
    logging.info('started training')

    shutil.rmtree('out/logs', ignore_errors=True)
    callbacks = [ImageChangeCallback(),
                 tf.keras.callbacks.TensorBoard(log_dir='out/logs', histogram_freq=1, write_graph=False),
                 StyleMetricCallback(style_image)]
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
    logging.set_verbosity('info')

    # Other commands?
    if len(argv) > 1:
        command = argv[1].lower()
        if command == 'assess':
            assess_gen_style(FLAGS.gen_path, FLAGS.image_size)
        else:
            raise ValueError(f'unknown command: {command}')
        exit()

    # Load the images
    style_image = load_style_image()
    image_shape = tf.shape(style_image).numpy()

    # Make the style model
    discriminator = make_discriminator(image_shape, FLAGS.disc_model, FLAGS.disc_layers, FLAGS.head_layers,
                                       FLAGS.conv_mod, FLAGS.dropout, FLAGS.lrelu)
    logging.info(f'disc layers: {FLAGS.disc_layers}')
    discriminator.summary()

    # Multiple gen images?
    image_shape[0] = FLAGS.batch_size
    generator = make_generator(image_shape, FLAGS.gen_path, FLAGS.gen_model, FLAGS.dropout, FLAGS.lrelu)
    generator.summary()

    style_model = make_and_compile_style_model(discriminator, generator, FLAGS.image_noise, FLAGS.debug_g_grad,
                                               FLAGS.disc_opt, FLAGS.disc_lr, FLAGS.disc_wd,
                                               FLAGS.gen_lr, FLAGS.gen_wd, FLAGS.gen_start, FLAGS.gen_decay,
                                               FLAGS.steps_exec)

    # Train the style model
    train(style_model, style_image)

    # Save the generated image
    out_path = 'out/gen.jpg'
    Image.fromarray(style_model.generator.get_gen_image()[0].numpy()).save(out_path)
    logging.info(f"saved generated image to '{out_path}'")


if __name__ == '__main__':
    app.run(main)
