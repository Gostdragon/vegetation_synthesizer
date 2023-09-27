import os
from datetime import datetime

import tensorflow as tf
import time
import config
from IPython import display
import glob
from pathlib import Path


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(in_channel):
    inputs = tf.keras.layers.Input(shape=[256, 256, in_channel])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target, loss_object):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (config.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator(in_channel):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, in_channel], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, config.OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


@tf.function
def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer,
               loss_object, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,
                                                                   loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)


    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    if step % 1000 == 0:
        tf.print('disc:',  disc_loss)
        tf.print('l1:',  gen_l1_loss )
        tf.print('gen total:', gen_total_loss)


def fit(a, generate_images, in_channels, train_ds, test_ds, output_pic_path, output_model_path, steps):
    test_ds_it = iter(test_ds)

    generator = Generator(in_channels)
    discriminator = Discriminator(in_channels)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = output_model_path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    start = time.time()
    i = 35000
    s = time.time()
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        step += 35000
        if step % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()
            example_input, example_target = next(test_ds_it)
            generate_images(a, generator, example_input, example_target, i, output_pic_path)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer,
                   loss_object, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if step % 5000 == 0:
           checkpoint.save(file_prefix=checkpoint_prefix)

        i += 1
        step += 1

    print(f'{time.time() - s:.2f} sec\n')
    checkpoint.save(file_prefix=checkpoint_prefix)


def generate_images_model(args, generate_images, in_channels, test_ds, output_pic_path, output_model_path, count):


    generator = Generator(in_channels)
    discriminator = Discriminator(in_channels)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    if 0:
        # Generate multiple Images from different models
        #models = (list(Path('synthesizer/vegetation/milestone/global_scale/').glob('**/*checkpoint'))
        #          + list(Path('synthesizer/vegetation/milestone/local_scale/').glob('**/*checkpoint')))
        models = ([x[0] for x in os.walk('synthesizer/vegetation/milestone/global_scale/')] +
                  [x[0] for x in os.walk('synthesizer/vegetation/milestone/local_scale/')])
        models = list(filter(lambda x: str(x).find('model') > 0 and str(x).find('1-1') > 0, models))
        models = ['synthesizer/vegetation/milestone/model_usa_vegetation/']

    models = [output_model_path]

    i = 0
    for model in models:
        checkpoint_dir = str(model)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.expect_partial()
        if status.assert_existing_objects_matched():
            print("Checkpoint restored successfully.")
        else:
            print("Checkpoint restoration failed.")

        loaded_gen = checkpoint.generator
        for (input_image, target) in test_ds.take(count):
            if input_image.shape[0] != config.HEIGHT:
                generate_images(args, loaded_gen, input_image[0:1, ...], target[0:1, ...], i, output_pic_path)
            else:
                generate_images(args, loaded_gen, input_image, target, i, output_pic_path)
            print(f'{i} {checkpoint_dir}')
            i += 1
    return
