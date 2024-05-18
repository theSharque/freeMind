import numpy as np
import dictionary
import os

from gan import GAN


def train(gan: GAN):
    # Load the dataset
    train_len = 10000
    test_len = 15
    print('Loading...')
    all_txt = np.array(dictionary.get_text_line('../data/harry.txt').split())

    # part = np.random.randint(len(all_txt) - train_len + gan.encdec.CONTEXT_SIZE)
    part = 0
    txt = all_txt[part:part + train_len + gan.encdec.CONTEXT_SIZE]
    in_data = gan.encdec.en.text2ints(txt[:train_len + gan.encdec.CONTEXT_SIZE])
    real_out = in_data[gan.encdec.CONTEXT_SIZE:]
    in_data = [in_data[i:i + gan.encdec.CONTEXT_SIZE] for i in range(len(in_data) - gan.encdec.CONTEXT_SIZE)]

    real = gan.encdec.en.body(real_out).numpy().transpose(1, 0, 2)[0]
    # real = tf.one_hot(real_out, gan.encdec.en.vocab_len)
    print('Train size {}'.format(train_len))
    for epoch in range(100000):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        print("Discriminator")

        d_loss = []
        enough = False
        while not enough:
            rand_data = np.random.rand(len(in_data), gan.encdec.br.RANDOM_SIZE)
            fake = gan.generator_model.predict_on_batch([np.array(in_data), rand_data])
            x = [np.concatenate([real, fake]), np.concatenate([np.array(in_data), np.array(in_data)])]
            y = np.concatenate([np.ones((train_len, 1)), np.zeros((train_len, 1))])

            history = gan.discriminator_model.fit(x, y, batch_size=64, epochs=1)
            d_loss = [history.history['loss'][-1], history.history['accuracy'][-1]]
            enough = history.history['accuracy'][-1] >= 0.95

        gan.discriminator_model.save_weights('top_desc.hdf5', overwrite=True)

        # ---------------------
        #  Train Generator
        # ---------------------
        print("Generator")
        g_loss = []
        enough = False
        while not enough:
            rand_data = np.random.rand(len(in_data), gan.encdec.br.RANDOM_SIZE)
            x = [np.array(in_data), rand_data]
            y = np.ones((train_len, 1))

            history = gan.combined.fit(x, y, batch_size=64, epochs=1)
            g_loss = [history.history['loss'][-1], history.history['accuracy'][-1]]
            enough = history.history['accuracy'][-1] >= 0.95
            print(gan.encdec.de.shorts2text(real_out[:test_len]).numpy().decode('utf-8'))
            print(gan.encdec.de.pack2text(gan.generator_model(np.array(in_data[:test_len]))).numpy().decode('utf-8'))

        gan.encdec.en.body.save_weights('top_enc.hdf5', overwrite=True)
        gan.encdec.de.body.save_weights('top_dec.hdf5', overwrite=True)
        gan.encdec.br.body.save_weights('top_brain.hdf5', overwrite=True)

        # ---------------------
        #  Test result
        # ---------------------
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss[0], 100 * d_loss[1]))
        print(gan.encdec.de.shorts2text(real_out[:test_len]).numpy().decode('utf-8'))
        print(gan.encdec.de.pack2text(gan.generator_model(np.array(in_data[:test_len]))).numpy().decode('utf-8'))


if __name__ == '__main__':
    g_gan = GAN()

    if os.path.isfile('top_enc.hdf5'):
        g_gan.encdec.en.body.load_weights('top_enc.hdf5')

    if os.path.isfile('top_dec.hdf5'):
        g_gan.encdec.de.body.load_weights('top_dec.hdf5')

    if os.path.isfile('top_brain.hdf5'):
        g_gan.encdec.br.body.load_weights('top_brain.hdf5')

    if os.path.isfile('top_desc.hdf5'):
        g_gan.discriminator_model.load_weights('top_desc.hdf5')

    train(g_gan)
