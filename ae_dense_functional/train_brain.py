import random

import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import dictionary
from enc_dec import EncDec


def plot_history(history, name):
    plt.figure(dpi=200)
    plt.plot(history['loss'])
    plt.plot(history['accuracy'])
    if 'val_loss' in history.keys() and 'val_accuracy' in history.keys():
        plt.plot(history['val_loss'])
        plt.plot(history['val_accuracy'])
    plt.title('Loss and accuracy')
    plt.ylabel('data')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['loss', 'accuracy'], loc='upper left')
    plt.savefig('./img/{}.png'.format(name))
    plt.close()


class StopIntsCallback(tf.keras.callbacks.Callback):

    def __init__(self, in_data, out_data, aed: EncDec):
        super().__init__()
        self.loss = 9999999
        self.bad_wait = 5
        self.good_wait = 5
        self.history = {}
        self.in_data = in_data
        self.out_data = out_data
        self.aed = aed

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        plot_history(self.history, 'brain')

        if logs.get('accuracy') > 0.95:
            self.good_wait -= 1
            if self.good_wait == 0:
                self.good_wait = 5
                self.model.stop_training = True
                print(" Model trained")
                # self.aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
                # self.aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
                # self.aed.br.body.save_weights('top_brain.hdf5', overwrite=True)
            else:
                print(" Good wait {}".format(self.good_wait))

        if logs.get('loss') < self.loss:
            self.bad_wait = 5
            self.loss = logs.get('loss')
            # self.aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
            # self.aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
            # self.aed.br.body.save_weights('top_brain.hdf5', overwrite=True)
        else:
            self.bad_wait -= 1
            if self.bad_wait == 0:
                self.bad_wait = 5
                self.model.stop_training = True
            else:
                print(" Bad wait {}".format(self.bad_wait))

        print()
        print(self.aed.de.shorts2text(self.out_data).numpy().decode('utf-8'))
        print(self.aed.de.ints2text(self.aed.ints_brain_ints(self.in_data)).numpy().decode('utf-8'))


def distillation_brain(aed):
    train_len = 10000
    while True:
        print('Train size {}'.format(train_len))
        txt = np.array(dictionary.get_text_line('../data/harry.txt').split()[:50000])
        in_data = aed.en.text2ints(txt)
        txt = np.array(txt[aed.CONTEXT_SIZE:])
        out_data = np.array(in_data[aed.CONTEXT_SIZE:])
        in_data = np.array([in_data[i:i + aed.CONTEXT_SIZE] for i in range(len(in_data) - aed.CONTEXT_SIZE)])

        rand_data = np.random.rand(out_data.shape[0], aed.br.RANDOM_SIZE)

        decoded = ' '.join([x.decode('utf-8') for x in
                            aed.de.ints2text.predict(aed.ints_brain_ints([in_data, rand_data]), batch_size=1000)]).split()

        if len(decoded) == len(out_data):
            learned = len(out_data[txt == decoded])
            in_data = in_data[txt != decoded][:train_len]
            out_data = out_data[txt != decoded][:train_len]
            rand_data = rand_data[:train_len]
        else:
            learned = 0
            in_data = in_data[:train_len]
            out_data = out_data[:train_len]
            rand_data = rand_data[:train_len]

        test_len = 30
        stop_training_callback = StopIntsCallback([in_data[:test_len], rand_data[:test_len]], out_data[:test_len], aed)
        print("Learned {}".format(learned))
        print(aed.de.shorts2text(out_data[:test_len]).numpy().decode('utf-8'))
        print(aed.de.ints2text(aed.ints_brain_ints([in_data[:test_len], rand_data[:test_len]])).numpy().decode('utf-8'))
        aed.ints_brain_ints.fit(x=[in_data, rand_data], y=out_data, validation_split=0.2, batch_size=64, epochs=10,
                                callbacks=[stop_training_callback], shuffle=True)

        aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
        aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
        aed.br.body.save_weights('top_brain.hdf5', overwrite=True)


def train_brain():
    encdec = EncDec(plot=True)

    if os.path.isfile('top_enc.hdf5'):
        encdec.en.body.load_weights('top_enc.hdf5')

    if os.path.isfile('top_dec.hdf5'):
        encdec.de.body.load_weights('top_dec.hdf5')

    if os.path.isfile('top_brain.hdf5'):
        encdec.br.body.load_weights('top_brain.hdf5')

    # train_brain_int(encdec)
    distillation_brain(encdec)


if __name__ == '__main__':
    train_brain()
