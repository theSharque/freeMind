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
        self.bad_wait = 10
        self.good_wait = 10
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

        if logs.get('loss') < self.loss:
            self.bad_wait = 10
            self.loss = logs.get('loss')
            self.aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
            self.aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
            self.aed.br.body.save_weights('top_brain.hdf5', overwrite=True)
        else:
            self.bad_wait -= 1
            if self.bad_wait == 0:
                self.bad_wait = 10
                self.model.stop_training = True
            else:
                print(" Bad wait {}".format(self.bad_wait))

        print()
        print(self.aed.de.shorts2text(self.out_data).numpy().decode('utf-8'))
        print(self.aed.de.ints2text(self.aed.ints_brain_ints(self.in_data)).numpy().decode('utf-8'))


def warmup(aed: EncDec):
    for i in range(8, 0, -1):
        words = np.array(sorted(set(dictionary.get_vocabulary(False))))
        np.random.shuffle(words)
        y_vals = aed.en.text2ints(words)
        x_vals = np.reshape(y_vals[1:], y_vals[1:].shape + 1).transpose([0, 2, 1])
        y_vals = y_vals[:-1]
        stop_training_callback = StopIntsCallback(np.concatenate([x_vals[:10], x_vals[-10:]]),
                                                  np.concatenate([y_vals[:10], y_vals[-10:]]),
                                                  aed)

        aed.ints_brain_ints.fit(x=x_vals, y=y_vals, batch_size=i, epochs=1000, callbacks=[stop_training_callback])


def train_brain_int(aed: EncDec):
    while True:
        txt = dictionary.get_text_line('../data/harry.txt').split()
        part_len = random.randint(0, len(txt) - 20000)
        txt = txt[part_len:part_len + 20000]
        in_data = aed.en.text2ints(np.array(txt))
        del txt
        out_data = np.array(in_data[aed.CONTEXT_SIZE:])
        in_data = np.array([in_data[i:i + aed.CONTEXT_SIZE] for i in range(len(in_data) - aed.CONTEXT_SIZE - 1)])

        test_len = 30
        stop_training_callback = StopIntsCallback(in_data[:test_len], out_data[:test_len], aed)

        print(aed.de.shorts2text(out_data[:test_len]).numpy().decode('utf-8'))
        print(aed.de.ints2text(aed.ints_brain_ints(in_data[:test_len])).numpy().decode('utf-8'))

        aed.ints_brain_ints.fit(x=in_data, y=out_data, validation_split=0.01, batch_size=32, epochs=20,
                                callbacks=[stop_training_callback], shuffle=True)


def train_brain():
    encdec = EncDec(plot=True)

    if os.path.isfile('top_enc.hdf5'):
        encdec.en.body.load_weights('top_enc.hdf5')

    if os.path.isfile('top_dec.hdf5'):
        encdec.de.body.load_weights('top_dec.hdf5')

    if os.path.isfile('top_brain.hdf5'):
        encdec.br.body.load_weights('top_brain.hdf5')

    train_brain_int(encdec)


if __name__ == '__main__':
    train_brain()
