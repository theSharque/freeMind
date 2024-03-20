import os.path

import matplotlib.pyplot as plt

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


class StopTrainingCallback(tf.keras.callbacks.Callback):

    def __init__(self, x_vals, y_vals, aed: EncDec):
        super().__init__()
        self.loss = 9999999
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.aed = aed
        self.bad_wait = 10
        self.good_wait = 10
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        plot_history(self.history, 'epoch')

        if logs.get('accuracy') > 0.95:
            self.good_wait -= 1
            if self.good_wait == 0:
                self.good_wait = 10
                self.model.stop_training = True
                print(" Model trained")
                self.aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
                self.aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
            else:
                print(" Good wait {}".format(self.good_wait))

        if logs.get('loss') < self.loss:
            self.bad_wait = 10
            self.loss = logs.get('loss')
            self.aed.en.body.save_weights('top_enc.hdf5', overwrite=True)
            self.aed.de.body.save_weights('top_dec.hdf5', overwrite=True)
        else:
            self.bad_wait -= 1
            if self.bad_wait == 0:
                self.bad_wait = 10
                self.model.stop_training = True
            else:
                print(" Bad wait {}".format(self.bad_wait))

        origin = ' '.join(self.aed.de.shorts2text(self.y_vals).numpy().decode('utf-8').split())
        calculated = ' '.join(self.aed.ints2text(self.x_vals).numpy().decode('utf-8').split())
        if calculated != origin:
            print('\n', origin, sep='')
            print(calculated)


def train_both_distillation(aed: EncDec, skip=False):
    while True:
        words = np.array(
            sorted(set(dictionary.get_words(file_name='../data/harry.txt', max_len=24, min_len=1).split())))
        # np.random.shuffle(words)
        # words = words[:200000]

        new_vals = aed.text2text(words).numpy().decode('utf-8').split()
        if len(words) == len(new_vals):
            new_vals = words[words != new_vals]
        else:
            new_vals = words

        batch_size = min(max(len(new_vals) // 1000, 1), 32)

        print("Problem words: {}".format(len(new_vals)))
        if len(new_vals) <= 1:
            print("Train finished")
            break

        x_vals = aed.en.text2ints(new_vals)
        stop_training_callback = StopTrainingCallback(np.concatenate([x_vals[:10], x_vals[-10:]]),
                                                      np.concatenate([x_vals[:10], x_vals[-10:]]),
                                                      aed)

        aed.ints2ints.fit(x=x_vals, y=x_vals, validation_split=0.01, batch_size=batch_size, epochs=20,
                          callbacks=[stop_training_callback], shuffle=True)
        aed.de.body.load_weights('top_dec.hdf5')
        aed.en.body.load_weights('top_enc.hdf5')

        decoded = aed.ints2text(x_vals).numpy().decode('utf-8').split()
        new_vals = new_vals[new_vals != decoded]

        print("Distilled result {}".format(len(new_vals)))
        if skip:
            print("Train finished")
            break


def warmup(aed: EncDec):
    for i in range(8, 0, -1):
        words = np.array(sorted(set(dictionary.get_vocabulary(False))))
        np.random.shuffle(words)
        x_vals = aed.en.text2ints(words)
        stop_training_callback = StopTrainingCallback(np.concatenate([x_vals[:10], x_vals[-10:]]),
                                                      np.concatenate([x_vals[:10], x_vals[-10:]]),
                                                      aed)

        aed.ints2ints.fit(x=x_vals, y=x_vals, batch_size=i, epochs=1000, callbacks=[stop_training_callback])


def train_shift_distillation(aed: EncDec, skip=False):
    warmup(aed)

    while True:
        words = np.array(dictionary.get_words(file_name='../data/small_h.txt', max_len=24, min_len=1).split())
        new_vals = aed.text2text(words).numpy().decode('utf-8').split()
        if len(words) == len(new_vals):
            new_vals = words[:-1][words[:-1] != new_vals[1:]]
        else:
            new_vals = words

        batch_size = min(max(len(new_vals) // 1000, 1), 32)

        print("Problem words: {}".format(len(new_vals)))
        if len(new_vals) <= 1:
            print("Train finished")
            break

        x_vals = aed.en.text2ints(new_vals[:-1])
        y_vals = aed.en.text2ints(new_vals[1:])
        stop_training_callback = StopTrainingCallback(x_vals[:15], y_vals[:15], aed)

        aed.ints2ints.fit(x=x_vals, y=y_vals, validation_split=0.01, batch_size=batch_size, epochs=20,
                          callbacks=[stop_training_callback], shuffle=True)
        aed.de.body.load_weights('top_dec.hdf5')
        aed.en.body.load_weights('top_enc.hdf5')

        decoded = aed.ints2text(x_vals).numpy().decode('utf-8').split()
        new_vals = new_vals[:-1][new_vals[:-1] != decoded]

        print("Distilled result {}".format(len(new_vals)))
        if skip:
            print("Train finished")
            break


def train_decoder():
    aed = EncDec(tr_en=False, plot=True)

    if os.path.isfile('top_enc.hdf5'):
        aed.en.body.load_weights('top_enc.hdf5')
    if os.path.isfile('top_dec.hdf5'):
        aed.de.body.load_weights('top_dec.hdf5')
    if os.path.isfile('top_brain.hdf5'):
        aed.br.body.load_weights('top_brain.hdf5')

    train_shift_distillation(aed, skip=True)


if __name__ == '__main__':
    train_decoder()
