import numpy as np
import tensorflow as tf

from ae_dense_functional.enc_dec import EncDec


def test_model():
    encdec = EncDec(plot=True)
    encdec.en.body.load_weights('top_enc.hdf5')
    encdec.de.body.load_weights('top_dec.hdf5')
    encdec.br.body.load_weights('top_brain.hdf5')

    context = 'Мистер и миссис Дёрсли , обитатели дома 4'.lower().split()
    print(' '.join(context), end='')
    while True:
        out = encdec.text_brain_text(np.array(context).reshape([1, 8])).numpy().decode('utf-8')
        context = context[1:] + [out]
        if out in (',', '.', ';', ':'):
            print(out, end='')
        else:
            print(' {}'.format(out), end='')


if __name__ == '__main__':
    test_model()
    # while True:
    # train_brain()
    # train_decoder()
