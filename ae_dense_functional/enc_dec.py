import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from context_brain import ContextBrain


class EncDec:
    def __init__(self, tr_en=True, tr_de=True, tr_br=True, plot=True):
        self.WORD_SIZE = 24
        self.ENC_DEC_SIZE = 128
        self.PACK_SIZE = 1024
        self.CONTEXT_SIZE = 8
        self.plot = plot

        self.en: Encoder = Encoder(self.WORD_SIZE, self.PACK_SIZE, self.ENC_DEC_SIZE, tr_en, plot)
        self.de: Decoder = Decoder(self.WORD_SIZE, self.PACK_SIZE, self.ENC_DEC_SIZE, tr_de, plot)
        self.br = ContextBrain(self.CONTEXT_SIZE, self.PACK_SIZE, tr_br, plot)

        self.text2text = tf.keras.models.Model(self.en.input_head,
                                               self.de.body_tail(self.en.head_body(self.en.input_head)))

        self.noise = self.get_noise(self.de.input_body, name='noise')

        self.context_input = tf.keras.layers.Input(shape=(self.CONTEXT_SIZE, self.WORD_SIZE,), dtype='int32')

        self.input_big_body = tf.keras.layers.Input(shape=(self.PACK_SIZE * self.CONTEXT_SIZE,), dtype='float32')
        self.big_noise = self.get_noise(self.input_big_body, name='big_noise')

        self.ints_brain_ints = tf.keras.models.Model(self.context_input, self.de.body(
            self.noise(self.br.body(self.big_noise(self.get_brain_encoder(self.context_input))))))

        self.ints_brain_ints.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
                                     metrics=[accuracy])

        if plot:
            tf.keras.utils.plot_model(self.ints_brain_ints,
                                      to_file='img/img_enc-brain-dec-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

    def get_brain_encoder(self, inputs):
        layer = [self.en.body(inputs[:, i, :]) for i in range(self.CONTEXT_SIZE)]
        layer = tf.keras.layers.Concatenate()(layer)
        return layer

    def get_noise(self, inputs, name):
        layer = inputs

        layer = tf.keras.layers.GaussianNoise(0.01)(layer)
        layer = tf.keras.layers.Dropout(0.01)(layer)

        model = tf.keras.Model(inputs=inputs, outputs=layer, trainable=True, name=name)

        if self.plot:
            tf.keras.utils.plot_model(model,
                                      to_file='img/img_noise.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return model


def accuracy(y_true, y_pred):
    return tf.boolean_mask(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred), tf.math.not_equal(y_true, 0))
