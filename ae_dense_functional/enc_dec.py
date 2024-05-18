import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from context_brain import ContextBrain


class EncDec:
    def __init__(self, tr_en=True, tr_de=True, tr_br=True, plot=True):
        self.WORD_SIZE = 24
        self.ENC_SIZE = 256
        self.PACK_SIZE = 512
        self.DEC_SIZE = 256
        self.CONTEXT_SIZE = 3
        self.plot = plot

        self.en: Encoder = Encoder(self.WORD_SIZE, self.ENC_SIZE, tr_en, plot)
        self.de: Decoder = Decoder(self.WORD_SIZE, self.DEC_SIZE, tr_de, plot)
        self.br = ContextBrain(self.CONTEXT_SIZE, self.PACK_SIZE, self.ENC_SIZE, self.DEC_SIZE, tr_br, plot)

        self.noise = self.get_noise(self.de.input_body, name='noise')

        self.context_word_input = tf.keras.layers.Input(shape=(self.CONTEXT_SIZE,), dtype='string')
        self.text_brain_text = tf.keras.models.Model(inputs=[self.context_word_input, self.br.random_input],
                                                     outputs=self.de.body_tail(self.br.body([
                                                         self.get_string_brain_encoder(self.context_word_input),
                                                         self.br.random_input])),
                                                     name='text_brain_text')

        if plot:
            tf.keras.utils.plot_model(self.text_brain_text,
                                      to_file='img/img_text_brain_text-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

        self.context_input = tf.keras.layers.Input(shape=(self.CONTEXT_SIZE, self.WORD_SIZE,), dtype='int32')

        self.input_big_body = tf.keras.layers.Input(shape=(self.CONTEXT_SIZE, self.ENC_SIZE,), dtype='float32')
        self.big_noise = self.get_noise(self.input_big_body, name='big_noise')

        self.ints_brain_ints = tf.keras.models.Model(inputs=[self.context_input, self.br.random_input],
                                                     outputs=self.de.body(self.noise(self.br.body([
                                                         self.big_noise(
                                                             self.get_int_brain_encoder(self.context_input)),
                                                         self.br.random_input
                                                     ]))),
                                                     name='ints_brain_ints')

        self.ints_brain_ints.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-9),
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

        self.noise_to_ints = tf.keras.models.Model(self.de.input_body, self.de.body(self.noise(self.de.input_body)),
                                                   name='noise_ints')
        self.noise_to_ints.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-9),
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
                                   metrics=[accuracy])
        if plot:
            tf.keras.utils.plot_model(self.noise_to_ints,
                                      to_file='img/img_noise-dec-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

        self.enc_out = tf.keras.layers.Input(shape=(1, self.PACK_SIZE,), dtype='float32')

    def get_string_brain_encoder(self, inputs: tf.Tensor):
        layer = [self.en.head_body(inputs[:, i]) for i in range(self.CONTEXT_SIZE)]
        layer = tf.keras.layers.Concatenate(-2)(layer)
        return layer

    def get_int_brain_encoder(self, inputs: tf.Tensor):
        layer = [self.en.body(inputs[:, i, :]) for i in range(self.CONTEXT_SIZE)]
        layer = tf.keras.layers.Concatenate(-2)(layer)
        return layer

    def get_noise(self, inputs, name):
        layer = tf.keras.layers.GaussianNoise(0.03)(inputs)
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
