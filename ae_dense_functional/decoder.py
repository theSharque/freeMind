import tensorflow as tf
import dictionary


class Decoder:
    def __init__(self, word_size, pack_size, brain_size, trainable, plot=True):
        self.word_size = word_size
        self.pack_size = pack_size
        self.brain_size = brain_size
        self.trainable = trainable
        self.plot = plot

        self.vocab = dictionary.get_vocabulary()
        self.vocab_len = len(self.vocab) + 2

        self.input_body = tf.keras.layers.Input(shape=(self.pack_size,), dtype='float32')
        self.input_tail = tf.keras.layers.Input(shape=(self.word_size, self.vocab_len,), dtype='float32')
        self.input_tail_short = tf.keras.layers.Input(shape=(self.word_size,), dtype='float32')

        self.body = self.get_body(self.input_body)
        self.tail_short = self.get_tail_short(self.input_tail_short)
        self.tail = self.get_tail(self.input_tail)

        self.body_tail = tf.keras.Model(inputs=self.input_body,
                                        outputs=self.tail(self.body(self.input_body)),
                                        trainable=trainable, name="body_tail")
        self.body_tail.compile()

        self.pack2text = tf.keras.models.Model(self.input_body, self.tail(self.body(self.input_body)), name="pack2text")
        self.ints2text = tf.keras.models.Model(self.input_tail, self.tail(self.input_tail), name="ints2text")
        self.shorts2text = tf.keras.models.Model(self.input_tail_short, self.tail_short(self.input_tail_short),
                                                 name="shorts2text")

    def get_body(self, inputs):
        layer = inputs

        layer = tf.keras.layers.Dense(self.brain_size)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = tf.keras.layers.Dense(self.vocab_len * self.word_size)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = tf.keras.layers.Reshape((self.word_size, self.vocab_len))(layer)
        layer = tf.keras.layers.Softmax(-1)(layer)

        body = tf.keras.Model(inputs=inputs, outputs=layer, trainable=self.trainable, name='de_body')

        if self.plot:
            tf.keras.utils.plot_model(body,
                                      to_file='img/img_de-body.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return body

    def get_tail(self, inputs):
        layer = tf.argmax(inputs, axis=-1)
        layer = self.tail_short(layer)

        tail = tf.keras.Model(inputs=inputs, outputs=layer, trainable=False, name='tail')
        tail.compile()

        if self.plot:
            tf.keras.utils.plot_model(tail,
                                      to_file='img/img_de-tail.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return tail

    def get_tail_short(self, inputs):
        layer = tf.keras.layers.StringLookup(vocabulary=self.vocab, mask_token='', invert=True, encoding='UTF-8')(
            inputs)
        layer = tf.strings.reduce_join(layer, axis=-1, separator='')
        layer = tf.strings.split(layer, sep='_', maxsplit=1)
        layer = tf.transpose(layer.to_tensor())[0]
        layer = tf.strings.reduce_join(layer, axis=-1, separator=' ')

        tail = tf.keras.Model(inputs=inputs, outputs=layer, trainable=False, name='tail_short')
        tail.compile()

        if self.plot:
            tf.keras.utils.plot_model(tail,
                                      to_file='img/img_de-tail_short.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return tail
