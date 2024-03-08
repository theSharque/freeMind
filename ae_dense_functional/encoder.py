import tensorflow as tf
import dictionary


class Encoder:
    def __init__(self, word_size, pack_size, brain_size, trainable=True, plot=True):
        self.word_size = word_size
        self.pack_size = pack_size
        self.brain_size = brain_size
        self.trainable = trainable
        self.plot = plot

        self.vocab = dictionary.get_vocabulary()
        self.vocab_len = len(self.vocab) + 2

        self.input_head = tf.keras.layers.Input(shape=(1,), dtype='string')
        self.input_body = tf.keras.layers.Input(shape=(self.word_size,), dtype='int32')

        self.head = self.get_head(self.input_head)
        self.body = self.get_body(self.input_body)

        self.head_body = tf.keras.Model(inputs=self.input_head,
                                        outputs=self.body(self.head(self.input_head)),
                                        trainable=trainable)

        self.head_body.compile()

        self.text2ints = tf.keras.models.Model(self.input_head, self.head(self.input_head))
        self.text2pack = tf.keras.models.Model(self.input_head, self.body(self.head(self.input_head)))

    def get_head(self, inputs):
        # layer = tf.strings.split(tf.strings.lower(inputs, encoding='utf-8'))
        # layer = tf.keras.layers.Permute((2, 1))(layer)
        layer = tf.keras.layers.Reshape(target_shape=(1,))(inputs)
        layer = tf.strings.join([layer, '_'])
        layer = tf.keras.layers.TextVectorization(standardize=None,
                                                  split="character",
                                                  vocabulary=self.vocab,
                                                  output_sequence_length=self.word_size)(layer)

        head = tf.keras.Model(inputs=inputs, outputs=layer, trainable=False, name='head')
        head.compile()

        if self.plot:
            tf.keras.utils.plot_model(head,
                                      to_file='img/img_en-head.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return head

    def get_body(self, inputs):
        layer = inputs
        layer = tf.one_hot(layer, self.vocab_len)
        layer = [self.get_part(layer[:, i, :]) for i in range(layer.shape[-2])]
        layer = tf.keras.layers.Add()(layer)

        body = tf.keras.Model(inputs=inputs, outputs=layer, trainable=self.trainable, name='en_body')

        if self.plot:
            tf.keras.utils.plot_model(body,
                                      to_file='img/img_en-body.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        return body

    def get_part(self, inputs: tf.Tensor):
        layer = inputs

        layer = tf.keras.layers.Dense(self.brain_size)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = tf.keras.layers.Dense(self.brain_size)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = tf.keras.layers.Flatten()(layer)

        return tf.keras.layers.Dense(self.pack_size)(layer)
