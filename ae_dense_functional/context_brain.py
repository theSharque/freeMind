import tensorflow as tf

from smart_dense_l2 import SmartDenseL2


class ContextBrain:
    def __init__(self, context_size, brain_size, enc_size, dec_size, trainable=True, plot=True):
        self.RANDOM_SIZE = 5

        self.context_size = context_size
        self.brain_size = brain_size
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.trainable = trainable
        self.plot = plot

        self.input_body = tf.keras.layers.Input(shape=(self.context_size, self.enc_size,), dtype='float32')
        self.random_input = tf.keras.layers.Input(shape=(self.RANDOM_SIZE,), dtype='float32')
        self.body = tf.keras.models.Model(inputs=[self.input_body, self.random_input],
                                          outputs=self.get_body(self.input_body, self.random_input),
                                          trainable=self.trainable, name='brain')
        if self.plot:
            tf.keras.utils.plot_model(self.body,
                                      to_file='img/img_br-body.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)
        self.body.compile()

    def get_body(self, inputs, random_input):
        layer = tf.keras.layers.Flatten()(inputs)

        layer = SmartDenseL2(self.brain_size)(layer)
        layer = tf.keras.layers.Activation('sigmoid')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = SmartDenseL2(self.brain_size)(layer)
        layer = tf.keras.layers.Activation('sigmoid')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = tf.concat([layer, random_input], axis=-1)

        layer = tf.keras.layers.Dense(self.dec_size)(layer)
        return layer
