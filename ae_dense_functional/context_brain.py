import tensorflow as tf

from ae_dense_functional.smart_dense import SmartDense


class ContextBrain:
    def __init__(self, context_size, pack_size, dec_size, trainable=True, plot=True):
        self.context_size = context_size
        self.pack_size = pack_size
        self.dec_size = dec_size
        self.trainable = trainable
        self.plot = plot

        self.input_body = tf.keras.layers.Input(shape=(self.context_size, self.pack_size,), dtype='float32')
        self.body = self.get_body(self.input_body)
        self.body.compile()

    def get_body(self, inputs):
        layer = tf.keras.layers.Dense(self.pack_size)(inputs)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = tf.keras.layers.Flatten()(layer)

        layer = SmartDense(self.pack_size)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = SmartDense(self.pack_size * self.context_size)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = SmartDense(self.pack_size)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        layer = SmartDense(self.dec_size)(layer)
        body = tf.keras.models.Model(inputs=self.input_body, outputs=layer, trainable=self.trainable, name='brain')

        if self.plot:
            tf.keras.utils.plot_model(body,
                                      to_file='img/img_br-body.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

        return body
