import tensorflow as tf


class SmartDense(tf.keras.layers.Layer):
    def __init__(self, units, activation='tanh', trainable=True):
        super().__init__(trainable=trainable)
        self.units = units
        self.activation = activation
        self.dense = None
        self.glass = None
        self.shape = None

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        self.glass = tf.keras.layers.Conv1D(filters=self.units, kernel_size=input_shape[-1])
        self.shape = tf.keras.layers.Reshape((input_shape[-1], 1))

    def call(self, inputs):
        glass = self.shape(inputs)
        glass = self.glass(glass)
        glass = tf.keras.layers.Flatten()(glass)
        dense = self.dense(inputs)
        return tf.keras.layers.Multiply()([glass, dense])
