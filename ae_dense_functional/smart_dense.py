import tensorflow as tf


class SmartDense(tf.keras.layers.Layer):
    def __init__(self, units, trainable=True):
        super().__init__(trainable=trainable)
        self.units = units
        self.flatten = None
        self.glass = None
        self.lens = None
        self.shape = None
        self.dense = None
        self.multiply = None

    def build(self, input_shape):
        self.shape = tf.keras.layers.Reshape((input_shape[-1], 1))
        self.glass = tf.keras.layers.Conv1D(filters=self.units, kernel_size=input_shape[-1], activation='sigmoid')
        self.lens = tf.keras.layers.Conv1D(filters=self.units, kernel_size=input_shape[-1], activation='sigmoid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=self.units, activation='tanh')

    def call(self, inputs):
        glass = self.shape(inputs)
        glass = self.glass(glass)
        glass = self.flatten(glass)

        lens = self.shape(inputs)
        lens = self.lens(lens)
        lens = self.flatten(lens)

        dense = self.dense(inputs)
        return glass * dense + lens
