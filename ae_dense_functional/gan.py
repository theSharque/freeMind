import tensorflow as tf

from smart_dense_l2 import SmartDenseL2
from enc_dec import EncDec


class GAN:
    def __init__(self, plot=True):
        self.encdec = EncDec()

        self.DESC_SIZE = 1024

        self.generator_body = self.encdec.br.body(self.encdec.big_noise(self.encdec.get_int_brain_encoder(
            self.encdec.context_input)))

        self.generator_model = tf.keras.Model(
            inputs=self.encdec.context_input,
            outputs=self.generator_body,
            trainable=True,
            name='generator')
        self.generator_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-9),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0))

        if plot:
            tf.keras.utils.plot_model(self.generator_model,
                                      to_file='img/img_generator-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

        self.desc_input = tf.keras.layers.Input(self.generator_body.shape[1:])
        self.encdec.en.body.trainable = False
        self.body = self.discriminator(self.desc_input, self.encdec.context_input)

        self.discriminator_model = tf.keras.Model(inputs=[self.desc_input, self.encdec.context_input],
                                                  outputs=self.body,
                                                  trainable=True,
                                                  name='discriminator')
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-9),
                                         metrics=['accuracy'])

        if plot:
            tf.keras.utils.plot_model(self.discriminator_model,
                                      to_file='img/img_discriminator-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

        self.encdec.en.body.trainable = True
        self.discriminator_model.trainable = False
        self.combined = tf.keras.Model(inputs=self.encdec.context_input,
                                       outputs=self.discriminator_model([
                                           self.generator_model(self.encdec.context_input),
                                           self.encdec.context_input
                                       ]),
                                       name='combined')
        self.combined.compile(loss=tf.losses.binary_focal_crossentropy,
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-9),
                              metrics=['accuracy'])

        if plot:
            tf.keras.utils.plot_model(self.combined,
                                      to_file='img/combined-model.png',
                                      show_layer_activations=True,
                                      expand_nested=True,
                                      show_shapes=True,
                                      show_dtype=True,
                                      show_trainable=True,
                                      show_layer_names=True)

    def discriminator(self, inputs, context_input):

        layer = tf.keras.layers.Flatten()(inputs)

        context = tf.keras.layers.Flatten()(context_input)
        context = tf.cast(context, tf.float32)
        layer = tf.keras.layers.Concatenate()([layer, context])

        layer = SmartDenseL2(self.DESC_SIZE)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = SmartDenseL2(self.DESC_SIZE // 2)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = SmartDenseL2(self.DESC_SIZE // 4)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = SmartDenseL2(self.DESC_SIZE // 8)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = SmartDenseL2(self.DESC_SIZE // 16)(layer)
        layer = tf.keras.layers.Activation('tanh')(layer)

        layer = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
        return layer
