
import tensorflow as tf
from keras import backend as K, Input, callbacks
from keras.engine import Model
from keras.layers import Dense, Activation, Concatenate, Lambda, Embedding, Conv2D, BatchNormalization, Dropout, Reshape
from keras.engine.topology import Layer
from keras.initializers import RandomUniform


def normal_likelihood_3D(y, yhat_sigma):
    sigma = tf.nn.softplus(yhat_sigma[:, :, 1]) + 1e-3
    yhat = yhat_sigma[:, :, 0]
    y = y[:, :, 0]
    return K.mean(K.mean(tf.log(sigma + K.epsilon()) + 0.5 * tf.square(y - yhat) / tf.square(sigma), axis=-1), axis=-1)


class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.attn_vector = None

    def build(self, input_shape):
        self.attn_vector = self.add_weight(
            name='attention_vector',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True,
            dtype='float32'
        )

        super(Attention, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        :param x: x is (batch, sentences, features)
        :return:
        """
        weights = K.squeeze(K.dot(x, self.attn_vector), axis=-1) # batch x sentences
        return K.sum(weights[:, :, None] * x, axis=-2) # batch x features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class CNNAttentionModel(object):

    def __init__(self, vocab_size):
        self.embedding_size = 100
        self.target_size = 3
        self.vocab_size = vocab_size
        self.n_grams = [1, 2, 3, 4]
        self.n_filters = 16

    def compile(self, embedding_matrix):

        def maxpool(x):
            return K.max(x, axis=-2)

        def maxpool_shape_4d(input_shape):
            return input_shape[0], input_shape[1], input_shape[-1]

        text_input = Input(shape=(None, None), dtype='int32') # batch x sentences x tokens

        glove_embeddings = Embedding(
            output_dim=self.embedding_size, input_dim=self.vocab_size, weights=[embedding_matrix],
            trainable=False,
        )
        embeddings = glove_embeddings(text_input)

        # embeddings = Lambda(custom_embedding, custom_embedding_shape)(text_input)

        convs = []
        for ngram_size in self.n_grams:
            conv = Conv2D(
                filters=self.n_filters, kernel_size=(1, ngram_size), strides=1, data_format='channels_last',
            )(embeddings) # None, sentences, MSL - ngram_size + 1, n_filters
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)
            convs.append(
                Lambda(maxpool, maxpool_shape_4d)(conv) # None, sentences, n_filters
            )

        max_embeddings = Concatenate(axis=-1)(convs) # None, sentences, n_filters x n_grams
        # max_embeddings = Dropout(0.3)(max_embeddings)


        dense = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(max_embeddings)
        dense = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(dense)
        attention = Attention()(dense)
        dense = Dense(self.target_size, activation='linear', kernel_initializer='glorot_uniform')(attention)
        # dense = Reshape((self.target_size, 2), name='lognormal')(dense) # None, target_size, 2

        self.model = Model(
            inputs=[text_input],
            outputs=[dense]
        )
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error'#normal_likelihood_3D,
        )
        print self.model.summary()
