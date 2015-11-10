
import tensorflow as tf
import tensorflow.models.rnn.rnn_cell
import numpy as np

class DenseLayer(object):

    def __init__(self, input):

        shape = input.get_shape()
        self.x = input
        self.W = tf.Variable(tf.truncated_normal(shape, stddev=tf.sqrt(tf.constant(2.0, shape=shape) / (shape[0] * shape[1]))))
        self.b = tf.Variable(tf.constant(0.1, tf.float32, shape=shape))

    def activation(self, x):
        return tf.nn.relu(tf.matmul(x, self.W) + self.b)

class FedLSTM(object):

    def __init__(
        self,
        embedding_size=300,
        n_rates=3,
        hidden_sizes=None,
        truncate=10,
        n_mixtures=5,
        target_size=3,
        l2_penalty=0,
        n_regimes=6,
        n_docs=2,
        regime_size=5,
        doctype_size=5,
        vocab_size=10,
        word_vectors=None,
    ):

        self.word_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.regime_embeddings = tf.Variable(tf.random_uniform([n_regimes, regime_size], -1.0, 1.0))
        self.doctype_embeddings = tf.Variable(tf.random_uniform([n_docs, doctype_size], -1.0, 1.0))

        # Placeholders for word inputs: batch_size x n_words_in_sentence x n_sentence
        #                  doctypes: batch_size x n_rates
        #                  regimes: batch_size x n_rates
        #                  rates: batch_size x n_rates

        word_inputs = tf.placeholder(tf.int32, shape=[None, None, None])
        regime_inputs = tf.placeholder(tf.int32, shape=[None])
        doctype_inputs = tf.placeholder(tf.int32, shape=[None])
        rates = tf.placeholder(tf.float32, shape=[None, n_rates])
        mask = tf.placeholder(tf.float32, shape=[None, None, None])

        words_embeddings = tf.nn.embedding_lookup(self.word_embeddings, word_inputs)
        regime_embeddings = tf.nn.embedding_lookup(self.regime_embeddings, regime_inputs)
        doctype_embeddings = tf.nn.embedding_lookup(self.doctype_embeddings, doctype_inputs)

        preprocess_layer = DenseLayer(words_embeddings)


def main():
    fed = FedLSTM()

if __name__ == '__main__':
    main()
