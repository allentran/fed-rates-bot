
import tensorflow as tf
import tensorflow.models.rnn
import numpy as np

class DenseLayer(object):

    def __init__(self, x_input, in_size, out_size):

        self.x = x_input
        self.W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=tf.sqrt(tf.constant(2.0, shape=[]) / tf.cast(in_size * out_size, 'float'))))
        self.b = tf.Variable(tf.constant(0.1, tf.float32, shape=[out_size]))

    def activated(self, activation=tf.nn.relu):
        return activation(tf.matmul(self.x, self.W) + self.b)

class FedLSTM(object):

    def __init__(
        self,
        embedding_size=300,
        n_rates=3,
        lstm_size=64,
        hidden_size=None,
        n_mixtures=5,
        target_size=3,
        l2_penalty=0,
        n_regimes=6,
        n_docs=2,
        regime_size=5,
        doctype_size=5,
        vocab_size=10,
        num_steps=100,
        word_vectors=None,
    ):
        early_stop = tf.placeholder(tf.int32)
        batch_size = tf.placeholder(tf.int32)
        n_sentences = tf.placeholder(tf.int32)

        self.word_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.regime_embeddings = tf.Variable(tf.random_uniform([n_regimes, regime_size], -1.0, 1.0))
        self.doctype_embeddings = tf.Variable(tf.random_uniform([n_docs, doctype_size], -1.0, 1.0))

        # Placeholders for word inputs: n_words_in_sentence x batch_size x n_sentence
        #                  doctypes: batch_size x n_rates
        #                  regimes: batch_size x n_rates
        #                  rates: batch_size x n_rates
        #                  mask: numsteps x batch_size x n_sentence

        word_inputs = tf.placeholder(tf.int32, shape=[num_steps, None, None])
        regime_inputs = tf.placeholder(tf.int32, shape=[None])
        doctype_inputs = tf.placeholder(tf.int32, shape=[None])
        rates = tf.placeholder(tf.float32, shape=[None, n_rates])
        mask = tf.placeholder(tf.float32, shape=[num_steps, None, None])

        initializer = tf.random_uniform_initializer(-1, 1)

        words_embeddings = tf.nn.embedding_lookup(self.word_embeddings, word_inputs) # t x n_batch x n_sentence x embedding_size
        regime_embeddings = tf.nn.embedding_lookup(self.regime_embeddings, regime_inputs)
        doctype_embeddings = tf.nn.embedding_lookup(self.doctype_embeddings, doctype_inputs)

        preprocess_layer = DenseLayer(tf.reshape(words_embeddings, [num_steps * 3 * 7, embedding_size]), embedding_size, lstm_size)

        inputs = tf.split(0, num_steps, tf.reshape(preprocess_layer.activated(), [num_steps, 3 * 7, lstm_size]))
        inputs = [tf.squeeze(input_) for input_ in inputs]

        cell = tf.models.rnn.rnn_cell.LSTMCell(lstm_size, lstm_size, initializer=initializer)
        initial_state = cell.zero_state(batch_size * n_sentences, tf.float32)
        outputs1, states = tf.models.rnn.rnn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop, scope="RNN1")

        cell2 = tf.models.rnn.rnn_cell.LSTMCell(lstm_size, lstm_size, initializer=initializer)
        initial_state2 = cell2.zero_state(batch_size * n_sentences, tf.float32)
        outputs2, states = tf.models.rnn.rnn.rnn(cell2, outputs1, initial_state=initial_state2, sequence_length=early_stop, scope="RNN2")

        outputs2 = tf.reshape(tf.pack(outputs2), [num_steps, 3, 7, lstm_size])

        lstm_outputs = tf.slice(outputs2, [0, 0, 0, 0], tf.pack([early_stop, -1, -1, -1]))
        masked_lstm_outputs = tf.mul(mask, lstm_outputs)

        max_pooled_words = tf.reduce_max(masked_lstm_outputs, reduction_indices=[0, 2])

        iop = tf.initialize_all_variables()
        #create initialize op, this needs to be run by the session!
        session = tf.Session()
        session.run(iop)

        feed = {
            early_stop :10,
            batch_size: 3,
            n_sentences: 7,
            regime_inputs: np.array([0, 0, 0]).astype('int32'),
            doctype_inputs: np.array([0, 0, 0]).astype('int32'),
            mask:np.random.rand(num_steps, 3, 7).astype('float32'),
            word_inputs:np.random.rand(num_steps, 3, 7).astype('float32')
        }

        outs = session.run(max_pooled_words, feed_dict=feed)

def main():
    fed = FedLSTM(hidden_size=23)

if __name__ == '__main__':
    main()
