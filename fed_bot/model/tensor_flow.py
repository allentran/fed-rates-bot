
import tensorflow as tf
import tensorflow.models.rnn
import numpy as np

class DenseLayer(object):

    def __init__(self, in_size, out_size):

        self.W = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=tf.sqrt(tf.constant(2.0, shape=[]) / tf.cast(in_size * out_size, 'float'))))
        self.b = tf.Variable(tf.constant(0.1, tf.float32, shape=[out_size]))

    def activated(self, x, activation=tf.nn.relu):
        return activation(tf.matmul(x, self.W) + self.b)

class FedLSTM(object):

    def __init__(
        self,
        embedding_size=300,
        n_rates=3,
        lstm_size=13,
        truncate=10,
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

        self.word_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.regime_embeddings = tf.Variable(tf.random_uniform([n_regimes, regime_size], -1.0, 1.0))
        self.doctype_embeddings = tf.Variable(tf.random_uniform([n_docs, doctype_size], -1.0, 1.0))

        #                  doctypes: batch_size x n_rates
        #                  regimes: batch_size x n_rates
        #                  rates: batch_size x n_rates
        #                  mask: numsteps x batch_size x n_sentence

        word_inputs = tf.placeholder(tf.int32, shape=[num_steps, None, None])
        regime_inputs = tf.placeholder(tf.int32, shape=[None])
        doctype_inputs = tf.placeholder(tf.int32, shape=[None])
        rates = tf.placeholder(tf.float32, shape=[None, n_rates])
        mask = tf.placeholder(tf.float32, shape=[num_steps, None, None])

        n_sentences = 7 #tf.shape(word_inputs)[2]
        batch_size = 8 #tf.shape(word_inputs)[1]

        word_embeddings = tf.nn.embedding_lookup(self.word_embeddings, word_inputs)
        regime_embeddings = tf.nn.embedding_lookup(self.regime_embeddings, regime_inputs)
        doctype_embeddings = tf.nn.embedding_lookup(self.doctype_embeddings, doctype_inputs)


        preprocess_layer = DenseLayer(embedding_size, lstm_size)
        preprocessed_activated = tf.reshape(
            preprocess_layer.activated(
                tf.reshape(
                    word_embeddings,
                    (num_steps * word_embeddings.get_shape()[1] * n_sentences, embedding_size)
                )
            ),
            (num_steps, batch_size * n_sentences, lstm_size)
        )

        lstm_inputs = [tf.squeeze(input_t) for input_t in tf.split(0, num_steps, preprocessed_activated)]

        initializer = tf.random_uniform_initializer(-1, 1)

        with tf.device("/cpu:0"):
            cell1 = tf.models.rnn.rnn_cell.LSTMCell(lstm_size, lstm_size, initializer=initializer)
            initial_state1 = cell1.zero_state(batch_size * n_sentences, tf.float32)
            outputs1, states1 = tf.models.rnn.rnn.rnn(cell1, lstm_inputs, initial_state=initial_state1,
                                sequence_length=early_stop, scope="RNN1")

        with tf.device("/cpu:0"):
            cell2 = tf.models.rnn.rnn_cell.LSTMCell(lstm_size, lstm_size, initializer=initializer)
            initial_state2 = cell2.zero_state(batch_size * n_sentences, tf.float32)
            outputs2, states2 = tf.models.rnn.rnn.rnn(cell2, outputs1, initial_state=initial_state2,
                                sequence_length=early_stop, scope="RNN2")

        truncated_lstm_output = tf.reshape(
            tf.slice(tf.pack(outputs2), [0, 0, 0], tf.pack([early_stop, -1, -1])),
            (early_stop, batch_size, n_sentences, lstm_size)
        )

        # Create initialize op, this needs to be run by the session!
        iop = tf.initialize_all_variables()

        # Create session with device logging
        session = tf.Session()

        # Actually initialize, if you don't do this you get errors about uninitialized
        session.run(iop)

        # First call to session has overhead? lets get that cleared out
        feed = {
            early_stop: 3,
            word_inputs: np.random.rand(num_steps, 8, 7).astype('float32')
        }

        session.run(truncated_lstm_output, feed_dict=feed)

def main():
    fed = FedLSTM(
        lstm_size=5
    )

if __name__ == '__main__':
    main()
