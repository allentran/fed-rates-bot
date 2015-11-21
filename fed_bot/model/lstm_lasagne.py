__author__ = 'allentran'

import numpy as np
import lasagne
import theano
import theano.tensor as TT

class LastTimeStepLayer(lasagne.layers.Layer):

    def __init__(self, incoming, batch_size, last_indexes, **kwargs):
        super(LastTimeStepLayer, self).__init__(incoming, **kwargs)
        self.batch_size = batch_size
        self.last_indexes = last_indexes

    def get_output_for(self, input, **kwargs):
        return input[TT.arange(self.batch_size), self.last_indexes]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[2]

class FedLSTMLasagne(object):

    def __init__(
            self,
            n_words=None,
            word_size=None,
            n_regimes=None,
            regime_size=None,
            n_docs=None,
            doc_size=None,
            lstm_size=32
    ):

        self.inputs_indexes = TT.imatrix() # batch_size x T
        self.last_indexes = TT.ivector() # batch_size
        self.regimes = TT.ivector() # minibatch
        self.doc_types = TT.ivector() # minibatch

        word_input_layer = lasagne.layers.InputLayer(shape=(None, None), input_var=self.inputs_indexes)
        regime_input_layer = lasagne.layers.InputLayer(shape=(None, ), input_var=self.regimes)
        doc_input_layer = lasagne.layers.InputLayer(shape=(None, ), input_var=self.doc_types)

        batch_size, T = word_input_layer.input_var.shape

        word_embeddings = lasagne.layers.EmbeddingLayer(word_input_layer, n_words, word_size)
        regime_embeddings = lasagne.layers.EmbeddingLayer(regime_input_layer, n_regimes, regime_size)
        doc_embeddings = lasagne.layers.EmbeddingLayer(doc_input_layer, n_docs, doc_size)

        word_embeddings = lasagne.layers.ReshapeLayer(word_embeddings, (-1, word_size))

        preprocessed_layer = lasagne.layers.DenseLayer(word_embeddings, lstm_size)

        reshaped_preprocessed_layer = lasagne.layers.ReshapeLayer(preprocessed_layer, shape=(batch_size, T, lstm_size))
        lstm_layer = lasagne.layers.LSTMLayer(reshaped_preprocessed_layer, lstm_size)
        lstm_layer2 = lasagne.layers.LSTMLayer(lstm_layer, lstm_size)

        lstm_last = LastTimeStepLayer(lstm_layer2, batch_size, self.last_indexes)

        loss = lasagne.layers.get_output(lstm_last).mean()

        params = lasagne.layers.get_all_params(lstm_last, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        self._train = theano.function([self.inputs_indexes, self.last_indexes], loss, updates=updates)

    def train(self):

        self._train(np.ones((7, 2)).astype('int32'), np.ones(7).astype('int32'))


if __name__ == "__main__":

    fedlstm_model = FedLSTMLasagne(20, 5, 50, 2, 13, 10)
    fedlstm_model.train()
