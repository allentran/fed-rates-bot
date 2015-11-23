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

    # from batch x T x k to batch x k
    def get_output_for(self, input, **kwargs):
        return input[TT.arange(self.batch_size), self.last_indexes]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[2]

# input should be batch x (2 + target_size) * n_mixtures
# target is batch x target_size
def mixture_density_loss(input, targets, target_size, n_mixtures, mask=None):

    batch_size = input.shape[0]

    m_by_n = n_mixtures * target_size

    priors = TT.nnet.softmax(input[:, :n_mixtures]) # batch, n_mixtures
    means = input[:, n_mixtures:n_mixtures + m_by_n].reshape((batch_size, target_size, n_mixtures)) # batch x target x mixtures
    stds = TT.exp(input[:, n_mixtures + m_by_n:]).reshape((batch_size, n_mixtures)) # batch x n_mixtures

    kernel_constant =  ((2 * np.pi) ** -0.5) * (1 / (stds ** target_size))
    norm_std =((targets[:, :, None] - means).norm(2, axis=1)) / (2 * TT.sqr(stds)) # normed over targets
    kernel = kernel_constant * TT.exp(-norm_std)

    e_prob = (priors * kernel).sum(axis=1) # summing over mixtures

    if mask:
        return -(mask * TT.log(e_prob))
    else:
        return -(TT.log(e_prob))

class FedLSTMLasagne(object):

    def __init__(
            self,
            n_words=None,
            word_size=None,
            n_regimes=None,
            regime_size=None,
            n_docs=None,
            doc_size=None,
            lstm_size=32,
            hidden_size=11,
            n_mixtures=2,
            target_size=3
    ):

        self.inputs_indexes = TT.imatrix() # sentences x T
        self.last_indexes = TT.ivector() # sentences
        self.targets = TT.matrix(dtype=theano.config.floatX) # batch_size
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

        preprocessed_layer = lasagne.layers.DenseLayer(word_embeddings, lstm_size, )
        preprocessed_dropout = lasagne.layers.DropoutLayer(preprocessed_layer, p=0.5)

        reshaped_preprocessed_layer = lasagne.layers.ReshapeLayer(preprocessed_dropout, shape=(batch_size, T, lstm_size))

        forget_gate = lasagne.layers.Gate(b=lasagne.init.Constant(5.0))
        lstm_layer = lasagne.layers.LSTMLayer(reshaped_preprocessed_layer, lstm_size, forgetgate=forget_gate)
        forget_gate = lasagne.layers.Gate(b=lasagne.init.Constant(5.0))
        lstm_layer2 = lasagne.layers.LSTMLayer(lstm_layer, lstm_size, forgetgate=forget_gate)

        sentence_summary = LastTimeStepLayer(lstm_layer2, batch_size, self.last_indexes)
        # sentence_summary = lasagne.layers.ReshapeLayer(sentence_summary, shape=(batch_size, 1, lstm_size))

        # forget_gate = lasagne.layers.Gate(b=lasagne.init.Constant(5.0))
        # doc_summary = lasagne.layers.LSTMLayer(sentence_summary, lstm_size, forgetgate=forget_gate)
        # doc_summary = lasagne.layers.ReshapeLayer(doc_summary, shape=(batch_size, lstm_size))

        merge_layer = lasagne.layers.ConcatLayer([sentence_summary, regime_embeddings, doc_embeddings])
        merge_dropout = lasagne.layers.DropoutLayer(merge_layer, p=0.5)

        preoutput_layer = lasagne.layers.DenseLayer(merge_dropout, hidden_size)
        output_layer = lasagne.layers.DenseLayer(preoutput_layer, (2 + target_size) * n_mixtures, nonlinearity=None)

        loss = mixture_density_loss(lasagne.layers.get_output(output_layer), self.targets, target_size, n_mixtures).mean()

        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        updates = lasagne.updates.adadelta(loss, params)

        self._train = theano.function(
            [
                self.inputs_indexes,
                self.last_indexes,
                self.regimes,
                self.doc_types,
                self.targets
            ],
            loss,
            updates=updates
        )

    def train(self, targets, sentences, last_indexes, regimes, doctypes):
        return self._train(
            sentences,
            last_indexes,
            regimes,
            doctypes,
            targets
        )
