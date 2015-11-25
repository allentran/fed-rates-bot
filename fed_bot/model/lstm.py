__author__ = 'allentran'

import numpy as np
import theano
import theano.tensor as TT
from theano_layers import layers

class FedLSTM(object):

    def __init__(
            self,
            input_size=300,
            output_size=3,
            hidden_size=None,
            lstm_size=None,
            truncate=10,
            n_mixtures=5,
            target_size=3,
            l2_penalty=0,
            n_regimes=6,
            regime_size=5,
            doctype_size=5,
            vocab_size=10,
            word_vectors=None,
    ):

        self.inputs_indexes = TT.tensor3(dtype='int32') # n_words x n_sentences x n_batch
        self.regimes = TT.ivector() # minibatch
        self.doc_types = TT.ivector() # minibatch

        self.unique_inputs = TT.ivector()
        self.unique_regimes = TT.ivector()
        self.unique_doc_types = TT.ivector()

        self.mask = TT.tensor3() # n_words x n_sentences
        self.outputs = TT.matrix() # n_batch x n_target_rates

        regime_layer = layers.VectorEmbeddings(
            n_vectors=n_regimes,
            size=regime_size
        )

        doctype_layer = layers.VectorEmbeddings(
            n_vectors=2,
            size=doctype_size
        )

        word_vectors_layer = layers.VectorEmbeddings(
            n_vectors=vocab_size,
            size=input_size
        )
        word_vectors_layer.V.set_value(word_vectors)

        regime_vectors = regime_layer.V[self.regimes] # n_batch x size
        doctype_vectors = doctype_layer.V[self.doc_types] # n_batch x size
        inputs = word_vectors_layer.V[self.inputs_indexes] # T x n_sentences x n_batch x size

        preprocess_layer = layers.DenseLayer(
            inputs,
            input_size,
            lstm_size,
            activation=TT.nnet.relu
        )
        # T x n_sentences x n_batch x hidden[0]

        lstmforward_layer = layers.LSTMStackedLayer(
            preprocess_layer.h_outputs,
            lstm_size,
            n_layers=2,
            input_size=lstm_size,
            truncate=truncate,
        )
        # T x n_sentences x n_batch x hidden[1]

        # max within a sentence (pick out phrases), then max over sentences
        # note first max eliminates first axis, so 2nd max(axis=0) kills 2nd axis

        max_pooled_words = (lstmforward_layer.h_outputs * self.mask[:, :, :, None]).max(axis=0).max(axis=0)
        # n_batch x hidden[1]

        words_and_context = TT.concatenate(
            [
                max_pooled_words,
                regime_vectors,
                doctype_vectors
            ],
            axis=1
        )
        # n_batch x hidden[2] + doctype_size + regime_size

        preoutput_layer = layers.DenseLayer(
            words_and_context,
            lstm_size + doctype_size + regime_size,
            hidden_size,
            activation=TT.nnet.relu,
            feature_axis=1,
            normalize_axis=0
        )
        # n_batch x hidden[3]

        output_layer = layers.DenseLayer(
            preoutput_layer.h_outputs,
            hidden_size,
            (2 + target_size) * n_mixtures,
            activation=TT.tanh
        )

        # n_batch x (2 * target_size) * n_mixtures

        mixture_density_layer = layers.MixtureDensityLayer(
            output_layer.h_outputs[None, :, :],
            self.outputs[None, :, :],
            target_size=output_size,
            n_mixtures=n_mixtures
        )

        self.layers = [
            preprocess_layer,
            lstmforward_layer,
            preoutput_layer,
            output_layer,
        ]

        l2_cost = 0
        for layer in self.layers:
            l2_cost += l2_penalty * layer.get_l2sum()
        l2_cost += l2_penalty * regime_layer.get_l2sum(self.unique_regimes)
        l2_cost += l2_penalty * doctype_layer.get_l2sum(self.unique_doc_types)
        l2_cost += l2_penalty * word_vectors_layer.get_l2sum(self.unique_inputs)

        self.loss_function = mixture_density_layer.nll_cost.mean()

        updates = []
        for layer in self.layers:
            updates += layer.get_updates(self.loss_function + l2_cost)
        updates += regime_layer.get_updates(self.loss_function + l2_cost, self.unique_regimes)
        updates += doctype_layer.get_updates(self.loss_function + l2_cost, self.unique_doc_types)
        updates += word_vectors_layer.get_updates(self.loss_function + l2_cost, self.unique_inputs)

        self._cost_and_update = theano.function(
            inputs=[
                self.inputs_indexes,
                self.outputs,
                self.mask,
                self.regimes,
                self.doc_types,
                self.unique_inputs,
                self.unique_doc_types,
                self.unique_regimes
            ],
            outputs=self.loss_function,
            updates=updates
        )

        self._cost= theano.function(
            inputs=[self.inputs_indexes, self.outputs, self.mask, self.regimes, self.doc_types],
            outputs=self.loss_function,
        )

        self._output = theano.function(
            inputs=[self.inputs_indexes, self.mask, self.regimes, self.doc_types],
            outputs=mixture_density_layer.outputs,
        )

    def get_cost_and_update(self, inputs, outputs, mask, regimes, doctypes):
        u_inputs = np.unique(inputs).flatten()
        u_regimes = np.unique(regimes).flatten()
        u_docs = np.unique(doctypes).flatten()
        return self._cost_and_update(inputs, outputs, mask, regimes, doctypes, u_inputs, u_docs, u_regimes)

    def get_cost(self, inputs, outputs, mask, regimes, doctypes):
        return self._cost(inputs, outputs, mask, regimes, doctypes)

    def get_output(self, inputs, mask, regimes, doctypes):
        return self._output(inputs, mask, regimes, doctypes)
