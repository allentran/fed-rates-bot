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
            hidden_sizes=None,
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

        self.inputs_indexes = TT.imatrix() # n_words x n_sentences
        self.regimes = TT.iscalar() # minibatch
        self.doc_types = TT.iscalar() # minibatch

        self.unique_inputs = TT.ivector()
        self.unique_regimes = TT.ivector()
        self.unique_doc_types = TT.ivector()

        self.mask = TT.matrix() # n_words x n_sentences
        self.outputs = TT.vector() # n_target_rates

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

        regime_vectors = regime_layer.V[self.regimes]
        doctype_vectors = doctype_layer.V[self.doc_types]
        inputs = word_vectors_layer.V[self.inputs_indexes]

        preprocess_layer = layers.DenseLayer(
            inputs,
            input_size,
            hidden_sizes[0],
            activation=TT.nnet.relu
        )

        lstmforward_layer = layers.LSTMLayer(
            preprocess_layer.h_outputs,
            hidden_sizes[1],
            hidden_sizes[0],
            truncate=truncate,
        )

        lstmforward2_layer = layers.LSTMLayer(
            lstmforward_layer.h_outputs,
            hidden_sizes[1],
            hidden_sizes[1],
            truncate=truncate,
        )

        # max within a sentence (pick out phrases), then max over sentences
        # note first max eliminates first axis, so 2nd max(axis=0) kills 2nd axis
        max_pooled_words = (lstmforward2_layer.h_outputs * self.mask[:, :, None]).max(axis=0).max(axis=0)
        words_and_context = TT.concatenate(
            [
                max_pooled_words,
                regime_vectors,
                doctype_vectors
            ],
        )

        preoutput_layer = layers.DenseLayer(
            words_and_context,
            hidden_sizes[1] + doctype_size + regime_size,
            hidden_sizes[2],
            normalize_axis=0,
            feature_axis=1,
            activation=TT.nnet.relu
        )

        output_layer = layers.DenseLayer(
            preoutput_layer.h_outputs,
            hidden_sizes[2],
            (2 + target_size) * n_mixtures,
        )

        mixture_density_layer = layers.MixtureDensityLayer(
            output_layer.h_outputs[None, None, :],
            self.outputs[None, None, :],
            target_size=output_size,
            n_mixtures=n_mixtures
        )

        self.layers = [
            preprocess_layer,
            lstmforward_layer,
            lstmforward2_layer,
            preoutput_layer,
            output_layer,
        ]

        l2_cost = 0
        for layer in self.layers:
            l2_cost += l2_penalty * layer.get_l2sum()
        l2_cost += l2_penalty * regime_layer.get_l2sum(self.regimes)
        l2_cost += l2_penalty * doctype_layer.get_l2sum(self.doc_types)
        l2_cost += l2_penalty * word_vectors_layer.get_l2sum(self.inputs_indexes)

        self.loss_function = mixture_density_layer.nll_cost.sum()

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
