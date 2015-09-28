__author__ = 'allentran'

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
            doctype_size=5):

        self.inputs = TT.tensor3() # n_words x minibatch x features
        self.outputs = TT.matrix() # minibatch x n_target_rates
        self.regimes = TT.ivector() # minibatch
        self.doc_types = TT.ivector() # minibatch
        self.mask = TT.matrix() # n_words x minibatch

        regime_layer = layers.VectorEmbeddings(
            n_vectors=n_regimes,
            size=regime_size
        )

        doctype_layer = layers.VectorEmbeddings(
            n_vectors=2,
            size=doctype_size
        )

        regime_vectors = regime_layer.V[self.regimes]
        doctype_vectors = doctype_layer.V[self.doc_types]

        preprocess_layer = layers.DenseLayer(
            self.inputs,
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

        lstmbackward_layer = layers.LSTMLayer(
            preprocess_layer.h_outputs[::-1, :, :],
            hidden_sizes[1],
            hidden_sizes[0],
            truncate=truncate,
        )
        
        lstm_concat = TT.concatenate(
            [
                lstmforward_layer.h_outputs,
                lstmbackward_layer.h_outputs,
            ],
            axis=2
        )

        max_pooled_words = (lstm_concat * self.mask[:, :, None]).max(axis=0)
        words_and_context = TT.concatenate(
            [
                max_pooled_words,
                regime_vectors,
                doctype_vectors
            ],
            axis=1
        )

        preoutput_layer = layers.DenseLayer(
            words_and_context,
            hidden_sizes[1] * 2 + doctype_size + regime_size,
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
            output_layer.h_outputs[None, :, :],
            self.outputs[None, :, :],
            target_size=output_size,
        )

        self.layers = [
            preprocess_layer,
            lstmbackward_layer,
            lstmforward_layer,
            preoutput_layer,
            output_layer,
        ]

        l2_cost = 0
        for layer in self.layers:
            l2_cost += l2_penalty * layer.get_l2sum()
        l2_cost += l2_penalty * regime_layer.get_l2sum(self.regimes)
        l2_cost += l2_penalty * doctype_layer.get_l2sum(self.doc_types)

        self.loss_function = mixture_density_layer.nll_cost.sum()

        updates = []
        for layer in self.layers:
            updates += layer.get_updates(self.loss_function + l2_cost)
        updates += regime_layer.get_updates(self.loss_function + l2_cost, self.regimes)
        updates += doctype_layer.get_updates(self.loss_function + l2_cost, self.doc_types)

        self._cost_and_update = theano.function(
            inputs=[self.inputs, self.outputs, self.mask, self.regimes, self.doc_types],
            outputs=self.loss_function,
            updates=updates
        )

        self._cost= theano.function(
            inputs=[self.inputs, self.outputs, self.mask, self.regimes, self.doc_types],
            outputs=self.loss_function,
        )

        self._output = theano.function(
            inputs=[self.inputs, self.mask, self.regimes, self.doc_types],
            outputs=mixture_density_layer.outputs,
        )

    def get_cost_and_update(self, inputs, outputs, mask, regimes, doctypes):
        return self._cost_and_update(inputs, outputs, mask, regimes, doctypes)

    def get_cost(self, inputs, outputs, mask, regimes, doctypes):
        return self._cost(inputs, outputs, mask, regimes, doctypes)

    def get_output(self, inputs, mask, regimes, doctypes):
        return self._output(inputs, mask, regimes, doctypes)
