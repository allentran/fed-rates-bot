__author__ = 'allentran'

import theano
import theano.tensor as TT
from theano_layers import layers

class FedLSTM(object):

    def __init__(self, input_size=300, output_size=3, hidden_sizes=None, truncate=30, n_mixtures=5, target_size=3):

        self.inputs = TT.tensor3() # n_words x minibatch x features
        self.outputs = TT.matrix() # minibatch x n_target_rates
        self.output_mask = TT.ivector() # minibatch x n_target_rates

        minibatch_size = self.inputs.shape[1]

        preprocess_layer = layers.DenseLayer(
            self.inputs,
            input_size,
            hidden_sizes[0],
            activation=TT.tanh
        )

        lstm1_layer = layers.LSTMLayer(
            preprocess_layer.h_outputs,
            hidden_sizes[1],
            hidden_sizes[0],
            truncate=truncate,
        )

        lstm2_layer = layers.LSTMLayer(
            lstm1_layer.h_outputs,
            hidden_sizes[2],
            hidden_sizes[1],
            truncate=truncate,
        )

        preoutput_layer = layers.DenseLayer(
            lstm2_layer.h_outputs[self.output_mask, theano.tensor.arange(minibatch_size), :],
            hidden_sizes[2],
            hidden_sizes[3],
            normalize_axis=0,
            feature_axis=1,
            activation=TT.tanh
        )

        output_layer = layers.DenseLayer(
            preoutput_layer.h_outputs,
            hidden_sizes[3],
            (2 + target_size) * n_mixtures,
        )

        mixture_density_layer = layers.MixtureDensityLayer(
            output_layer.h_outputs[None, :, :],
            self.outputs[None, :, :],
            target_size=output_size,
        )

        self.loss_function = mixture_density_layer.nll_cost.sum()

        self.layers = [
            preprocess_layer,
            lstm1_layer,
            lstm2_layer,
            output_layer
        ]

        updates = []
        for layer in self.layers:
            updates += layer.get_updates(self.loss_function)

        self._cost_and_update = theano.function(
            inputs=[self.inputs, self.outputs, self.output_mask],
            outputs=self.loss_function,
            updates=updates
        )

    def get_cost_and_update(self, inputs, outputs, mask):
        return self._cost_and_update(inputs, outputs, mask)
