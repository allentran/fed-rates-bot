__author__ = 'allentran'

import theano
import theano.tensor as TT
from theano_layers import layers

class FedLSTM(object):

    def __init__(self, input_size=300, output_size=3, hidden_sizes=None, truncate=10, n_mixtures=5, target_size=3):

        self.inputs = TT.tensor3() # n_words x minibatch x features
        self.outputs = TT.matrix() # minibatch x n_target_rates
        self.output_mask = TT.ivector() # minibatch x n_target_rates

        minibatch_size = self.inputs.shape[1]

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
            axis=0
        )

        preoutput_layer = layers.DenseLayer(
            lstm_concat.mean(axis=0),
            hidden_sizes[1],
            hidden_sizes[2],
            normalize_axis=0,
            feature_axis=1,
            activation=TT.tanh
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

        self.loss_function = mixture_density_layer.nll_cost.sum()

        self.layers = [
            preprocess_layer,
            lstmbackward_layer,
            lstmforward_layer,
            preoutput_layer,
            output_layer
        ]

        updates = []
        for layer in self.layers:
            updates += layer.get_updates(self.loss_function)

        self._cost_and_update = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=self.loss_function,
            updates=updates
        )

        self._cost= theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=self.loss_function,
        )

        self._output = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=mixture_density_layer.outputs,
        )

    def get_cost_and_update(self, inputs, outputs, mask):
        return self._cost_and_update(inputs, outputs)

    def get_cost(self, inputs, outputs):
        return self._cost(inputs, outputs)

    def get_output(self, inputs, outputs):
        return self._output(inputs, outputs)
