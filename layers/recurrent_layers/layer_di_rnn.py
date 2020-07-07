import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Reshape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import (
    DropoutRNNCellMixin, _caching_device, _generate_zero_filled_state_for_cell)
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures

from helpers.helper_config import Config
from helpers.helper_enums import Gates
from helpers.helper_length_normalizer import LengthNormalizer
from layers.layer_gate import Gate
from layers.layer_mlp_gate import MLPGate

RECURRENT_DROPOUT_WARNING_MSG = (
    "RNN `implementation=2` is not supported when `recurrent_dropout` is set. "
    "Using `implementation=1`."
)

class DIRNNCell(DropoutRNNCellMixin, Layer):
    def __init__(
        self,
        units,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs
    ):
        self._enable_caching_device = kwargs.pop("enable_caching_device", False)
        super(DIRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.state_size = [
            self.units,
            23,
        ]  # Changed to having two elements. 23 is the size of an edge-input
        self.output_size = self.units

        # Additional fields needed to handle the dual-input structure.
        self.current_batch_size = 0
        self.initial_state = []
        self.softmax = Dense(Config.output_size, activation="softmax")

        self.what_gate = Config.gate
        self.gate = Gate()
        self.mlp = MLPGate()
        # Trainable mlp gate
        self.mlp = self.mlp.gate

        self.normalizer = LengthNormalizer()

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        def call_gate(inputs, prev_output):
            """We call this function with the input and previous output - depending on the initialized gate
               we choose to combine the edges in different ways, and then return this combined distribution
               representing the two edges as one combined edge, or path."""
            gate_output = None
            if self.what_gate == Gates.MLPGate:
                gate_output = self.mlp(tf.concat([inputs, prev_output], 1))
            elif self.what_gate == Gates.MathGate:
                gate_output = self.gate.combine_tensors(
                    prev_output, inputs, input_lengths, prev_output_lengths
                )

            return gate_output

        def init_first_training(states):
            """When we receive the first input from the training loop, we need to overwrite the state and
               prev_output to be that of the initial state. 
               If we are not calling from the first iteration of the training loop, we just return the 
               states as they are."""

            if (
                states[0].shape == (self.current_batch_size, 300)
                and tf.keras.backend.sum(states[0]) == 0
            ):
                state = self.initial_state[0]
                prev_output = tf.reshape(
                    self.initial_state[1], (self.current_batch_size, 23)
                )
                self.gate = Gate()
                self.normalizer = LengthNormalizer()

                return state, prev_output

            return states[0], states[1]

        state, prev_output = init_first_training(states)
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(state, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(state, training)

        # We extract the lengths of each edge, such that we can use it for the output edge.
        input_lengths = tf.slice(inputs, [0, 22], [self.current_batch_size, 1])
        prev_output_lengths = tf.slice(
            prev_output, [0, 22], [self.current_batch_size, 1]
        )
        combined_length = input_lengths + prev_output_lengths

        # We normalize the lengths, such that they can be used in the MLP gate.
        input_lengths, prev_output_lengths = self.normalizer.normalize(
            input_lengths, prev_output_lengths
        )
        inputs = tf.concat(
            [tf.slice(inputs, [0, 0], [self.current_batch_size, 22]), input_lengths], 1
        )
        prev_output = tf.concat(
            [
                tf.slice(prev_output, [0, 0], [self.current_batch_size, 22]),
                prev_output_lengths,
            ],
            1,
        )

        gate_output = tf.concat([call_gate(inputs, prev_output), combined_length], 1)

        if dp_mask is not None:
            h = K.dot(gate_output * dp_mask, self.kernel)
        else:
            h = K.dot(gate_output, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            state = state * rec_dp_mask

        output = h + K.dot(state, self.recurrent_kernel)

        if self.activation is not None:
            output = self.activation(output)

        # Try used as the first couple of iterations has no contents, and will potentioally crash
        try:
            # We send the output of the cell through a softmax layer, to find the distribution, which
            # is then parsed as the second state output (prev_input) to the next iteration.
            softmaxed = self.softmax(output)

            # We combine the softmaxed distribution with the non-normalized length of both edges.
            combined_edges = tf.concat([softmaxed, combined_length], 1)
            state_output = [output, combined_edges]

            return output, state_output
        except:
            print("Precondition not valid, Softmax:")

        return output, [output, states[1]]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }

        base_config = super(DIRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
