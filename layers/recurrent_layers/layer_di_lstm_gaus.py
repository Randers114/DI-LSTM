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


class DILSTMGaus(DropoutRNNCellMixin, Layer):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        implementation=1,
        **kwargs
    ):
        def nnelu(inputs):
            return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))
        self._enable_caching_device = kwargs.pop("enable_caching_device", False)
        super(DILSTMGaus, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        self.state_size = [self.units, self.units, 25]
        self.output_size = self.units

        # Additional fields needed to handle the dual-input structure.
        self.current_batch_size = 0
        self.initial_state = []
        # self.softmax = Dense(Config.output_size, activation="softmax")
        self.alphas = Dense(Config.mdn_mixes, activation='softmax', name='Alphas')
        self.mus = Dense(Config.mdn_mixes, activation=None, name='Mus')
        self.sigmas = Dense(Config.mdn_mixes, activation=nnelu, name='Sigmas')
        self.mdn_layer = Concatenate()
        
        self.mlp = MLPGate()
        # Trainable mlp gate
        self.mlp = self.mlp.gate

        self.normalizer = LengthNormalizer()
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate(
                        [
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + K.dot(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=None):
        def call_gate(inputs, prev_output):
            """We call this function with the input and previous output - depending on the initialized gate
            we choose to combine the edges in different ways, and then return this combined distribution
            representing the two edges as one combined edge, or path."""

            return  self.mlp(tf.concat([inputs, prev_output], 1))

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
                carry = self.initial_state[1]
                prev_output = tf.reshape(
                    self.initial_state[2], (self.current_batch_size, 25)
                )
                self.normalizer = LengthNormalizer()

                return state, carry, prev_output

            return states[0], states[1], states[2]

        h_tm1, c_tm1, prev_output = init_first_training(states)

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)
        # We extract the lengths of each edge, such that we can use it for the output edge.
        input_lengths = tf.slice(inputs, [0, 24], [self.current_batch_size, 1])
        prev_output_lengths = tf.slice(
            prev_output, [0, 24], [self.current_batch_size, 1]
        )
        combined_length = input_lengths + prev_output_lengths
        
        # We normalize the lengths, such that they can be used in the MLP gate.
        input_lengths, prev_output_lengths = self.normalizer.normalize(
            input_lengths, prev_output_lengths
        )
        inputs = tf.concat(
            [tf.slice(inputs, [0, 0], [self.current_batch_size, 24]), input_lengths], 1
        )
        prev_output = tf.concat(
            [
                tf.slice(prev_output, [0, 0], [self.current_batch_size, 24]),
                prev_output_lengths,
            ],
            1,
        )

        gate_output = tf.concat([call_gate(inputs, prev_output), combined_length], 1)

        if self.implementation == 1:
            if 0 < self.dropout < 1.0:
                inputs_i = gate_output * dp_mask[0]
                inputs_f = gate_output * dp_mask[1]
                inputs_c = gate_output * dp_mask[2]
                inputs_o = gate_output * dp_mask[3]
            else:
                inputs_i = gate_output
                inputs_f = gate_output
                inputs_c = gate_output
                inputs_o = gate_output
            k_i, k_f, k_c, k_o = array_ops.split(
                self.kernel, num_or_size_splits=4, axis=1
            )
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0
                )
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            z = K.dot(inputs, self.kernel)
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        # Try used as the first couple of iterations has no contents, and will potentioally crash
        try:
            # We send the output of the cell through a softmax layer, to find the distribution, which
            # is then parsed as the second state output (prev_input) to the next iteration.
            # softmaxed = self.softmax(h)
            alpha = self.alphas(h)
            my = self.mus(h)
            sigma = self.sigmas(h)
            
            mdn_output = self.mdn_layer([alpha, my, sigma])

            # mdn_output = concat_mdn_form(alpha, my, sigma, self.current_batch_size)

            # We combine the softmaxed distribution with the non-normalized length of both edges.
            combined_edges = tf.concat([mdn_output, combined_length], 1)
            state_output = [h, c, combined_edges]

            return h, state_output
        except:
            print("Precondition not valid, Softmax:")

        return h, [h, c, states[2]]

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "implementation": self.implementation,
        }
        base_config = super(DILSTMGaus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(
            _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)
        )
