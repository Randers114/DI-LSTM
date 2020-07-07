import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Dropout, Input, Masking, Concatenate
from tensorflow.keras.optimizers import Adam
from helpers.helper_enums import Dataset

from models.model_loss import Custom_Loss, Custom_Metric
from helpers.helper_config import Config


def nnelu(inputs):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))

class MLPGate():
    def __init__(self):
        self.init_gate()


    def init_gate(self):
        main_input = Input(shape=(Config.edge_size * 2))

        layer = Dense(Config.gate_layer_size, activation='relu')(main_input)

        for _ in range(Config.mlp_hidden_layer_numbers):
            layer = Dense(Config.gate_layer_size, activation='relu')(layer)
            layer = Dropout(Config.mlp_dropout)(layer)

        if Config.dataset == Dataset.Normal:
            output = Dense(Config.output_size, activation='softmax')(layer)
        else:
            alphas = Dense(Config.mdn_mixes, activation='softmax', name='Alphas')(layer)
            mus = Dense(Config.mdn_mixes, activation=None, name='Mus')(layer)
            sigmas = Dense(Config.mdn_mixes, activation=nnelu, name='Sigmas')(layer)

            output = Concatenate(name='Output')([alphas, mus, sigmas])

        self.gate = tf.keras.Model(inputs=main_input, outputs=output)

        self.gate.compile(
            loss=Config.loss_function,
            optimizer=Adam()
        )

