import pickle

import tensorflow as tf
from tensorflow.keras.layers import (RNN, Concatenate, Dense, Dropout, Input,
                                     Masking)
from tensorflow.keras.optimizers import Adam

from helpers.helper_config import Config
from helpers.helper_enums import Dataset, RNNCell, TrainingLoop
from layers.recurrent_layers.layer_di_lstm import DILSTMCell
from layers.recurrent_layers.layer_di_lstm_gaus import DILSTMGaus
from layers.recurrent_layers.layer_di_rnn import DIRNNCell
from layers.recurrent_layers.layer_lstm import OriginalLSTMCell
from layers.recurrent_layers.layer_rnn import OriginalRNNCell
from models.model_loss import Custom_Loss, Custom_Metric

def nnelu(inputs):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))

class NovelModel:
    def __init__(self):
        self.model = None

    def initialize_self(self):
        main_input = Input(shape=(None, Config.edge_size), dtype="float32", name='main_input')
        feature_input = Input(shape=(Config.feature_input_size,), dtype="float32", name="feature_input")

        rnn_cell = get_cell_from_config()(Config.recurrent_layer_size, recurrent_dropout=Config.recurrent_dropout_rate)

        # Apply the masking and introduce the LSTM layer
        composed_layer = Masking(mask_value=0., input_shape=(None, Config.edge_size))(main_input)
        composed_layer = RNN(rnn_cell, input_shape=(None, None, Config.edge_size))(composed_layer)

        # MLP to extract and find depencies between features
        composed_layer = tf.keras.layers.concatenate([composed_layer, feature_input])
        composed_layer = Dense(Config.dense_layer_sizes[0], activation='relu')(composed_layer)
        composed_layer = Dropout(Config.dropout_rate)(composed_layer)
        composed_layer = Dense(Config.dense_layer_sizes[1], activation='relu')(composed_layer)

        # Softmax layer to return the data into distribution
        if Config.dataset == Dataset.Normal:
            composed_layer = Dense(Config.output_size, activation='softmax')(composed_layer)
        else:
            alphas = Dense(Config.mdn_mixes, activation='softmax', name='Alphas')(composed_layer)
            mus = Dense(Config.mdn_mixes, activation=None, name='Mus')(composed_layer)
            sigmas = Dense(Config.mdn_mixes, activation=nnelu, name='Sigmas')(composed_layer)

            composed_layer = Concatenate(name='Output')([alphas, mus, sigmas])

        self.model = tf.keras.Model(inputs=[main_input, feature_input], outputs=composed_layer)

        self.model.summary()
        print("Starting")

        return self.model

    def compile_self(self):
        model = self.initialize_self()
        model.compile(
            loss=Config.loss_function,
            optimizer=Config.optimizer,
            metrics=[Custom_Metric.correlation, Custom_Metric.intersection])
        return model

def get_cell_from_config():
    if Config.recurrent_cell == RNNCell.LSTM_Cell:
        return DILSTMCell
    elif Config.recurrent_cell == RNNCell.Original_RNN_Cell:
        return OriginalRNNCell
    elif Config.recurrent_cell == RNNCell.RNN_Cell:
        return DIRNNCell
    elif Config.recurrent_cell == RNNCell.Original_LSTM_Cell:
        return OriginalLSTMCell
    elif Config.recurrent_cell == RNNCell.MDN_LSTM_CELL:
        return DILSTMGaus
