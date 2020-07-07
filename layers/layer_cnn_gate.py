import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Reshape, Dropout

from helpers.helper_config import Config
from helpers.helper_sparse_connection import get_sparse_connection_tensor
from layers.layer_sparse_connected import Sparse
from models.model_loss import Custom_Loss, Custom_Metric

class CNNGate:
    def __init__(self):
        self.init_gate()

    def init_gate(self):
        input_layer = Input(shape=(46,))
        
        composed_layer = Dense(44, activation='relu')(input_layer)

        composed_layer = Reshape((2, 22, 1))(composed_layer)

        for conv_filter in Config.convolution_filters:
            composed_layer = Conv2D(Config.filter_count, conv_filter, activation='relu')(composed_layer)

        composed_layer = Flatten()(composed_layer)
        composed_layer = Dense(100, activation='relu')(composed_layer)
        composed_layer = Dropout(0.4)(composed_layer)
        composed_layer = Dense(100, activation='relu')(composed_layer)
        composed_layer = Dropout(0.4)(composed_layer)
        composed_layer = Dense(100, activation='relu')(composed_layer)
        composed_layer = Dropout(0.4)(composed_layer)
        composed_layer = Dense(100, activation='relu')(composed_layer)
        composed_layer = Dropout(0.4)(composed_layer)
        composed_layer = Dense(100, activation='relu')(composed_layer)
        composed_layer = Dropout(0.4)(composed_layer)
        output = Dense(22, activation='softmax')(composed_layer)

        self.gate = tf.keras.Model(inputs=input_layer, outputs=output)

        self.gate.compile(
            loss=Custom_Loss.comb,
            optimizer=Config.optimizer,
            metrics=[Custom_Metric.correlation, Custom_Metric.intersection]
        )
