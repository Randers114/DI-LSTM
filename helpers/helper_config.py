from tensorflow.keras.optimizers import Adam

from helpers.helper_enums import Gates, RNNCell, TrainingLoop, Dataset
from models.model_data import Data, DataTruthCompare
from models.model_loss import Custom_Loss

def set_config(
    name = 'model_test',
    controller_config = None,
    novel_config = None,
    gate_config = None,
    mdn_config = None
    ):
    """ Set_config sets the configuration file to whatever is used as input.
    It takes a name, and four lists, controller_config, novel_config, gate_config, and mdn_config, as its inputs.

    Lists expects the following:\n
        Controller expects: training loop type, batch size, and epocs
        Novel expects: recurrent cell, recurrent layer size, recurrent dropout, dense layer size, dropout, loss function, edge size, output size, feature input size, and path length size
        Gate expects: gate type, gate layer size
        Mdn expects: input size, hiddel layer size, output size, and number of mixes

    The following inputs all expects a string:\n
        Training loop type can be; 'LSTM', 'RNN', 'Base', 'CNN', 'CNN-NO-SPARSE'
        Recurrent cell can be; 'NovelRNN', 'NovelLSTM', 'OriginalRNN', 'OringialLSTM'
        Loss function can be; 'Intersetion', 'Correlation', 'KLD'
        Gate type can be; 'MLP', 'CNN', 'Math'


    """
    
    def set_controller(
        training_loop_type = 'LSTM',
        batch_size = 400,
        epochs = 3
        ):
        def get_training_loop(training_loop_type):
            
            if training_loop_type == 'LSTM':
                return TrainingLoop.LSTM
            elif training_loop_type == 'RNN':
                return TrainingLoop.RNN
            elif training_loop_type == 'Base':
                return TrainingLoop.Base
            elif training_loop_type == 'CNN':
                return TrainingLoop.CNN
            elif training_loop_type == 'CNN-NO-SPARSE':
                return TrainingLoop.CNN_NO_SPARSE
            else:
                raise NotImplementedError()

        Config.batch_size = batch_size
        Config.epochs = epochs
        Config.training_loop_type = get_training_loop(training_loop_type)

    def set_novel(
        recurrent_cell = 'NovelRNN',
        recurrent_layer_size = 300,
        recurrent_dropout_rate = 0.2, 
        dense_layer_sizes = [100, 100],
        dropout_rate = 0.3,
        loss_function = 'Intersection',
        edge_size = 23,
        output_size = 22,
        feature_input_size = 24
        # path_length_size = 98
        ):
        def get_recurrent_cell(recurrent_cell):
            if recurrent_cell == 'NovelLSTM':
                return RNNCell.LSTM_Cell
            elif recurrent_cell == 'NovelRNN':
                return RNNCell.RNN_Cell
            elif recurrent_cell == 'OriginalLSTM':
                return RNNCell.Original_LSTM_Cell
            elif recurrent_cell == 'OriginalRNN':
                return RNNCell.Original_RNN_Cell
            else:
                raise NotImplementedError()

        def get_loss_func(loss_function):
            if loss_function == 'Intersection':
                return Custom_Loss.intersection
            elif loss_function == 'Correlation':
                return Custom_Loss.correlation
            elif loss_function == 'KLD':
                return Custom_Loss.kl_divergence
            else:
                raise NotImplementedError()

        Config.recurrent_cell = get_recurrent_cell(recurrent_cell)
        Config.recurrent_layer_size = recurrent_layer_size
        Config.recurrent_dropout_rate = recurrent_dropout_rate
        Config.dense_layer_sizes = get_dense_sizes(dense_layer_sizes)
        Config.dropout_rate = dropout_rate
        Config.loss_function = get_loss_func(loss_function)
        Config.edge_size = edge_size
        Config.output_size = output_size
        Config.feature_input_size = feature_input_size
        # Config.path_length_size = path_length_size
            
    def set_gate(
        gate_type = 'MLP',
        gate_layer_size = 150
        ):
        Config.gate = get_gate_type(gate_type)
        Config.gate_layer_size = gate_layer_size

    def set_mdn(
        mdn_input_size = 9439,
        mdn_hidden_size = 15,
        mdn_output_size = 1,
        mdn_mixes = 2
        ):
        Config.mdn_input_size = mdn_input_size
        Config.mdn_hidden_size = mdn_hidden_size
        Config.mdn_output_size = mdn_output_size
        Config.mdn_mixes = mdn_mixes

    Config.name = name
    if controller_config is not None:
        if len(controller_config) <= 3:
            set_controller(*controller_config)
        else:
            raise ValueError('Controller Config too long')
    
    if novel_config is not None:
        if len(novel_config) <= 10:
            set_novel(*novel_config)
        else:
            raise ValueError('Novel Model Config too long')
    
    if gate_config is not None:
        if len(gate_config) <= 2:
            set_gate(*gate_config)
        else:
            raise ValueError('Model Gate Config too long')
    
    if gate_config is not None:
        if len(mdn_config) <= 4:
            set_mdn(*mdn_config)
        else:
            raise ValueError('Mdn Config too long')

def get_gate_type(gate_type):
    if gate_type == 'MLP':
        return Gates.MLPGate
    elif gate_type == 'CNN':
        return Gates.CNNGate
    elif gate_type == 'Math':
        return Gates.MathGate
    else:
        raise NotImplementedError()

def get_dense_sizes(dense_layer_sizes):
    a = dense_layer_sizes.split(':')
    return list(map(int, a))

def set_hyperparams(
    name,
    epochs,
    batch_size,
    recurrent_layer_size, 
    recurrent_dropout_rate,
    dense_layer_sizes,
    dropout_rate,
    ):
    """This function sets the hyperparamets of the model
    
    Arguments:\n
        name {string} -- The file name of the model 
        epochs {integer} -- The number of epocs the model should run
        batch_size {integer} -- The batch_size of the model
        recurrent_layer_size {integer} -- The layer size of the recurrent layer in the model
        recurrent_dropout_rate {integer} -- The dropout rate of the recurrent layer
        dense_layer_sizes {string} -- The layer sizes of the dense layers, should be ints seperated with : - 100:100:100
        dropout_rate {integer} -- The dropout rate of the dropout layers
        gate_type {string} -- The type of gate in the recurrent layer. 'MLP', 'CNN', or 'Math'
    """

    Config.name = name
    Config.epochs = int(epochs)
    Config.batch_size = int(batch_size)
    Config.recurrent_layer_size = int(recurrent_layer_size)
    Config.recurrent_dropout_rate = float(recurrent_dropout_rate)
    Config.dense_layer_sizes = get_dense_sizes(dense_layer_sizes)
    Config.dropout_rate = float(dropout_rate)

def print_model_config():
    '''Prints all the variables on the config that are changable from the set_config function'''
    print('Model name: ', Config.name)
    print('-----------------------')
    
    print("Controller Config:")
    print('Batch size:           ', Config.batch_size)
    print('Epocs:                ', Config.epochs)
    print('Training loop:        ', Config.training_loop_type)
    print('-----------------------')

    print("Novel Model Config")
    print('Recurrent layer size: ', Config.recurrent_layer_size)
    print('Recurrent droput:     ', Config.recurrent_dropout_rate)
    print('Dense layer size:     ', Config.dense_layer_sizes)
    print('Dropout:              ', Config.dropout_rate)
    print('Cell:                 ', Config.recurrent_cell)
    print('Loss:                 ', Config.loss_function)
    print('Edge size:            ', Config.edge_size)
    print('Output size:          ', Config.output_size)
    print('Feature input size:   ', Config.feature_input_size)
    # print('Path length size:     ', Config.path_length_size)
    print('-----------------------')

    print('Gate Config')
    print('Gate:                 ', Config.gate)
    print('Gate layer size:      ', Config.gate_layer_size)
    print('-----------------------')

    print("MDN Config")
    print('MDN input size:       ', Config.mdn_input_size)
    print('MDN hidden size:      ', Config.mdn_hidden_size)
    print('MDN output size:      ', Config.mdn_output_size)
    print('MDN mixes:            ', Config.mdn_mixes)
    print('_______________________')
        
class Config:
    # Test Name
    name = "DI-LSTM-SG-previous_conf_3"

    # DATASET
    dataset = Dataset.Gaus_Speed
    Custom_Loss.dataset = dataset
    same_truth = False

    # Controller Config
    batch_size = 20
    epochs = 2
    if same_truth:
        data = DataTruthCompare(10)
    else:
        data = Data(10, dataset)
    validation_split = 0.1
    training_loop_type = TrainingLoop.LSTM

    # Model Config
    recurrent_layer_size = 50
    dense_layer_sizes = [100, 100]
    recurrent_dropout_rate = 0.2
    dropout_rate = 0.3
    recurrent_cell = RNNCell.MDN_LSTM_CELL
    optimizer = Adam()
    loss_function = Custom_Loss.comb_gaus

    # Input/Output sizes for model
    edge_size = 25
    output_size = 24
    feature_input_size = 24
    path_length_size = 138

    # Custom cell config
    gate = Gates.MLPGate
    gate_layer_size = 50
    mlp_hidden_layer_numbers = 3
    mlp_dropout = 0.3

    # MDN config
    mdn_input_size = 20
    mdn_hidden_size = 15
    mdn_output_size = 1
    mdn_mixes = 8
    mdn_batch_size = 50
    mdn_epochs = 250