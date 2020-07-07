import enum

class Gates(enum.Enum):
    MathGate = 1
    MLPGate = 2
    CNNGate = 3

class TrainingLoop(enum.Enum):
    Base = 1
    RNN = 2
    LSTM = 3
    CNN = 4
    CNN_NO_SPARSE = 5
    BaseFit = 6
    MLP = 7

class RNNCell(enum.Enum):
    RNN_Cell = 1
    Original_RNN_Cell = 2
    LSTM_Cell = 3
    Original_LSTM_Cell = 4
    MDN_LSTM_CELL = 5

class ModelType(enum.Enum):
    Novel = 1
    CNN_MLP = 2

class Dataset(enum.Enum):
    Normal = 1
    Gaus_Speed = 2
    Gaus_Time = 3
