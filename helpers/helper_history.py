import pickle
import sys

import matplotlib.pyplot as plt

class CustomHistory:
    def __init__(self, training_loss, validation_loss):
        self.history = {}
        self.history["loss"] = training_loss
        self.history["val_loss"] = validation_loss

# Plot history over epochs to see if the model overfits
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def dump_history(history, name):
    pickle.dump(history, open(name, "wb"))

if __name__ == "__main__":
    path = "models/test_data/" + sys.argv[1] + "_history" + ".p"

    plot_history(pickle.load(open(path, "rb")))
