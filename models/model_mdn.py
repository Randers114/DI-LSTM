import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Input, Masking, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd

from helpers.helper_config import Config
# from plot_mixture import plot_mixture_custom
# from pre_processing.generate_mdn_data import get_n_mdn_data
from pre_processing.process_mdn_training_data import open_pickle

def nnelu(inputs):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))

def nll_loss(y, param_list):
    alpha, mu, sigma = slice_param_list(param_list)
    sigma += 1e-8
    alpha += 1e-8
    mu += 1e-8

    gaussian_mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma
        )
    )

    log_likelyhood = gaussian_mixture.log_prob(tf.transpose(y) + 1e-8) 
    
    return -tf.reduce_mean(log_likelyhood, axis=-1)

def load_dicts_from_pickle():
    file_name = "models/data/mdn_data/training/time/data_dict/mdn_pickle_training_data_dicts_"
    temp_dict = dict()
    
    for suffix in range(11):
        temp_dict.update(pickle.load(open(file_name + str(suffix) + ".p", "rb")))
    
    return temp_dict

def print_sample_of_prediction(x_test_dict, number_to_show):
    for index_of_mix, e in enumerate(x_test_dict[shape]):
        print("MU", list(map(lambda x: x, mus[index_of_mix])))
        print("________")
        print("SI", list(map(lambda x: x, sigs[index_of_mix])))
        print("________")
        print("PI", list(map(lambda x: x, pis[index_of_mix])))

        truth = np.sort(e, axis=None)

        print(truth)

        if index_of_mix == (number_to_show - 1):
            break

class CustomMDNModel():
    def __init__(self):
        self.model = self.init_model()
        print("Done")
    
    def init_model(self):
        input_layer = Input(shape=(None, 1), dtype='float32')
        masking_layer = Masking(mask_value=0., input_shape=(None, Config.mdn_input_size))(input_layer)
        composed_layer = LSTM(Config.mdn_mixes*25)(masking_layer)
        composed_layer = Dense(Config.mdn_hidden_size, activation='relu')(composed_layer)
        composed_layer = Dense(Config.mdn_hidden_size, activation='relu')(composed_layer)

        # Create the three vectors needed for gaussian mixture
        alphas = Dense(Config.mdn_mixes, activation='softmax', name='Alphas')(composed_layer)
        mus = Dense(Config.mdn_mixes, activation=None, name='Mus')(composed_layer)
        sigmas = Dense(Config.mdn_mixes, activation=nnelu, name='Sigmas')(composed_layer)
        output_layer = Concatenate(name='Output')([alphas, mus, sigmas])

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=nll_loss, optimizer=Adam())

        model.summary()

        return model

def build_run_predict(x_test_dict):
    def nan_protection(predictions_shapes, length):
        if str(predictions_shapes[length][0][0]) == 'nan':
            result_file = open("models/test_data/mdn_results.txt", "a+")

            result_file.write(get_result_string('nan', 'nan'))
            result_file.close()
            exit()
    def build(x_test_dict):
        x_train = np.array(x_test_dict[Config.mdn_input_size]).reshape(len(x_test_dict[Config.mdn_input_size]), Config.mdn_input_size, 1)

        y_train = x_train

        model = CustomMDNModel().model

        return model, x_train, y_train

    def run(model):
        print("Fitting the model")
        model.fit(x=x_train, y=y_train, batch_size=Config.mdn_batch_size, epochs=Config.mdn_epochs, validation_split=0.1)
    
    def predict(model, x_test_dict):
        predictions_shapes = dict()

        # model.save("mdn_model_time_trained.h5")
        # exit()

        # COMMENT IN FOR PREDICTION DATA INSTEAD OF TRAINING DATA (START)
        file_path = "models/data/mdn_data/prediction_data/index_data/"
        x_test_dict = pickle.load(open(file_path + "mdn_all_paths_time_data_dict.p", "rb"))
        # END

        # Make predictions from the model
        print("Making the Predictions")
        length_of_data = 0
        suffix = 1
        run_the_prediction_bool = False
        current_prediction = 223
        for length in tqdm.tqdm(x_test_dict):
            if(run_the_prediction_bool):
                test_data = np.array(x_test_dict.get(length)).reshape(len(x_test_dict.get(length)), length, 1)
                predictions_shapes[length] = (model.predict(test_data))
                length_of_data += 1
                # print(length_of_data)
            else:
                if length == current_prediction:
                    run_the_prediction_bool = True
            # Perhaps not relevant anymore
            #nan_protection(predictions_shapes, length)

            if length_of_data > 200:
                # COMMENT IN FOR PREDICTION DATA INSTEAD OF TRAINING DATA (START)
                file_path = "models/data/mdn_data/prediction_data/predicted_data/time/"
                file_name = file_path + "mdn_paths_time_prediction_" + str(suffix)
                pickle.dump(predictions_shapes, open(file_name + ".p", "wb"))

                predictions_shapes = dict()
                suffix += 1
                length_of_data = 0
                print(length)
                exit()

                # END
        # file_path = "models/data/mdn_data/prediction_data/predicted_data/"
        # file_name = file_path + "all_paths_prediction"
        # pickle.dump(predictions_shapes, open(file_name + ".p", "wb"))
        file_path = "models/data/mdn_data/prediction_data/predicted_data/time/"
        file_name = file_path + "mdn_paths_time_prediction_" + str(suffix)
        pickle.dump(predictions_shapes, open(file_name + ".p", "wb"))
        print("doneso")
        exit()
        
        return predictions_shapes

    # model, x_train, y_train = build(x_test_dict)

    # run(model)

    # return predict(model, x_test_dict)

    model = tf.keras.models.load_model('mdn_model_time_trained.h5')
    model.summary()

    return predict(model, x_test_dict)

def get_result_string(nll, mse):
    if nll == 'nan':
        return "nan,nan," + str(Config.mdn_epochs) + "," + str(Config.mdn_batch_size) + "," + str(Config.mdn_input_size) + "\n"
    else:
        return "{:1.3f},".format(nll) + "{:1.3f},".format(mse) + str(Config.mdn_epochs) + "," + str(Config.mdn_batch_size) + "," + str(Config.mdn_input_size) + "\n"

def run_mdn():
    x_test_dict = load_dicts_from_pickle()

    predictions_shapes = build_run_predict(x_test_dict)

    # COMMENT IN FOR PREDICTION DATA INSTEAD OF TRAINING DATA (START)
    # file_path = "models/data/mdn_data/prediction_data/index_data/"
    # x_test_dict = pickle.load(open(file_path + "mdn_all_edges_time_data_dict.p", "rb"))
    # END

    print("Summing up all he NLL MSE's")
    for shape in predictions_shapes:
        y_pred = predictions_shapes[shape]
        y_test = x_test_dict[shape]

        pis, mus, sigs = slice_param_list(y_pred)

        print_sample_of_prediction(x_test_dict, number_to_show=1) # Can be used to view the data
        
        nll_sum, mse_sum = list(), list()


        nll_temp, mse_temp = eval_mdn_model(pis, mus, sigs, np.array(y_test))
        nll_sum.append(nll_temp)
        mse_sum.append(mse_temp)

    nll = np.array(nll_sum).mean()
    mse = np.array(mse_sum).mean()

    print("MDN-NLL: {:1.3f}\n".format(nll))
    print("MDN-MSE: {:1.3f}".format(mse))
    
    result_file = open("models/test_data/mdn_results_time.txt", "a+")

    result_file.write(get_result_string(nll, mse))
    result_file.close()
    # Only used to visualize the plots 
    # for i in range(len(pis)):
    #     some_plotting(pis[i], mus[i], sigs[i]) # Two different methods of the same thing
    #     plot_mixture_custom(mus[i], pis[i], sigs[i]) # Two different methods of the same thing

def slice_param_list(param_vector):
    return [param_vector[:,i*Config.mdn_mixes:(i+1)*Config.mdn_mixes] for i in range(3)]

def eval_mdn_model(alpha_pred, mu_pred, sigma_pred, y_test):
    def nll_eval(y, alpha, mu, sigma):
        """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        """
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(
                loc=mu,       
                scale=sigma+1e-8))

        log_likelihood = gm.log_prob(tf.transpose(y))

        return -tf.reduce_mean(log_likelihood, axis=-1)

    collected_nll = nll_eval(np.array(y_test).astype(np.float32), alpha_pred, mu_pred, sigma_pred).numpy()

    nll = collected_nll.mean()

    mse = list()
    for i in tqdm.tqdm(range(len(y_test))):
        mse.append(tf.losses.mean_squared_error(np.multiply(alpha_pred[i], mu_pred[i]).sum(axis=-1), y_test[i]).numpy())

    return nll, np.array(mse).mean()

def some_plotting(alpha, mu, sigma):
    def remove_ax_window(ax):
        """
            Remove all axes and tick params in pyplot.
            Input: ax object.
        """
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)  
        ax.tick_params(axis=u'both', which=u'both',length=0)

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,       
            scale=sigma))

    x = np.linspace(0,44,44)
    pyx = gm.prob(x)
    # print(pyx)

    ax = plt.gca()

    ax.plot(x,pyx,alpha=1, color=sns.color_palette()[0], linewidth=2)

    ax.set_xlabel(r"y")
    ax.set_ylabel(r"$p(y|x=8)$")

    remove_ax_window(ax)

    plt.tight_layout()
    plt.show()

def dump_dictionary_to_files(dictionary):
    file_name = "models/data/mdn_data/training/time/data_dict/mdn_pickle_training_data_dicts_"
    suffix = 0
    temp = dict()
    i = 0

    for key in dictionary:
        temp[key] = dictionary[key]
        i += len(dictionary[key])

        if i > 40000:
            pickle.dump(temp, open(file_name + str(suffix) + ".p", "wb"))
            temp = dict()
            i = 0
            suffix += 1
    
    pickle.dump(temp, open(file_name + str(suffix) + ".p", "wb"))

def process_mdn_data_to_dict():
    """IF NEW DATA NEEDED"""
    samples = open_pickle() 

    x_test_dict = dict()
    for i, y in tqdm.tqdm(enumerate(samples)):
        x_test_dict[len(y)] = [*x_test_dict.get(len(y)), y] if x_test_dict.get(len(y)) is not None else [y]

    dump_dictionary_to_files(x_test_dict)

if __name__ == "__main__":
    tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})
    tf.keras.utils.get_custom_objects().update({'nll_loss': nll_loss})

    run_mdn()
