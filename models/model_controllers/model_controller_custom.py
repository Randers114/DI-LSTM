import datetime
import random

import numpy as np
import tensorflow as tf

from helpers.helper_compare_hist import compare_truth
from helpers.helper_config import Config
from helpers.helper_enums import Dataset, TrainingLoop
from helpers.helper_generators import generator
from helpers.helper_history import CustomHistory, dump_history
from helpers.helper_length_normalizer import remove_zero_lengths
from models.model_controllers.model_controller import ModelController
from models.model_validation_output import model_results_append

def slice_param_list(param_vector):
    return [param_vector[:,i*Config.mdn_mixes:(i+1)*Config.mdn_mixes] for i in range(3)]

def nnelu(inputs):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))

class ModelControllerCustomTraining(ModelController):
    def __init__(self, model):
        super().__init__(model)

        def split_for_validation():
            data = list(
                zip(
                    zip(
                        Config.data.edge_distributions,
                        Config.data.time_vectors),
                    Config.data.truths))
            index = 1
            data_len = len(data)
            temp_train, temp_time, temp_truth = list(), list(), list()
            validation_data = list()
            while index < data_len * 0.10:
                x = random.randint(0, data_len - index)
                x = data.pop(x)
                temp_train.append(x[0][0])
                temp_time.append(x[0][1])
                temp_truth.append(x[1])

                index += 1

            validation_data = [((temp_train, temp_time), temp_truth)]

            return data, validation_data


        self.data, self.val_data = split_for_validation()

        self.dataset = tf.data.Dataset.from_generator(generator(self.data), output_types=((tf.float32, tf.float32), tf.float32))
        self.dataset = self.dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
        self.dataset = self.dataset.batch(Config.batch_size)

        self.val_dataset = tf.data.Dataset.from_generator(generator(self.val_data), output_types=((tf.float32, tf.float32), tf.float32))

    def print_batch_loop(self, batch_size, distributions_length, loop_index):
        print("                      ", end="\r")
        print("  ", loop_index * batch_size, "/", len(distributions_length)*0.9, end="\r")

    def print_epoch_loop(self, time, epoch, loss_avg, loss_avg_val):
        print("Time: ", datetime.datetime.now() - time)
        print(f"Epoch {epoch + 1}: Loss: {loss_avg}: Val_Loss: {loss_avg_val}")
    
    def set_initial_cell_state(self, distributions, zeros, batch_size=Config.batch_size, path_length=Config.path_length_size):
        # Overwrite cell size to batch size
        self.model.layers[2].cell.current_batch_size = batch_size
        
        if Config.training_loop_type == TrainingLoop.RNN:
            self.model.layers[2].cell.initial_state = [zeros, tf.slice(distributions, [0, 0, 0], [batch_size, 1, Config.edge_size])]
        elif Config.training_loop_type == TrainingLoop.LSTM:
            self.model.layers[2].cell.initial_state = [zeros, zeros, tf.slice(distributions, [0, 0, 0], [batch_size, 1, Config.edge_size])]

        # Remove the first input from the model input as it is first state
        return tf.slice(distributions, [0, 1, 0], [batch_size, path_length, Config.edge_size])

    def train_batches(self, loss, optimizer, epoch_loss_avg):
        '''Loops over each batch in the dataset, and applies gradient-
            decent from the loss of each batch'''
        def grad(model, inputs, time, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(model, inputs, time, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)
        
        # Loops over each batch in the dataset
        for i, ((distributions, time_vectors), truths) in enumerate(self.dataset):
            if distributions.shape[0] < Config.batch_size:
                continue
            
            # Run the variant of the training loop based on training_loop_type
            if Config.training_loop_type is not TrainingLoop.Base:
                # Set initial_state for the recurrent cell to zeroes and the first input
                zeros = tf.zeros(shape=(Config.batch_size, Config.recurrent_layer_size))
                distributions = self.set_initial_cell_state(distributions, zeros)

            loss_value, grads = grad(self.model, distributions, time_vectors, truths)

            # Apply gradiants to the trainable variables
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            epoch_loss_avg(loss_value)

            self.print_batch_loop(Config.batch_size, Config.data.edge_distributions, i)

    def validate_epoch(self, loss, epoch_loss_avg, epoch_loss_avg_val, train_loss_results, val_loss_results, time_at_start, epoch):
        '''Validates an epoch with the loss over the current validation set, 
            adds loss to running loss for validation'''
        # Overwrite cell size to validation size
        validation_batch_size = len(self.val_data[0][1])

        # Initialize a zero tensor to be used in the first state
        zeros = tf.zeros(shape=(validation_batch_size, Config.recurrent_layer_size))
        for (x_val, t_val), y_val in self.val_dataset:
            # Run the variant of the training loop based on training_loop_type
            if Config.training_loop_type is not TrainingLoop.Base:
                # Set initial_state for the recurrent cell to zeroes and the first input
                zeros = tf.zeros(shape=(validation_batch_size, Config.recurrent_layer_size))
                x_val = self.set_initial_cell_state(x_val, zeros, validation_batch_size)
            
            # Predict and add the loss to the validation loss
            epoch_loss_avg_val(loss(self.model,
                                    np.asarray(x_val),
                                    np.array(t_val, dtype=np.float32),
                                    np.asarray(y_val),
                                    training=False,
                                    batch_size=validation_batch_size))
        train_loss_results.append(epoch_loss_avg.result())
        val_loss_results.append(epoch_loss_avg_val.result())
        
        self.print_epoch_loop(time_at_start, epoch, epoch_loss_avg.result(), epoch_loss_avg_val.result())

    def test_model(self, test_file_name):
        method, method_box_data, testing_distributions, testing_time_vectors, testing_truths = self.initialize_testing_variables()

        # Holds the weighted results of the four metrics
        method_gran = [[], [], [], [], []]
        # Holds the un-weighted results of the four metrics
        method_box_data_gran = [[], [], [], [], []]

        if Config.same_truth:
            # Holds the weighted results of the four metrics
            method_same_truth = [[], [], [], [], []]
            # Holds the un-weighted results of the four metrics
            method_box_data_same_truth = [[], [], [], [], []]

            testing_truths_normal = remove_zero_lengths(Config.data.testing_truths_normal)

        for i, shape in enumerate(testing_distributions):
            dataset = tf.data.Dataset.from_generator(generator([(shape, testing_time_vectors[i])], include_truths=False), output_types=(tf.float32, tf.float32))
            prediction = []

            for distributions, time_vectors in dataset:
                # Run the variant of the training loop based on training_loop_type
                if Config.training_loop_type is not TrainingLoop.Base:
                    # Set initial_state for the recurrent cell to zeroes and the first input
                    zeros = tf.zeros(shape=(len(shape), Config.recurrent_layer_size))
                    distributions = self.set_initial_cell_state(distributions, zeros, len(shape), len(shape[0]) - 1)
                
                prediction = np.array(self.model([distributions, time_vectors], training=False))

            if (Config.dataset == Dataset.Gaus_Speed or Config.dataset == Dataset.Gaus_Time):
                compare_truth(method, method_box_data, prediction, testing_truths, i, method_gran, method_box_data_gran, True)

                if Config.same_truth:
                    compare_truth(method_same_truth, method_box_data_same_truth, prediction, testing_truths_normal, i, integrate_prediction=True)


            else:
                compare_truth(method, method_box_data, prediction, testing_truths, testing_index=i)

        print("Printing results")
        model_results_append(test_file_name, method, method_box_data)

        if (Config.dataset == Dataset.Gaus_Speed or Config.dataset == Dataset.Gaus_Time):
            model_results_append(test_file_name+"_gran", method_gran, method_box_data_gran)

            if Config.same_truth:
                model_results_append(test_file_name+"_same_truth", method_same_truth, method_box_data_same_truth)



    def run_model(self, test_file_name):
        def custom_training_loop():
            def loss(model, distributions, time, y_true, training, batch_size=Config.batch_size, epoch_number=0):
                y_pred = model([distributions, time], training=training)

                return Config.loss_function(y_true, y_pred, batch_size)

            train_loss_results = []
            val_loss_results = []
            optimizer = Config.optimizer

            for epoch in range(Config.epochs):
                time_at_start = datetime.datetime.now()
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_loss_avg_val = tf.keras.metrics.Mean()

                # Train
                self.train_batches(loss, optimizer, epoch_loss_avg)

                # Validate
                self.validate_epoch(
                    loss,
                    epoch_loss_avg,
                    epoch_loss_avg_val,
                    train_loss_results,
                    val_loss_results,
                    time_at_start,
                    epoch)
                
            return CustomHistory(train_loss_results, val_loss_results)
            
        self.model = self.model.initialize_self()

        history = custom_training_loop()

        dump_history(history, "models/test_data/" + Config.name + "_history.p")

        self.test_model(test_file_name)
