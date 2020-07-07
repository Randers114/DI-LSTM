import datetime

import numpy as np
import tensorflow as tf

from helpers.helper_compare_hist import compare_truth
from helpers.helper_config import Config
from helpers.helper_enums import TrainingLoop
from helpers.helper_generators import cnn_generator
from helpers.helper_length_normalizer import remove_zero_lengths
from models.model_controllers.model_controller import ModelController
from models.model_validation_output import model_results_append

class ModelControllerCNN(ModelController):
    def __init__(self, model):
        super().__init__(model)
        
        self.dataset = tf.data.Dataset.from_generator(cnn_generator(Config.data.cnn_data), output_types=(tf.float32, tf.float32))
        self.dataset = self.dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
        self.dataset = self.dataset.batch(Config.batch_size)

    def initialize_testing_variables(self):
        # Holds the weighted results of the four metrics
        method = [[], [], [], [], []]
        # Holds the un-weighted results of the four metrics
        method_box_data = [[], [], [], [], []]

        # Removes all empty input/truth distributions and empty time vectors
        testing_truths = remove_zero_lengths(Config.data.testing_truths)
        testing_distributions = remove_zero_lengths(Config.data.testing_distributions[0])

        return method, method_box_data, testing_distributions, testing_truths

    def print_batch_loop(self, batch_size, distributions_length, loop_index):
        print("                      ", end="\r")
        print("  ", loop_index * batch_size, "/", len(distributions_length), end="\r")

    def print_epoch_loop(self, time, epoch, loss_avg):
        print("Time: ", datetime.datetime.now() - time)
        print(f"Epoch {epoch + 1}: Loss: {loss_avg}:")

    def train_batches(self, loss, optimizer, epoch_loss_avg):
        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(model, inputs, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

        for i, (distributions, truths) in enumerate(self.dataset):
            if distributions.shape[0] < Config.batch_size:
                continue
            if Config.training_loop_type == TrainingLoop.CNN_NO_SPARSE:
                distributions = tf.reshape(distributions, shape=(400, 2, 23, 1))

            loss_value, grads = grad(self.model, distributions, truths)
            # Apply gradiants to the trainable variables
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            epoch_loss_avg(loss_value)

            self.print_batch_loop(Config.batch_size, Config.data.cnn_data, i)

    def test_model(self, test_file_name):
        method, method_box_data, testing_distributions, testing_truths = self.initialize_testing_variables()

        dataset = tf.data.Dataset.from_generator(cnn_generator(testing_distributions, include_truths=False), output_types=(tf.float32))
        dataset = dataset.batch(len(testing_distributions))
        prediction = []

        for distributions in dataset:
            if Config.training_loop_type == TrainingLoop.CNN_NO_SPARSE:
                distributions = tf.reshape(distributions, shape=(1267, 2, 23, 1))
            # Run the variant of the training loop based on training_loop_type
            prediction = np.array(self.model([distributions], training=False))

        compare_truth(method, method_box_data, prediction, testing_truths, testing_index=0)

        model_results_append(test_file_name, method, method_box_data)

    def run_model(self, test_file_name):
        def custom_training_loop():
            def loss(model, distributions, y_true, training, batch_size=Config.batch_size):
                y_pred = model(distributions, training=training)

                return Config.loss_function(y_true, y_pred, batch_size)

            optimizer = Config.optimizer

            for epoch in range(Config.epochs):
                time_at_start = datetime.datetime.now()
                epoch_loss_avg = tf.keras.metrics.Mean()

                # Train
                self.train_batches(loss, optimizer, epoch_loss_avg)

                self.print_epoch_loop(time_at_start, epoch, epoch_loss_avg.result())
                
        self.model = self.model.gate
        self.model.summary()
        
        custom_training_loop()

        self.test_model(test_file_name)
