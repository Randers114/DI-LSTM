import numpy as np

from helpers.helper_compare_hist import compare_truth
from helpers.helper_config import Config
from helpers.helper_length_normalizer import remove_zero_lengths
from models.model_validation_output import model_results_append

class ModelController:
    def __init__(self, model):
        self.model = model

    def initialize_testing_variables(self):
        # Holds the weighted results of the four metrics
        method = [[], [], [], [], []]
        # Holds the un-weighted results of the four metrics
        method_box_data = [[], [], [], [], []]

        # Removes all empty input/truth distributions and empty time vectors
        testing_truths = remove_zero_lengths(Config.data.testing_truths)
        testing_distributions = remove_zero_lengths(Config.data.testing_distributions)
        testing_time_vectors = remove_zero_lengths(Config.data.testing_time_vectors)

        return method, method_box_data, testing_distributions, testing_time_vectors, testing_truths

    # Test the results of the model and return collected results and individual for each length
    def test_model(self, test_file_name):
        method, method_box_data, testing_distributions, testing_time_vectors, testing_truths = self.initialize_testing_variables()

        for i, shape in enumerate(testing_distributions):
            prediction = np.array(self.model.predict([np.array(shape), np.array(testing_time_vectors[i])]))

            # Adds to the method data, with the comparehist
            compare_truth(method, method_box_data, prediction, testing_truths, testing_index=i)

        model_results_append(test_file_name, method, method_box_data)


    def run_model(self, test_file_name):
        self.model = self.model.compile_self()

        history = self.model.fit(
            [np.asarray(Config.data.edge_distributions), np.asarray(Config.data.time_vectors)], 
            np.asarray(Config.data.truths),
            batch_size=Config.batch_size,
            validation_split=Config.validation_split, 
            epochs=Config.epochs,
            shuffle=True)

        self.test_model(test_file_name)
