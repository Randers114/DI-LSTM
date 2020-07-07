import cv2
import numpy as np

from helpers.helper_compare_hist import compare_truth
from helpers.helper_enums import Dataset
from helpers.helper_length_normalizer import remove_zero_lengths
from layers.layer_gate import Gate
from models.model_data import Data
from models.model_validation_output import (model_results_append,
                                            print_model_validation)

def initialize_testing_variables(data):
        # Holds the weighted results of the four metrics
        method = [[], [], [], [], []]
        # Holds the un-weighted results of the four metrics
        method_box_data = [[], [], [], [], []]

        # Removes all empty input/truth distributions and empty time vectors
        testing_truths = remove_zero_lengths(data.testing_truths)
        testing_distributions = remove_zero_lengths(data.testing_distributions)

        return method, method_box_data, testing_distributions, testing_truths

def run_only_gate(data, test_file_name):
    method, method_box_data, testing_distributions, testing_truths = initialize_testing_variables(data)

    # results = [[] for i in range(len(testing_distributions))]
    results = [[]]
    
    gate = Gate()

    for i, path_lengths in enumerate(testing_distributions):
        for distribution in path_lengths:
            if (len(distribution) != 2):
                break
            results[i].append(gate.combine_distributions_list(distribution)[:-1])

    compare_truth(method, method_box_data, results[0], testing_truths, 0)

    model_results_append(test_file_name, method, method_box_data)
    print_model_validation(test_file_name)

for _ in range(1):
    run_only_gate(Data(testing_percent=10), "name_of_model", Dataset.Normal) # Missing Dataset enum
