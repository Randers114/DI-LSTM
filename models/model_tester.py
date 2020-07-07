import sys

from helpers.helper_config import (Config, print_model_config, set_config,
                                   set_hyperparams)
from helpers.helper_controller import get_model_controller
from helpers.helper_find_validation import find_best_validation
from models.model_validation_output import (model_results_append,
                                            print_box_plots_path_lengths,
                                            print_dynamic_model_comparison,
                                            print_grouped_box_plots,
                                            print_model_validation)


def run_model(file_name):
    model_controller = get_model_controller()
    model_controller.run_model(file_name)
    print_model_validation(file_name)
    print_model_validation(file_name+"_gran")
    if Config.same_truth:
        print_model_validation(file_name+"_same_truth")
    

def get_metric_bounds(metric):
    if metric == 0:
        return (-0.03, 1.03)
    elif metric == 2:
        return (-0.03, 1.03)
    else:
        return (-0.03, 11)

if __name__ == "__main__":
    run_model(Config.name)

    # An example of how to print box plots from different models, "i" is the metric index. 
    # i = 0 is correlation
    # i = 2 is intersection
    # i = 4 is kl divergence
    # for i in [0, 2, 4]:
    #     print_grouped_box_plots(
    #         [
    #             "LSTM-TG", 
    #             "DI-LSTM-MLP-TG",
    #             "LSTM-SG",
    #             "DI-LSTM-MLP-SG"], 
    #         get_metric_bounds(i), 0, None, i, [30, 70, 115, 169])
