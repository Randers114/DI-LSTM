import math

import numpy as np

from helpers.helper_config import Config
from helpers.helper_enums import Dataset, RNNCell
from helpers.helper_mdn import integral_gaussian_numpy


def compareHist(left_hist, right_hist, method):
    def hat(hist):
        return (1.0 / len(hist)) * np.sum(hist)

    histogram_length = len(left_hist)

    if method == 0:
        top_expr = sum([(left_hist[i] - hat(left_hist)) * (right_hist[i] - hat(right_hist)) for i in range(histogram_length)])
        bottom_expr = math.sqrt(sum([(left_hist[i] - hat(left_hist))**2 for i in range(histogram_length)]) * sum([(right_hist[i] - hat(right_hist))**2 for i in range(histogram_length)]))

        return top_expr / bottom_expr

    elif method == 1:
        return 1
    elif method == 2:
        return sum([min(left_hist[i], right_hist[i]) for i in range(histogram_length)])
    
    elif method == 3:
        c_ = np.sum(np.sqrt(np.multiply(left_hist, right_hist)))

        h1_hat = hat(left_hist)
        h2_hat = hat(right_hist)
        b_ = 1.0 / np.sqrt(h1_hat * h2_hat * (histogram_length ** 2))
        a_ = 1.0
        if (c_ > 1):
            c_ = 1
        d_ = (b_ * c_)

        if (d_ > 1):
            d_ = 1

        return np.math.sqrt(a_ - d_)
    elif method == 4:
        h_len = len(right_hist)
        sum_ = 0

        for i in range(h_len):
            left_hist[i] += 1e-09
            right_hist[i] += 1e-09
            b = right_hist[i] * math.log(right_hist[i] / left_hist[i])
            sum_ += b
        return sum_
    else:
        raise NotImplementedError

def compare_truth(method, method_box_data, prediction, testing_truths, testing_index, method_gran=[], method_box_data_gran=[], run_gran=False, integrate_prediction=False):
    '''Compares the prediction with the corresponding labels,
    and adds the results to the method and method_box_data lists'''

    def run_comparison(prediction_data, truths_data, weighted, non_weighted):
        temp = [[], [], [], [], []]

        for i, y_pred in enumerate(prediction_data):
            truths = np.asarray(truths_data[i]).astype(np.float32, copy=False)

            for k in range(5):
                temp[k].append(compareHist(y_pred, truths, method=k))
        for k in range(5):
            weighted[k] += temp[k]
            non_weighted[k].append(temp[k])

    if run_gran:
        result = integral_gaussian_numpy(np.array(testing_truths[testing_index]), np.array(prediction), 1, len(prediction), Config.dataset, True)
        run_comparison(result[1], result[0], method_gran, method_box_data_gran)
        
        result = integral_gaussian_numpy(np.array(testing_truths[testing_index]), np.array(prediction), 1, len(prediction), Config.dataset)
        testing_truths[testing_index] = result[0]
        prediction = result[1]

    if integrate_prediction:
        prediction = integral_gaussian_numpy(np.array(prediction), np.array(prediction), 1, len(prediction), Config.dataset, True)[1]

    run_comparison(prediction, testing_truths[testing_index], method, method_box_data)
