import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from helpers.helper_enums import Dataset
tfd = tfp.distributions

def integral_gaussian(y_truth, y_pred, granularity, batch_size, dataset):
    """Integrates over y_truth, and y_pred GMM's to create histograms that we can use to evaluate the models with the same 
    metrics. This function uses tensors."""
    def loc(tensor):
        return tf.reshape(tf.slice(tensor, [0,1,0], [batch_size,1,8]), (batch_size,8,))

    def scale(tensor):
        return tf.reshape(tf.slice(tensor, [0,2,0], [batch_size,1,8]), (batch_size,8,))+1e-9

    def weight(tensor):
        return tf.reshape(tf.slice(tensor, [0,0,0], [batch_size,1,8]), (batch_size,8,))

    def get_observations(normal, weight_tensor):
        return tf.map_fn(lambda x: tf.map_fn(tf.keras.backend.sum, x), normal.prob(gran_list)*weight_tensor)

    if dataset == Dataset.Gaus_Speed:
        gran_list = [[[i/granularity for _ in range(8)] for _ in range(batch_size)] for i in range(44*granularity)] # Speed
    else:
        gran_list = [[[i/granularity for _ in range(8)] for _ in range(batch_size)] for i in range(350*granularity)] # Time

    y_truth = tf.reshape(tf.constant(y_truth), (batch_size, 3, 8))
    y_pred = tf.reshape(tf.constant(y_pred), (batch_size, 3, 8))

    prediction_observation = get_observations(tfd.Normal(loc=loc(y_pred), scale=scale(y_pred)), weight(y_pred))
    truth_observation = get_observations(tfd.Normal(loc=loc(y_truth), scale=scale(y_truth)), weight(y_truth))
    return truth_observation, prediction_observation

def integral_gaussian_numpy(y_truth, y_pred, granularity, batch_size, dataset, compare_gran=False):
    """Integrates over y_truth, and y_pred GMM's to create histograms that we can use to evaluate the models with the same 
    metrics. This function uses numpy arrays."""
    def loc(array):
        return array[:len(array), 8:16]
    def scale(array):
        return array[:len(array), 16:24]+1e-9
    def weight(array):
        return array[:len(array), 0:8]

    if dataset == Dataset.Gaus_Speed:
        gran_list = [[[i/granularity for _ in range(8)] for _ in range(batch_size)] for i in range(44*granularity)] # Speed
    else:
        gran_list = [[[i/granularity for _ in range(8)] for _ in range(batch_size)] for i in range(350*granularity)] # Time

    get_normal_distribution = lambda x: scipy.stats.norm(loc(x), scale(x)).pdf(gran_list)*weight(x)

    get_observations = lambda x: np.array(list(map(lambda y: list(map(sum, y)), x)))

    y_truth = get_observations(get_normal_distribution(y_truth)).T
    y_pred = get_observations(get_normal_distribution(y_pred)).T


    if compare_gran:
        combine_buckets = lambda x: [x[i] + x[i+1] for i in range(0, len(x), 2)]

        y_truth = np.array(list(map(combine_buckets, y_truth)))
        y_pred = np.array(list(map(combine_buckets, y_pred)))

    return y_truth, y_pred
