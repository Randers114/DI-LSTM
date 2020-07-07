import tensorflow as tf
import tensorflow_probability as tfp
from helpers.helper_mdn import integral_gaussian

tfd = tfp.distributions

class Custom_Loss:
    dataset = []
    @staticmethod
    def bhattacharyya(y_truth, y_pred):
        _batch_size = 400
        c = tf.reduce_sum(tf.math.sqrt(y_truth * y_pred + 1e-10))

        return tf.math.sqrt(_batch_size - c)

    @staticmethod
    def correlation(y_truth, y_pred, _batch_size = 400):
        return 1 - Custom_Metric.correlation(y_truth, y_pred)

    @staticmethod
    def chi_square(y_truth, y_pred):
        res = tf.math.divide_no_nan((y_pred - y_truth) ** 2, y_pred)
        return tf.keras.backend.sum(res)
    
    @staticmethod
    def intersection(y_truth, y_pred, _batch_size = 400, granularity = 1):
        return 1 - (tf.keras.backend.sum(tf.keras.backend.minimum(y_pred, y_truth)) / (_batch_size * granularity))
    
    @staticmethod
    def kl_divergence(y_truth, y_pred, _batch_size = 400, granularity = 1):
        a = tf.keras.backend.log(tf.math.divide(y_truth  + 1e-09, y_pred + 1e-09))
        b = y_truth * a
        return tf.keras.backend.sum(b) / (_batch_size * granularity)

    @staticmethod
    def intersection_gaus(y_truth, y_pred, _batch_size = 400):
        granularity = 1
        y_truth, y_pred = integral_gaussian(y_truth, y_pred, granularity, _batch_size, Custom_Loss.dataset) 

        return Custom_Loss.intersection(y_truth, y_pred, _batch_size, granularity)

    @staticmethod
    def kl_divergence_gaus(y_truth, y_pred, _batch_size=400):
        granularity = 1
        y_truth, y_pred = integral_gaussian(y_truth, y_pred, granularity, _batch_size, Custom_Loss.dataset) 

        return Custom_Loss.kl_divergence(y_truth, y_pred, _batch_size, granularity)
        
    @staticmethod
    def comb_gaus(y_truth, y_pred, _batch_size=400):
        granularity = 1
        y_truth, y_pred = integral_gaussian(y_truth, y_pred, granularity, _batch_size, Custom_Loss.dataset)
        
        return Custom_Loss.comb(y_truth, y_pred, _batch_size, granularity) 

    @staticmethod
    def comb(y_truth, y_pred, _batch_size=400, granularity=1):
        return Custom_Loss.intersection(y_truth, y_pred, _batch_size, granularity) + Custom_Loss.kl_divergence(y_truth, y_pred, _batch_size, granularity)

class Custom_Metric:
    @staticmethod
    def bhattacharyya(y_truth, y_pred):
        _batch_size = 400
        return Custom_Loss.bhattacharyya(y_truth, y_pred)/_batch_size

    @staticmethod
    def correlation(y_truth, y_pred):
        t_hat = 1/22
        p_hat = 1/22
        
        a = tf.keras.backend.sum((y_pred - p_hat)*(y_truth - t_hat))
        b = tf.math.sqrt(tf.keras.backend.sum((y_pred - p_hat)**2)*tf.keras.backend.sum((y_truth - t_hat)**2))

        res = tf.math.divide_no_nan(a, b)

        return res
    @staticmethod
    def chi_square(y_truth, y_pred):
        _batch_size = 400
        return Custom_Loss.chi_square / _batch_size

    @staticmethod
    def intersection(y_truth, y_pred):
        _batch_size = 400
        return (tf.keras.backend.sum(tf.keras.backend.minimum(y_pred, y_truth)) / _batch_size)
