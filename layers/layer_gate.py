import tensorflow as tf
import datetime


class Gate():
    def __init__(self):
        self.length_to_normalize = 1


    def combine_distributions(self, distribution_left, distribution_right):
        running_length = distribution_left[-1]
        input_length = distribution_right[-1]

        new_distribution = []

        for i in range(len(distribution_right) - 1):
            new_bucket_value = round(
                (
                    (
                        running_length / 
                        self.length_to_normalize) * 
                    distribution_left[i] + 
                    (
                        input_length / 
                        self.length_to_normalize) * 
                    distribution_right[i]) / 
                    (
                        (
                            (running_length + input_length) / 
                            self.length_to_normalize)), 
                3)

            new_distribution.append(new_bucket_value)

        new_distribution.append(round(running_length + input_length, 7))
        self.length_to_normalize += 1

        return new_distribution

    def combine_distributions_list(self, distributions):
        temp = distributions[0]

        for d in distributions[1:]:
            temp = self.combine_distributions(temp, d)

        return temp

    def combine_tensors(self, left, right, left_length, right_length):
        temp = [0. for _ in range(left.shape[1])]
        left_length = left_length / self.length_to_normalize
        right_length = right_length / self.length_to_normalize

        new_distribution = tf.constant([temp])
        if(left.shape[0] is None):
            return tf.slice(left, [0, 0], [0, left.shape[1] - 1])

        for i in range(left.shape[0]):
            running_length = left_length[i]
            input_length = right_length[i]

            new_distribution = tf.concat(
                [
                    new_distribution,
                    my_tf_round(
                        (
                            (running_length) * tf.slice(left, [i, 0], [1, left.shape[1]]) +
                            (input_length) * tf.slice(right, [i, 0], [1, left.shape[1]]))
                        / (running_length + input_length), 3)], 0)

        self.length_to_normalize += 1

        return tf.slice(new_distribution, [1, 0], [left.shape[0], left.shape[1] - 1])

    def new_combine_tensors(self, left, right, left_length, right_length):
        left = tf.multiply(left, tf.reshape(left_length, (-1, 1)))
        right = tf.multiply(right, tf.reshape(right_length, (-1, 1)))

        res = my_tf_round((left + right) / (left_length + right_length), 3)

        return tf.slice(res, [0, 0], [left.shape[0], left.shape[1] - 1])

def my_tf_round(tensor, decimals=0):
    multiplier = tf.constant(10**decimals, dtype=tensor.dtype)
    return tf.round(tensor * multiplier) / multiplier
