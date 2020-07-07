import tensorflow as tf

class LengthNormalizer():
    def __init__(self):
        self.iteration = 1

    def normalize(self, tensor_one, tensor_two):
        tensor_one = tensor_one / self.iteration
        tensor_two = tensor_two / self.iteration
        self.iteration += 1

        return tensor_one, tensor_two

def remove_zero_lengths(data):
    '''Removes all entries that does not have any data, ex. if there are
    no paths with length 66 the entry is removed'''

    return [entry for entry in data if len(entry) > 0]