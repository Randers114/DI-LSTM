from helpers.helper_config import Config
from helpers.helper_enums import TrainingLoop

def generator(data_input, include_truths=True):
    '''Defines a generator that can yield either all (input and truths) or just input'''
    def truth_generator():
        for (distributions, time_vectors), truths in data_input:
            yield (distributions, time_vectors), truths
    def distributions_generator():
        for (distributions, time_vectors) in data_input:
            yield (distributions, time_vectors)

    return truth_generator if include_truths else distributions_generator

def cnn_generator(data_input, include_truths=True):
    def truth_generator():
        for distributions, truths in data_input:
            if(Config.training_loop_type == TrainingLoop.CNN):
                yield distributions[0][0] + distributions[0][1], truths
            else:
                yield [distributions[0][0], distributions[0][1]], truths
    def test_generator():
        for distribution in data_input:
            if(Config.training_loop_type == TrainingLoop.CNN):
                yield distribution[0] + distribution[1]
            else:
                yield [distribution[0], distribution[1]]
    return truth_generator if include_truths else test_generator

def mlp_generator(data_input, include_truths=True):
    def truth_generator():
        for distributions, truths in data_input:
            yield distributions[0][0] + distributions[0][1], truths
    def test_generator():
        for distribution in data_input:
            yield distribution[0] + distribution[1]
    return truth_generator if include_truths else test_generator