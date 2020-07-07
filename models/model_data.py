import math
import os
import pickle
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from helpers.helper_length_normalizer import remove_zero_lengths
from helpers.helper_enums import Dataset

class Data:
    def __init__(self, testing_percent, dataset):
        self.testing_percent = testing_percent
        self.dataset = dataset
        self.cnn_data = list()
        self.edge_distributions, self.truths, self.testing_distributions, self.testing_time_vectors, self.testing_truths = self.testing_split()
        self.edge_distributions, self.time_vectors, self.truths = self.flatten_and_split_time(self.edge_distributions, self.truths)

        if self.dataset == Dataset.Gaus_Speed or self.dataset == Dataset.Gaus_Time:
            self.normalize_testing(self.testing_distributions)

    def normalize_testing(self, testing_distributions):
        if self.dataset == Dataset.Gaus_Speed:
            MAX_MY = 45.994026 # Speed
            MAX_SIG = 11.446073 # Speed
        else:
            MAX_MY = 320.60922 # Time
            MAX_SIG = 160.50746 # Time

        func = lambda input_list, placement: input_list[placement*8:placement*8+8]
        
        temp_truth = list()
        print("Normalize Gaus testing")

        for shape in testing_distributions:
            temp_shape = list()

            for truth in shape:
                temp_distribution = list()

                for distribution in truth:
                    mus = [x/MAX_MY for x in func(distribution, 1)]
                    sigmas = [x/MAX_SIG for x in func(distribution, 2)]
                    pis = func(distribution, 0)
                    length = distribution[-1]

                    temp_distribution.append(list(pis) + list(mus) + list(sigmas) + [length])
                temp_shape.append(temp_distribution)
            temp_truth.append(temp_shape)


        self.testing_distributions = temp_truth

    def load_from_file(self):
        file_edge_distributions, file_truths = [], []
        file_names = []
        if self.dataset == Dataset.Normal:
            data_file_folder = "models/data/full" # SPEED
        elif self.dataset == Dataset.Gaus_Speed:
            data_file_folder = "models/data/full_mdn/speed" # GAUS SPEED
        elif self.dataset == Dataset.Gaus_Time:
            data_file_folder = "models/data/full_mdn/time" # GAUS TIME

        for file in os.listdir(data_file_folder):
            if file.endswith(".p"):
                file_names.append(os.path.join(data_file_folder, file))

        # Sort files according to path length
        file_names = sorted(file_names)
        

        for file_name in file_names:
            # Load distributions and truths from each file
            current_edge_distributions, current_truths = pickle.load(open(file_name, "rb"))

            file_edge_distributions.append(current_edge_distributions)
            file_truths.append(current_truths)


        return file_edge_distributions, file_truths
    
    def testing_split(self):
        # Reading data 
        edge_distributions, truths = self.load_from_file()

        index = []

        testing_distributions = []
        testing_truths = []
        testing_time_vectors = []
        
        
        # Making the testing split 
        for i, layer in enumerate(edge_distributions):
            testing_size = (math.floor(len(layer) * (self.testing_percent / 100)))
            temp = []
            temp2 = []
            temp3 = []

            # Finding random indexes to fill the validaiton until full
            while(len(index) != testing_size and len(layer) > 10):
                x = random.randint(0, len(layer)-1)
                
                index.append(x)
                distribution_time_tuple = layer.pop(x)
                temp.append(distribution_time_tuple[0])
                temp2.append(truths[i].pop(x))
                temp3.append(distribution_time_tuple[1])
            
            if (len(index) == 0 and len(layer[0][0]) != 169 and len(layer[0][0]) != 161 and len(layer[0][0]) != 151):
                distribution_time_tuple = layer.pop()
                temp.append(distribution_time_tuple[0])
                temp2.append(truths[i].pop())
                temp3.append(distribution_time_tuple[1])
            elif(len(layer[0][0]) == 169 or len(layer[0][0]) == 161 or len(layer[0][0]) == 151):
                while (len(layer) > 0):
                    distribution_time_tuple = layer.pop()
                    temp.append(distribution_time_tuple[0])
                    temp2.append(truths[i].pop())
                    temp3.append(distribution_time_tuple[1])

            testing_distributions.append(temp)
            testing_truths.append(temp2)
            testing_time_vectors.append(temp3)
            
            index = []
        
        # Making cnn or MLP testing data 
        temp_cnn = list()
        for i in range(len(edge_distributions[0])):
            temp_cnn.append([edge_distributions[0][i], truths[0][i]])
        self.cnn_data = temp_cnn

        return edge_distributions, truths, testing_distributions, testing_time_vectors, testing_truths

    def flatten_and_split_time(self, edge_distributions, truths):

        if self.dataset == Dataset.Gaus_Speed:
            MAX_MY = 45.994026 # Speed
            MAX_SIG = 11.446073 # Speed
        else:
            MAX_MY = 320.60922 # Time
            MAX_SIG = 160.50746 # Time
        flattened_distributions_with_time = [distribution for path_length_distributions in edge_distributions for distribution in path_length_distributions]
        flattened_truths = [truth for path_length_truths in truths for truth in path_length_truths]

        time_vectors = [distribution[1] for distribution in flattened_distributions_with_time]
        flattened_distributions = [distribution[0] for distribution in flattened_distributions_with_time]

        # NORMALIZE GAUSSIANS
        if self.dataset == Dataset.Gaus_Speed or self.dataset == Dataset.Gaus_Time:
            func = lambda input_list, placement: input_list[placement*8:placement*8+8]
            print("Normalize Gaus training")
            for i, edges_distributions in tqdm(enumerate(flattened_distributions)):
                temp = list()
                for distribution in edges_distributions:
                    mus = [x/MAX_MY for x in func(distribution, 1)]
                    sigmas = [x/MAX_SIG for x in func(distribution, 2)]
                    pis = func(distribution, 0)
                    length = distribution[-1]
                    
                    temp.append(list(pis) + list(mus) + list(sigmas) + [length])

                flattened_distributions[i] = temp

        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sequences=flattened_distributions, padding='post', value=0.0, dtype=np.float32)
        return padded_inputs, time_vectors, flattened_truths

    def get_data(self):
        return self.edge_distributions, self.time_vectors, self.truths, self.testing_distributions, self.testing_time_vectors, self.testing_truths


    def get_regression_data(self):
        new_shape = np.reshape(self.edge_distributions, (81620, 3197))

        temp = list()

        for shape in new_shape:
            temp.append(np.append(shape, np.zeros(690, np.float32)))

        return np.array(temp)

    def get_regression_test_distributions(self):
        temp = list()
        for e in self.testing_distributions:
            edge = np.array(e)
            flat = np.reshape(e, (edge.shape[0], edge.shape[1] * edge.shape[2]))

            temp.append(list(map(lambda x: np.append(x, [0.0] * (3887 - len(x))), flat)))

        return np.array(temp)

class DataTruthCompare:
    def __init__(self, testing_percent):
        self.testing_percent = testing_percent
        self.edge_distributions, self.truths, self.testing_distributions, self.testing_time_vectors, self.testing_truths, self.testing_truths_normal = self.testing_split()
        self.edge_distributions, self.time_vectors, self.truths = self.flatten_and_split_time(self.edge_distributions, self.truths)

        self.normalize_testing(self.testing_distributions)

    def normalize_testing(self, testing_distributions):
        MAX_MY = 45.994026 # Speed
        MAX_SIG = 11.446073 # Speed

        func = lambda input_list, placement: input_list[placement*8:placement*8+8]
        
        temp_truth = list()
        print("Normalize Gaus testing")

        for shape in testing_distributions:
            temp_shape = list()

            for truth in shape:
                temp_distribution = list()

                for distribution in truth:
                    mus = [x/MAX_MY for x in func(distribution, 1)]
                    sigmas = [x/MAX_SIG for x in func(distribution, 2)]
                    pis = func(distribution, 0)
                    length = distribution[-1]

                    temp_distribution.append(list(pis) + list(mus) + list(sigmas) + [length])
                temp_shape.append(temp_distribution)
            temp_truth.append(temp_shape)


        self.testing_distributions = temp_truth

    def load_from_file(self, data_file_folder):
        file_edge_distributions, file_truths = [], []
        file_names = []

        for file in os.listdir(data_file_folder):
            if file.endswith(".p"):
                file_names.append(os.path.join(data_file_folder, file))

        # Sort files according to path length
        file_names = sorted(file_names)
        

        for file_name in file_names:
            # Load distributions and truths from each file
            current_edge_distributions, current_truths = pickle.load(open(file_name, "rb"))

            file_edge_distributions.append(current_edge_distributions)
            file_truths.append(current_truths)


        return file_edge_distributions, file_truths
    
    def testing_split(self):
        # Reading data 
        edge_distributions_gaus, truths_gaus = self.load_from_file("models/data/full_mdn/speed")
        edge_distributions_normal, truths_normal = self.load_from_file("models/data/full")

        lengths_to_remove = [
            (411, 25), (6990, 5), (2722, 8), (13481, 3), 
            (19076, 2), (378, 27), (16077, 2), (607, 17), 
            (1469, 9), (1105, 10), (1183, 9), (7985, 2), 
            (1748, 6), (5821, 2), (472, 11), (2685, 2), 
            (124, 9), (6, 27), (20, 19), (2, 40), (186, 3), 
            (154, 3), (120, 2), (0, 3)]

        index = []

        testing_distributions = []
        testing_truths = []
        testing_truths_normal = []
        testing_time_vectors = []
        
        for i in range(len(truths_gaus)):
            for removeable in lengths_to_remove:
                if removeable[1] == len(edge_distributions_gaus[i][0][0]):
                    edge_distributions_gaus[i].pop(removeable[0])
                    truths_gaus[i].pop(removeable[0])


        # Making the testing split 
        for i, layer in enumerate(edge_distributions_gaus):
            testing_size = (math.floor(len(layer) * (self.testing_percent / 100)))
            temp = []
            temp2 = []
            temp3 = []
            temp4 = []

            # Finding random indexes to fill the validaiton until full
            while(len(index) != testing_size and len(layer) > 10):
                x = random.randint(0, len(layer)-1)
                
                index.append(x)
                distribution_time_tuple = layer.pop(x)
                temp.append(distribution_time_tuple[0])
                temp2.append(truths_gaus[i].pop(x))
                temp4.append(truths_normal[i].pop(x))
                temp3.append(distribution_time_tuple[1])
            
            if (len(index) == 0 and len(layer[0][0]) != 169 and len(layer[0][0]) != 161 and len(layer[0][0]) != 151):
                distribution_time_tuple = layer.pop()
                temp.append(distribution_time_tuple[0])
                temp2.append(truths_gaus[i].pop())
                temp4.append(truths_normal[i].pop())
                temp3.append(distribution_time_tuple[1])
            elif(len(layer[0][0]) == 169 or len(layer[0][0]) == 161 or len(layer[0][0]) == 151):
                while (len(layer) > 0):
                    distribution_time_tuple = layer.pop()
                    temp.append(distribution_time_tuple[0])
                    temp2.append(truths_gaus[i].pop())
                    temp4.append(truths_normal[i].pop())
                    temp3.append(distribution_time_tuple[1])

            testing_distributions.append(temp)
            testing_truths.append(temp2)
            testing_truths_normal.append(temp4)
            testing_time_vectors.append(temp3)
            
            index = []

        return edge_distributions_gaus, truths_gaus, testing_distributions, testing_time_vectors, testing_truths, testing_truths_normal

    def flatten_and_split_time(self, edge_distributions, truths):
        MAX_MY = 45.994026 # Speed
        MAX_SIG = 11.446073 # Speed

        flattened_distributions_with_time = [distribution for path_length_distributions in edge_distributions for distribution in path_length_distributions]
        flattened_truths = [truth for path_length_truths in truths for truth in path_length_truths]

        time_vectors = [distribution[1] for distribution in flattened_distributions_with_time]
        flattened_distributions = [distribution[0] for distribution in flattened_distributions_with_time]

        # NORMALIZE GAUSSIANS
        func = lambda input_list, placement: input_list[placement*8:placement*8+8]
        print("Normalize Gaus training")
        for i, edges_distributions in tqdm(enumerate(flattened_distributions)):
            temp = list()
            for distribution in edges_distributions:
                mus = [x/MAX_MY for x in func(distribution, 1)]
                sigmas = [x/MAX_SIG for x in func(distribution, 2)]
                pis = func(distribution, 0)
                length = distribution[-1]
                
                temp.append(list(pis) + list(mus) + list(sigmas) + [length])

            flattened_distributions[i] = temp

        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sequences=flattened_distributions, padding='post', value=0.0, dtype=np.float32)
        return padded_inputs, time_vectors, flattened_truths
