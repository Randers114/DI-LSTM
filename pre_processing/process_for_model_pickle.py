import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

def retrieve_edge_data(file_name):
    # Get data from file
    dataset = pd.read_csv(file_name)
    # Set max length as the max length of longest edge to facilitate normalization, eval is needed for strings
    max_length = max(list(map(max, map(eval, list(dataset.iloc[:, 2])))))

    shape_dict = {}
    truths_dict = {}

    # Iterate all tuples to create the distributions and ground truth
    for row in tqdm(dataset.itertuples()):
        # Create time vector of [24]
        time = [1 if row[-1] == n else 0 for n in range(24)]

        assert(len(time) > 0)

        # Create RNN input
        r = [eval(row[2])[i] + [eval(row[3])[i]] for i in range(len(eval(row[2])))] 

        shape = np.asarray(r).shape[0]
        rnn_input = shape_dict.get(shape) if shape_dict.get(shape) is not None else []
        truths_input = truths_dict.get(shape) if truths_dict.get(shape) is not None else []

        rnn_input.append((r, time))
        truths_input.append(eval(row[-2]))

        shape_dict[shape] = rnn_input
        truths_dict[shape] = truths_input
    # Is used in another file for extracting length
    print("Length", max_length)
    return list(shape_dict.values()), list(truths_dict.values())

def write_to_file(edge_distributions, truths, file_name):
    pickle.dump((edge_distributions, truths), open(file_name, "wb"))

def load_from_file(file_name):
    return pickle.load(open(file_name, "rb"))

def read_encode_distribution_files(i):
    # f_name = "models/data/new_data/distributions" + str(input).zfill(2) + ".csv"
    f_name = "models/data/mdn_data/prediction_data/predicted_data/time/all_edges_and_paths_predicted.csv"

    distributions, labels = retrieve_edge_data(f_name)

    write_to_file(distributions, labels, "models/data/full_mdn/speed/data_pickle_" + str(i).zfill(2) + ".p")

if __name__ == "__main__":
    i = sys.argv[1]
    read_encode_distribution_files(i)
