import datetime
import math

import numpy
import pandas as pd
import tqdm
from helpers.helper_function_time import time_decorator

BUCKET_SIZE = 2
MAX_SPEED = 43
SPEED_BUCKETS = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44]

def calc_distribution_for_edge(data, distance, mdn_data):
    # Calculate the speed based on TravelTime and Distance
    edge_data = pd.DataFrame(columns=['Speed'])
    # edge_data['Speed'] = distance / data.iloc[:, 0] # SPEED
    edge_data['Speed'] = data.iloc[:, 0] # TIME

    if mdn_data:
        return list(edge_data.iloc[:, 0])
    else:
        return get_probabilities_from_edge(edge_data.iloc[:, 0])

# Returns the probalilty buckets for an edge.
def get_probability_buckets(edge_data):
    prob_dist_edge = []
    
    # For all the different buckets, find the probability distribution and add it to the 
    # prop_dist_edge vector
    for interval in SPEED_BUCKETS:
        # Find the probability by finding the number of rows in the bucket subset
        # and divide that with the total number of rows of the hour of day.
        num_of_durs_in_interval = len(edge_data[
            (edge_data > (interval - BUCKET_SIZE)) &
            (edge_data <= interval)])

        prob_dist_edge.append(round(
            num_of_durs_in_interval / len(edge_data),
            5))
    
    return prob_dist_edge

def get_probabilities_from_edge(edge_data):
    return get_probability_buckets(edge_data)

@time_decorator
def edge_preprocess(edge_dataset, edge_distribution_lookup, training_distributions_mdn, mdn_data, hour):
    saved_index = 0
    dataset_length = len(edge_dataset)

    for i, edge in enumerate(edge_dataset.itertuples()):
        if edge.sd_pair in edge_distribution_lookup:
            if edge_distribution_lookup.get(edge.sd_pair) == []:
                edge_data = edge_dataset[edge_dataset.sd_pair == edge.sd_pair]
                edge_distribution_lookup[edge.sd_pair] = calc_distribution_for_edge(edge_data.iloc[:, [6]], edge.Distance, mdn_data)
        elif edge.sd_pair not in training_distributions_mdn and mdn_data:
            edge_data = edge_dataset[edge_dataset.sd_pair == edge.sd_pair]
            training_distributions_mdn[edge.sd_pair] = calc_distribution_for_edge(edge_data.iloc[:, [6]], edge.Distance, mdn_data)
        else:
            if saved_index % 5000 == 0:
                print("Refused:", saved_index, "Accepted:", len(training_distributions_mdn), "Total:", i, "of", dataset_length, "Hour:", hour)
            saved_index += 1

def distribution_path_processing(path_dataset, edge_distribution_lookup, path_distribution_lookup, mdn_data, time_of_day):
    temp_dataframe = pd.DataFrame()
    
    for path in path_dataset.itertuples():
        path_durs_df = pd.DataFrame(eval(path.pathDurs))
        edge_list = []
        edge_distances = []

        # Finds the edge distributions that are part of the path.
        for edge in eval(path.edges):
            edge_distances.append(eval(edge)[1])
            edge_list.append((edge_distribution_lookup[eval(edge)[0]]))

        path_dist = []

        if path.edges not in path_distribution_lookup:
            path_dist = calc_distribution_for_edge(path_durs_df, path.pathDist, mdn_data)
            path_distribution_lookup[path.edges] = path_dist
        else:
            path_dist = path_distribution_lookup[path.edges]

        temp_dataframe = temp_dataframe.append([[edge_list, edge_distances,
                                                    path_dist,
                                                    time_of_day]])
    return temp_dataframe

def distribution_mdn_processing(path_dataset, edge_distribution_lookup, path_distribution_lookup, mdn_data, time_of_day, get_id):
    temp_dataframe = pd.DataFrame()

    for path in path_dataset.itertuples():
        path_durs_df = pd.DataFrame(eval(path.pathDurs))
        edge_list = []

        # Finds the edge distributions that are part of the path.
        for edge in eval(path.edges):
            edge_list.append(get_id(eval(edge)[0], time_of_day))

        path_dist = []

        if path.edges not in path_distribution_lookup:
            path_dist = calc_distribution_for_edge(path_durs_df, path.pathDist, mdn_data)
            path_distribution_lookup[path.edges] = path_dist
        else:
            path_dist = path_distribution_lookup[path.edges]

        temp_dataframe = temp_dataframe.append([[edge_list,
                                                    path_dist,
                                                    time_of_day]])
    return temp_dataframe

# Creates distribution for each edge and path in the input files.
def preprocess_data(edge_file, path_file, mdn_data, hour):
    def get_id(sd, hour):
        return str(sd) + "_" + str(hour)
        
    # Retrieve the edge and path data.
    edge_dataset = pd.read_csv(edge_file)
    edge_dataset['sd_pair'] = list(zip(edge_dataset['Source'], edge_dataset['Destination']))
    time_of_day = edge_dataset.iloc[1,7][11:13] # Returns the hour from the string element of column 7.
    path_dataset = pd.read_csv(path_file, sep=';')

    distribution_df = pd.DataFrame()
    training_distributions = pd.DataFrame()

    # Dictionaries are used for speedup if multiple of the same edge or path is part of the data.
    edge_distribution_lookup = {}
    path_distribution_lookup = {}
    training_distributions_mdn = {}
    length_lookup = {}

    for path in path_dataset.itertuples():
        for edge in eval(path.edges):
            edge_distribution_lookup[eval(edge)[0]] = []
            length_lookup[eval(edge)[0]] = eval(edge)[1]


    print('Starting Edge Preprocess', len(edge_dataset), "Hour:", hour)
    edge_preprocess(edge_dataset, edge_distribution_lookup, training_distributions_mdn, mdn_data, hour)

    edge_mdn_dataframe = pd.DataFrame()

    for key in edge_distribution_lookup:
        edge_mdn_dataframe = edge_mdn_dataframe.append([[get_id(key, time_of_day), edge_distribution_lookup[key], length_lookup[key]]], ignore_index=True)

    edge_mdn_dataframe.columns = ["id","speeds","length"]
    edge_mdn_dataframe.to_csv("models/data/mdn_data/prediction_data/index_data/time_data_edges_" + str(time_of_day) + ".csv", index=False)

    print('Starting Path Preprocess', len(path_dataset), "Hour:", hour)
    if(mdn_data):
        distribution_df = distribution_mdn_processing(path_dataset, edge_distribution_lookup, path_distribution_lookup, mdn_data, time_of_day, get_id)
    else:
        distribution_df = distribution_path_processing(path_dataset, edge_distribution_lookup, path_distribution_lookup, mdn_data, time_of_day)
    
    print("Starting training edges", len(training_distributions_mdn))
    for key in training_distributions_mdn:
        training_distributions = training_distributions.append([[training_distributions_mdn[key], time_of_day]])

    return distribution_df, training_distributions

if __name__ == '__main__':
    hour = 1
    is_mdn = True
    path = "models/data/new_data/"
    file_name = "distributions" if not is_mdn else "speed_data_for_mdn"

    result, training_result = preprocess_data(path + 'filteredEdges' + str(hour).zfill(2) + '.csv', path + 'paths' + str(hour).zfill(2) + '.csv', is_mdn, hour)

    result.to_csv(file_name + str(hour).zfill(2) + ".csv", header=False)
    if is_mdn and False:
        training_result.to_csv("training_mdn" + str(hour).zfill(2) + ".csv", header=False)

def run_preprocess_data(input_time, mdn_data=False):
    path = "models/data/new_data/"
    result, training_result = preprocess_data(path + 'filteredEdges' + str(input_time).zfill(2) + '.csv', path + 'paths' + str(input_time).zfill(2) + '.csv', mdn_data, input_time)
    file_name = "distributions" if not mdn_data else "models/data/mdn_data/prediction_data/index_data/time_data_for_mdn_paths_"

    result.to_csv(file_name + str(input_time).zfill(2) + '.csv', header=False)

    if mdn_data:
        training_result.to_csv("models/data/mdn_data/training/time/training_mdn" + str(input_time).zfill(2) + ".csv", header=False)

