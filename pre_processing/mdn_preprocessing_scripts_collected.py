import pandas as pd
import pickle
from tqdm import tqdm
from helpers.helper_config import Config
from helpers.helper_plot_mixture import plot_mixture_custom
import numpy as np

def find_best_mdn_conf():
    dataframe = pd.read_csv("models/test_data/mdn_results_speed.txt")

    dataframe = dataframe.sort_values(by=['NLL']).head(15).sort_values(by=["MSE"])

    print(dataframe)

def dump_dictionary_to_files(file_path, dictionary, name):
    file_name = file_path + name
    pickle.dump(dictionary, open(file_name + ".p", "wb"))

def get_all_edges_data(file_path):
    file_name = file_path + "all_edges_time.csv"

    dataframe = pd.read_csv(file_name)

    data = list(dataframe.speeds.apply(eval))

    return data

def get_all_paths_data(file_path):
    file_name = file_path + "all_paths_time.csv"

    dataframe = pd.read_csv(file_name, index_col=0)

    data = list(dataframe.speeds.apply(eval))

    return data

def process_mdn_data_to_dict(file_path):
    """IF NEW DATA NEEDED"""
    samples = get_all_edges_data(file_path) # FOR EDGES
    # samples = get_all_paths_data(file_path) # FOR PATHS

    x_test_dict = dict()
    for y in tqdm(samples):
        x_test_dict[len(y)] = [*x_test_dict.get(len(y)), y] if x_test_dict.get(len(y)) is not None else [y]

    print(len(x_test_dict))
    dump_dictionary_to_files(file_path, x_test_dict, "mdn_all_edges_time_data_dict") # FOR EDGES
    # dump_dictionary_to_files(file_path, x_test_dict, "mdn_all_paths_time_data_dict") # FOR PATHS

def sort_df(df):
    sorted_index = df.speeds.apply(eval).apply(len).sort_values().index
    
    df_sorted = df.reindex(sorted_index)
    return df_sorted.reset_index(drop=True)

def combine_mdn_predict_edge_data():
    def get_file_name(file_path, hour):
        return file_path + "time_data_edges_" + str(hour).zfill(2) + ".csv"
    file_path = "models/data/mdn_data/prediction_data/index_data/"
    dataframe = pd.read_csv(get_file_name(file_path, 0))

    for i in range(1, 24):
        dataframe = dataframe.append(pd.read_csv(get_file_name(file_path, i)))

    dataframe = dataframe.reset_index(drop=True)
    dataframe = sort_df(dataframe)
    dataframe.to_csv(file_path + "all_edges_time.csv")

def combine_mdn_predict_path_data():
    def get_file_name(file_path, hour):
        return file_path + "time_data_for_mdn_paths_" + str(hour).zfill(2) + ".csv"

    file_path_index = "models/data/mdn_data/prediction_data/index_data/"

    dataframe = pd.read_csv(get_file_name(file_path_index, 0), header=None, index_col=0)

    for i in range(1, 24):
        dataframe = dataframe.append(pd.read_csv(get_file_name(file_path_index, i), header=None, index_col=0))

    dataframe = dataframe.reset_index(drop=True)
    dataframe.columns = ["edges", "speeds", "time"]
    dataframe = sort_df(dataframe)
    dataframe.to_csv(file_path_index + "all_paths_time.csv")

def test_csv_vs_pickle(file_path):
    csv_all_edges = pd.read_csv(file_path + "all_edges_time.csv")
    
    pickle_all_edges = pickle.load(open(file_path + "mdn_all_edges_time_data_dict.p", "rb"))

    print(csv_all_edges.speeds[12352])
    # key = list(pickle_all_edges.keys())[0]
    print(pickle_all_edges[6][1000])

def slice_param_list(param_vector):
    return [param_vector[:,i*Config.mdn_mixes:(i+1)*Config.mdn_mixes] for i in range(3)]

def combine_mdn_predicted_data(file_path_predicted, file_path_index):
    file_name = file_path_predicted + "all_paths_prediction.p"
    collected = pickle.load(open(file_name, "rb"))

    csv_original = pd.read_csv(file_path_index + "all_paths.csv", index_col=0)

    pis, mus, sigs = list(), list(), list()

    for key in collected:
        temp_p, temp_m, temp_s = slice_param_list(collected[key])

        pis += list(temp_p)
        mus += list(temp_m)
        sigs += list(temp_s)

    pis = [list(map(lambda x: round(x, 3), y)) for y in pis]

    print("length", len(pis))

    temp_dataframe = pd.DataFrame(columns=["edges", "speeds", "time"])

    for number in tqdm(range(len(csv_original))):
        zipped = list(zip(mus[number], sigs[number], pis[number]))
        
        temp_data = csv_original.iloc[number,:]
        temp_data.speeds = str(zipped)
        temp_dataframe = temp_dataframe.append(temp_data, ignore_index=True)

    temp_dataframe.to_csv(file_path_predicted + "all_paths_predicted.csv")

def test_combine_mdn_predicted_data(file_path_index, file_path_predicted):
    file_predicted = file_path_predicted + "all_edges_predicted.csv"
    file_index = file_path_index + "all_edges.csv"

    predicted = pd.read_csv(file_predicted, index_col=0)
    index = pd.read_csv(file_index, index_col=0)
    number = 120000
    row = index.iloc[number,:].speeds

    print(
        sorted(
            list(map(lambda x: round(x, 1),
                eval(
                    row
                    )
                ))
            )
        )
    print("______________________")
    print("______________________")
    print("______________________")
    print(list(eval(predicted.iloc[number,:].speeds)))

def flatten_gaussians(gaus_list):
    return [x[i] for x in gaus_list for i in range(len(x))]

def sort_gaussians(gaus_list):
    return sorted(gaus_list, key=lambda x: x[0])

def replace_edge_ids_with_data(file_path_predicted):
    csv_edges = pd.read_csv(file_path_predicted + "all_edges_predicted.csv", index_col=0)
    csv_paths = pd.read_csv(file_path_predicted + "all_paths_predicted.csv", index_col=0)

    edge_lookup = dict()

    for edge_tuple in tqdm(csv_edges.itertuples()):
        edge_id = edge_tuple[1]
        edge_prediction = eval(edge_tuple[2])
        edge_length = edge_tuple[3]

        edge_lookup[edge_id] = (edge_prediction, edge_length)

    csv_paths.insert(loc=1, column='lengths', value='nan')
    
    for i, path_tuple in tqdm(enumerate(csv_paths.itertuples())):
        edge_id_list = eval(path_tuple[1])
        length_list = list()
        edge_gaussians = list()

        for edge_id in edge_id_list:
            edge_data = edge_lookup[edge_id]
            length_list.append(edge_data[1])

            edge_gaussians.append(edge_data[0])

        csv_paths.iloc[i,0] = str(edge_gaussians)
        csv_paths.iloc[i,1] = str(length_list)
        csv_paths.iloc[i,2] = str(eval(csv_paths.iloc[i,2]))
        
    print(csv_paths)

    csv_paths.to_csv(file_path_predicted + "all_edges_and_paths_predicted.csv")

def replace_to_mdn_normal_form(file_path_predicted):
    my = lambda x: [x[i] for i in range(len(x)) if i % 3 == 0]
    sigma = lambda x: [x[i] for i in range(len(x)) if i % 3 == 1]
    alpha = lambda x: [x[i] for i in range(len(x)) if i % 3 == 2]

    mdn_normal_form = lambda x: alpha(x) + my(x) + sigma(x)

    csv_paths = pd.read_csv(file_path_predicted + "all_edges_and_paths_predicted.csv", index_col=0)

    for i, path_tuple in tqdm(enumerate(csv_paths.itertuples())):
        edges = eval(path_tuple[1])
        edge_list = list()
        truth = eval(path_tuple[3])

        for edge in edges:
            edge_list.append(mdn_normal_form(edge))
        
        csv_paths.iloc[i, 0] = str(edge_list)
        csv_paths.iloc[i,2] = str(mdn_normal_form(truth))

    csv_paths.to_csv(file_path_predicted + "all_speed_mdn_normal_form.csv")

def combine_individual_predictions(file_path_predicted, file_path_index):
    file_name = file_path_predicted + "time/mdn_edges_time_prediction_"
    collected = dict()
    values = list()
    for i in range(11):
        res = pickle.load(open(file_name + str(i) + ".p", "rb"))
        for key in res:
            collected[key] = res[key]

    for key in collected:
        values += list(collected[key])

    csv_original = pd.read_csv(file_path_index + "all_edges_time.csv", index_col=0)

    for i in tqdm(range(len(values))):
        csv_original.iloc[i,1] = str(list(values[i]))

    csv_original.to_csv(file_path_predicted + "time/all_edges_predicted.csv")

def combine_individual_predictions_paths(file_path_predicted, file_path_index):
    file_name = file_path_predicted + "time/mdn_paths_time_prediction_"
    collected = dict()
    values = list()

    for i in range(2):
        res = pickle.load(open(file_name + str(i) + ".p", "rb"))
        for key in res:
            collected[key] = res[key]

    for key in collected:
        values += list(collected[key])

    csv_original = pd.read_csv(file_path_index + "all_paths_time.csv", index_col=0)

    for i in tqdm(range(len(values))):
        csv_original.iloc[i,1] = str(list(values[i]))

    csv_original.to_csv(file_path_predicted + "time/all_paths_predicted.csv")
    
    
def quick_time_test(file_path_index, file_path_predicted):
    csv_original = pd.read_csv(file_path_predicted + "time/all_paths_predicted.csv", index_col=0)
    l = list()
    for i in range(len(csv_original)):
        times = eval(csv_original.iloc[i,1])

        p = lambda x, y: x[8*y:8*(y+1)]
        m = lambda x: p(x, 1)
        a = lambda x: p(x, 0)
        s = lambda x: p(x, 2)

        for z, time in enumerate(m(times)):
            if time > 330:
                l += [(round(time, 3), round(s(times)[z], 3), round(a(times)[z], 3))]

if __name__ == "__main__":
    file_path_index = "models/data/mdn_data/prediction_data/index_data/"
    file_path_predicted = "models/data/mdn_data/prediction_data/predicted_data/time/"

    """All the methods called in this file needs to be called in a specific order over multiple executions"""

    # find_best_mdn_conf()

    # quick_time_test(file_path_index, file_path_predicted)
    # combine_individual_predictions(file_path_predicted, file_path_index)
    # combine_individual_predictions_paths(file_path_predicted, file_path_index)

    # combine_mdn_predict_edge_data()
    # combine_mdn_predict_path_data()

    # process_mdn_data_to_dict(file_path_index)

    # test_csv_vs_pickle(file_path_index)

    # combine_mdn_predicted_data(file_path_predicted, file_path_index)

    # test_combine_mdn_predicted_data(file_path_index, file_path_predicted)

    # replace_edge_ids_with_data(file_path_predicted)

    # dataframe = pd.read_csv(file_path_predicted + "all_edges_and_paths_predicted.csv")

    # replace_to_mdn_normal_form(file_path_predicted)

    # for i, d_tuple in enumerate(dataframe.itertuples()):
    #     path_truth = eval(dataframe.iloc[i,3])
        
    #     # path_truth = eval(path_truth)

    #     print(path_truth)
    #     print(sort_gaussians(path_truth))
    #     print(flatten_gaussians(sort_gaussians(path_truth)))
        

    #     break