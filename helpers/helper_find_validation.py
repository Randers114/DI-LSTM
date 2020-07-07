import os
import pickle

import pandas as pd

from models.model_validation_output import print_model_validation

def find_best_validation(path, condition):
    def find_quantiles_bias(quantiles_list: list, dataframe: pd.DataFrame) -> list:
        quantiles_bias = []

        for q in quantiles_list:
            quantiles_bias.append(list(map(lambda x: round(x, 4), list(dataframe.quantile(q)))))

        return [[i[0] for i in quantiles_bias], [i[1] for i in quantiles_bias], [i[2] for i in quantiles_bias], [i[3] for i in quantiles_bias]]

    files = os.listdir(path)
    quantiles_list = [0.1]

    best_two_files = [("", 0), ("", 0)]

    for name in files:
        if "history" in name or not name.endswith(".p") or not condition is None and not condition in name:
            continue
        print(name)

        (val, _) = pickle.load(open(path + name, "rb"))

        # Dataframe with overall box plot
        frame_data = [v for i, v in enumerate(val) if i != 1]
        dataframe = pd.DataFrame(frame_data).T
        dataframe.columns = ["C", "I", "B", "KL"]

        quantiles_bias = find_quantiles_bias(quantiles_list, dataframe)

        means_bias = list(map(lambda x: round(x, 4), list(dataframe.mean())))

        # Compare
        new_value = quantiles_bias[0][0] + means_bias[0]

        if (new_value > best_two_files[0][1]):
            best_two_files[0] = (name, new_value)
            best_two_files = sorted(best_two_files, key=lambda x: x[1])
        
    print(best_two_files[0][0][:-2] + "_" + best_two_files[1][0][:-2])
    print_model_validation(best_two_files[0][0][:-2], best_two_files[1][0][:-2])
