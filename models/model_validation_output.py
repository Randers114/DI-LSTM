import os
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def model_results_append(name: str, collected_values: list, box: list) -> None:
    """Creates a file inside the test_data folder if one does not exist.
    If one does exist it will append the results if given a name, a list of collected data(method)
    and a list of box data (method box)"""

    name = "models/test_data/" + name + ".p"
    try:
        previous_values, val_box_plot = pickle.load(open(name, "rb"))

        for i, prev in enumerate(previous_values):
            prev += collected_values[i]

        for i, metric in enumerate(val_box_plot):
            for n, length in enumerate(metric):
                length += box[i][n]

        pickle.dump((previous_values, val_box_plot), open(name, "wb"))
    except:
        if not os.path.exists('models/test_data'):
            os.makedirs('models/test_data')
        pickle.dump((collected_values, box), open(name, "wb"))

def print_dynamic_model_comparison(file_names: list, columns: list, title: str, x_label: str) -> None:
    """Takes a list of file names, a list of column names, a title of the box plot and a label for the x-axis
    Will print the weighted box plot for each file name next to each other"""

    data = [pickle.load(open("models/test_data/" + n + ".p", "rb"))[0] for n in file_names]

    frame_data = []
    for i in [0, 2, 3, 4]:
        for sample in data:
            frame_data.append(sample[i])

    df = pd.DataFrame(frame_data).T

    means = list(map(lambda x: round(x, 4), list(df.mean())))
    columns = ["Correlation\n" + i for i in columns] + ["Intersection\n" + i for i in columns] + ["Bhat\n" + i for i in columns] + ["KL-D\n" + i for i in columns]

    df.columns = columns

    compute_box_plot(df, title, x_label, means)

def combine_columns(dataframe, column_values = 10):
    """Used to create different path categories in order to test performance on something other than the overall average.
    Returns a dataframe of combined columns that can be used to genereate results for those categories."""
    if type(column_values) == int:
        current_df = pd.DataFrame()
        current_max = 0
        for i in range(0, len(dataframe.columns), column_values):
            if i + column_values > len(dataframe.columns):
                current_max = i
                break
            col_name = str(dataframe.columns[i]) + '-' + str(dataframe.columns[i+column_values-1])
            cols = [dataframe.iloc[:,k].dropna() for k in range(i, i+column_values)]
            current_df[col_name] = pd.concat(cols, ignore_index=True)
            


        if len(dataframe.columns) % column_values != 0:
            col_name = str(dataframe.columns[current_max]) + '-' + str(dataframe.columns[-1])
            cols = [dataframe.iloc[:,k].dropna() for k in range(len(dataframe.columns) - len(dataframe.columns) % column_values, len(dataframe.columns))]
            current_df[col_name] = pd.concat(cols, ignore_index=True)

        return current_df

    else:
        current_min = 2
        current_df = pd.DataFrame()

        for val in column_values:
            current_max = val
            col_name = str(current_min) + "-" + str(current_max)
            cols = list()

            for col in dataframe.columns:
                if col >= current_min and col <= current_max:
                    cols.append(dataframe[col].dropna())

            current_df[col_name] = pd.concat(cols, ignore_index=True)
            current_min = current_max + 1

        return current_df

def compute_box_plot(df: pd.DataFrame, name: str, x_label: str, means: bool = False) -> None:
    """Creates and plots box plot from a dataframe"""
    ax = sns.boxplot(data=df, whis=[10, 90])

    if means:
        pos = range(len(means))
        for tick in pos:
            ax.text(pos[tick], means[tick], str(means[tick]),
                    horizontalalignment='center', size='medium', color='b', weight='semibold')
    plt.ylim(0, 1)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel('Values', fontsize=18)
    plt.title(name, fontsize=22)
    plt.show(ax)

def print_box_plots_path_lengths(file_names: list, start: int = 0, end: int = 15) -> None:
    """Will print each path length as a seperate box plot if given file names in a list. Has optional parameters
    of which interval to show the path lengths in"""
    for val_box_plot in [pickle.load(open("models/test_data/" + n + ".p", "rb"))[1] for n in file_names]:
        names = ["Correlation", "CHI", "Intersection", "Bhattacharyya", "KL-Div"]

        for i, b_plot in enumerate(val_box_plot):
            df = pd.DataFrame(b_plot).T

            df.columns = [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 101, 102, 104, 105, 107, 108, 109, 11, 110, 111, 113, 
                115, 116, 12, 123, 124, 125, 127, 128, 13, 133, 134, 135, 137, 139, 14, 15, 151, 16, 161, 
                169, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

            df = df.sort_index(axis=1)

            # df = combine_columns(df, [30, 70, 115, 169])
            means = list(map(lambda x: round(x, 3), list(df.mean())))

            if i != 1:
                compute_box_plot(df.iloc[:,start:end], names[i], "Path Length", means[start:end])

def print_grouped_box_plots(file_names: list, bounds, start: int = 0, end: int = None, metric = 0, groups=[30,169]):
    """Creates a box plot for the given metric for each of the given groups. Can take a list of file names to create 
    a boxplot for each of the models in those files. Can be changed to consider more models at the same time, by
    changing the if-else chain that creates the plots, as well as the label assigned to the plot further down."""
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='#000000')
        plt.setp(bp['means'], color="#00FFFF")
        for patch in bp['boxes']:
            patch.set_facecolor(color)
    
    def grouped_box_plot(df: pd.DataFrame, name: str, x_label: str, means: bool = False, number = 0):
        k = [[x for x in list(df[df.columns[i]]) if str(x) != 'nan'] for i in range(len(df.columns))]
        
        if number == 0:
            bpl = plt.boxplot(k, patch_artist=True, positions=np.array(range(len(k)))*2.0-0.51, sym='', widths=0.3, meanline=True, showmeans=True)
            
            set_box_color(bpl, '#2C7BB6')
            number += 1
        elif number == 1:
            bpl = plt.boxplot(k, patch_artist=True, positions=np.array(range(len(k)))*2.0-0.17, sym='', widths=0.3, meanline=True, showmeans=True)

            set_box_color(bpl, '#D7191C')
            number += 1

        elif number == 2:
            bpl = plt.boxplot(k, patch_artist=True, positions=np.array(range(len(k)))*2.0+0.17, sym='', widths=0.3, meanline=True, showmeans=True)

            set_box_color(bpl, '#008000')
            number += 1
        elif number == 3:
            bpl = plt.boxplot(k, patch_artist=True, positions=np.array(range(len(k)))*2.0+0.51, sym='', widths=0.3, meanline=True, showmeans=True)

            set_box_color(bpl, '#D2691E')
            number += 1
        # else:
        #     bpl = plt.boxplot(k, patch_artist=True, positions=np.array(range(len(k)))*2.0-0.70, sym='', widths=0.3, meanline=True, showmeans=True)

        #     set_box_color(bpl, '#f803fc')

        return number
    plt.figure(figsize=(7,4))

    number = 0

    for val_box_plot in [pickle.load(open("models/test_data/" + n + ".p", "rb"))[1] for n in file_names]:
        names = ["Correlation", "CHI", "Intersection", "Bhattacharyya", "KL-Div"]

        for i, b_plot in enumerate(val_box_plot):
            b_plot = filter_path_lengths(b_plot, metric)
            df = pd.DataFrame(b_plot).T

            df.columns = [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 101, 102, 104, 105, 107, 108, 109, 11, 110, 111, 113, 
                115, 116, 12, 123, 124, 125, 127, 128, 13, 133, 134, 135, 137, 139, 14, 15, 151, 16, 161, 
                169, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

            df = df.sort_index(axis=1)

            df = combine_columns(df, groups)

            # ticks = df.columns
            ticks = ["Short", "Medium", "Long", "Very Long"]
            means = list(map(lambda x: round(x, 3), list(df.mean())))

            if i == metric:
                print("Model number: 0 = our, 1 = original, 2 = math, 3 = lr, 4 = pure math")
                print("Model number:", number, "Metric:", i, means)

                number = grouped_box_plot(df.iloc[:, start:end], names[i], "Path Length", means[start:end], number)

    plt.plot([], c='#2C7BB6', label="LSTM-TG")
    plt.plot([], c='#D7191C', label="DI-LSTM-MLP-TG")
    plt.plot([], c='#008000', label='LSTM-SG')
    plt.plot([], c='#D2691E', label='DI-LSTM-MLP-SG')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=5, mode="expand", borderaxespad=0.)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=15)
    plt.yticks(fontsize = 15)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(bounds[0], bounds[1])
    plt.tight_layout()
    plt.savefig('boxcompare_' + str(metric) + '.png', dpi=300)

def print_model_validation(file_1, file_2=None) -> None:
    """Prints model validation metrics to a file if given a list of file names"""

    def file_string_constructor(res: list) -> str:
        # Formats strings with proper indentation
        def indent(string_list: list) -> str:
            string = ""
            for st in string_list:
                string += str(st) + " " * (10 - len(str(st)))

            return string + "\n"
        
        file_string = ""
        header = ["Metric", "Res", "Bias", "No-Bias"]
        metrics = ["Correlation", "Intersection", "Bhattacharyya", "KL-Div"]
        results = ["Mean", "90th", "75th", "25th", "10th"]
        file_string = indent(header)

        for metric_index, metric in enumerate(metrics):
            file_string += indent([metric, "", "", ""])

            for result_index, result in enumerate(results):
                file_string += indent(["", result, res[metric_index][result_index][0], res[metric_index][result_index][1]])

        return file_string
    
    def file_string_compare_constructor(res):
        # Formats strings with proper indentation
        def indent(string_list: list) -> str:
            string = ""
            for st in string_list:
                string += str(st) + " " * (15 - len(str(st)))

            return string + "\n"
        
        def compare_result(a, b):
            r = round(((a - b)/ b) * 100, 2)

            if r < 0:
                r = " " * (12 - len(str(a)) - len(str(r))) + str(r)
            else:
                r = " " * (11 - len(str(a)) - len(str(r))) + "+" + str(r)
            
            return str(a) + r + "%"

        file_string = ""
        
        names = [file_1, "", "", "", "", file_2, "", "", "", ""]
        header = ["Metric", "Res", "Weighted", "Not-Weighted"]
        metrics = ["Correlation", "Intersection", "Bhattacharyya", "KL-Div"]
        results = ["Mean", "90th", "75th", "25th", "10th"]
        file_string = indent(names)
        file_string += indent([*header, "", *header])

        for metric_index, metric in enumerate(metrics):
            file_string += indent([metric, "", "", "", "", metric, "", "", ""])

            for result_index, result in enumerate(results):
                file_string += indent([
                    "", result, compare_result(res[0][metric_index][result_index][0], res[1][metric_index][result_index][0]), compare_result(res[0][metric_index][result_index][1], res[1][metric_index][result_index][1]), "",
                    "", result, compare_result(res[1][metric_index][result_index][0], res[0][metric_index][result_index][0]), compare_result(res[1][metric_index][result_index][1], res[0][metric_index][result_index][1])])

        return file_string

    # Computes the quantiles for weighted data
    def find_quantiles_bias(quantiles_list: list, df: pd.DataFrame) -> list:
        quantiles_bias = []

        for q in quantiles_list:
            quantiles_bias.append(list(map(lambda x: round(x, 4), list(df.quantile(q)))))

        return [[i[0] for i in quantiles_bias], [i[1] for i in quantiles_bias], [i[2] for i in quantiles_bias], [i[3] for i in quantiles_bias]]

    # Computes the quantiles and means for unweighted data
    def find_quantiles_means_no_bias(quantiles_list: list) -> Tuple[list, list]:
        quantiles_no_bias = []
        means_no_bias = []


        val_box_plot[0] = filter_path_lengths(val_box_plot[0], 0)

        for i, b_plot in enumerate(val_box_plot):
            df = pd.DataFrame(b_plot).T

            means = list(df.mean())
            quantiles = []
            if i != 1:
                means_no_bias.append(round(sum(means)/len(means), 4))
                for q in quantiles_list:
                    quantiles.append(round(sum(list(df.quantile(q)))/len(list(df.quantile(q))), 4))
                quantiles_no_bias.append(quantiles)

        return quantiles_no_bias, means_no_bias

    # Formats the data for the file constructor function
    def format_input(means_bias: list, means_no_bias: list, quantiles_bias: list, quantiles_no_bias: list) -> list:
        temp = []
        means = list(zip(means_bias, means_no_bias))
        for i, q_b in enumerate(quantiles_bias):
            temp.append([means[i]] + list(zip(q_b, quantiles_no_bias[i])))

        return temp

    names = [file_1, file_2] if file_2 is not None else [file_1]
    all_results = []
    for name in names:
        (val, val_box_plot) = pickle.load(open("models/test_data/" + name + ".p", "rb"))
        quantiles_list = [0.9, 0.75, 0.25, 0.1]

        # Dataframe with overall box plot
        frame_data = [v for i, v in enumerate(val) if i != 1]
        frame_data[0] = filter_caps(frame_data[0], None, 0) # Filter the negative values

        df = pd.DataFrame(frame_data).T
        df.columns = ["C", "I", "B", "KL"]

        quantiles_bias = find_quantiles_bias(quantiles_list, df)
        quantiles_no_bias, means_no_bias = find_quantiles_means_no_bias(quantiles_list)
        means_bias = list(map(lambda x: round(x, 4), list(df.mean())))

        if file_2 is not None:
            all_results.append(format_input(means_bias, means_no_bias, quantiles_bias, quantiles_no_bias))
        else:
            file_string = file_string_constructor(format_input(means_bias, means_no_bias, quantiles_bias, quantiles_no_bias))

            file_connection = open("models/test_data/" + name + ".txt", "w+")
            file_connection.write(file_string)
            file_connection.close()

    if file_2 is not None:
        file_string = file_string_compare_constructor(all_results)
        file_connection = open("models/test_data/" + file_1 + "_" + file_2 + ".txt", "w+")
        file_connection.write(file_string)
        file_connection.close()

def filter_caps(data, upper_bound, lower_bound):
    """Filters all incorrect prediction values to be the theoretic value - correlation values can be < 0 in the code,
    which is theoretically impossible. So we set those values to zero to get proper values for the box-plots.
    The problems are due to pythons handling of floats."""
    if upper_bound is not None:
        data = [x if x <= upper_bound else upper_bound for x in data]
    if lower_bound is not None:
        data = [x if x >= lower_bound else lower_bound for x in data]

    return data

def filter_path_lengths(data, metric):
    """Uses the filter_caps method to filter all the path lengths in the method_box_plot lists of data."""
    temp = list()

    if metric == 0:
        upper_bound = 1
        lower_bound = 0
    elif metric == 2:
        upper_bound = 1
        lower_bound = 0
    elif metric == 4:
        upper_bound = None
        lower_bound = 0

    for length in data:
        temp.append(filter_caps(length, upper_bound, lower_bound))

    return temp
