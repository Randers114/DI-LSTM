import pickle
import pandas as pd
import tqdm

def open_training_data(hour):
    dataframe = pd.read_csv('models/data/mdn_data/training/time/training_mdn' + str(hour).zfill(2) + '.csv')
    data = list()

    for row in dataframe.itertuples(index=False):
        data.append(eval(row[1]))

    return data

def pickle_training_data(pickle_length):
    edges = list()

    for i in tqdm.tqdm(range(24)):
        edges += (open_training_data(i))


    for i in tqdm.tqdm(range(0, len(edges), pickle_length)):
        temp = list()

        for edge in edges[i:i+pickle_length]:
            temp.append(edge)
        pickle.dump(temp, open("models/data/mdn_data/training/time/pickle/mdn_pickle_training_" + str(int(i / pickle_length)) + ".p", 'wb'))

def open_pickle():
    edges = list()
    for i in range(11):
        edges += pickle.load(open("models/data/mdn_data/training/time/pickle/mdn_pickle_training_"+ str(i) + ".p", "rb"))

    return edges

if __name__ == "__main__":
    pickle_training_data(50000)
