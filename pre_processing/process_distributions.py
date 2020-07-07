import os
import pickle
import subprocess

def pickle_distributions():
    process_manager = []
    lengths = []

    # Puts all time intervals on RNN input shape (dist, len, time)
    for i in range(24):
        process_manager.append(subprocess.Popen(["python", "-m", "pre_processing.process_for_model_pickle", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE))

    # process manager waits for each process to output its max length for comparison 
    for p in process_manager:
        p.wait()
        out, _ = p.communicate()
        string = str(out)
        index = str(out).find("Length")
        
        lengths.append(float(string[index + 7:-3]))

    # Return the longest edge of all time intervals 
    return max(lengths)

# Opens each distribution for each time interval normalizes them and sorts the files compared to length
def collect_normalize(max_length):
    distributions_dict = {}
    truths_dict = {}
    file_path = "models/data/full_mdn/time/"

    # Loop each temp_file to load distributions for each time interval
    for file_index in range(1):
        file_name = file_path + "data_pickle_" + str(file_index).zfill(2) + ".p"
        distribution, truth = pickle.load(open(file_name, "rb"))
        os.remove(file_name) # Remove the temp files 

        # Enumerate each distribution shape
        for truth_shape_index, distribution_shape_list in enumerate(distribution):
            # Enumerate each input for specific shape
            for truth_index, rnn_input_list in enumerate(distribution_shape_list):
                shape = len(rnn_input_list[0])

                for rnn_input in rnn_input_list[0]:
                    rnn_input[-1] = rnn_input[-1] / max_length
                
                distributions_temp = distributions_dict.get(shape) if distributions_dict.get(shape) is not None else []
                truth_temp = truths_dict.get(shape) if truths_dict.get(shape) is not None else []

                distributions_temp.append(rnn_input_list)
                truth_temp.append(truth[truth_shape_index][truth_index])

                # Add all distributions to dict to divide into path length and not time
                distributions_dict[shape] = distributions_temp
                truths_dict[shape] = truth_temp


    for key in distributions_dict.keys():
        pickle.dump((distributions_dict[key], truths_dict[key]), open(file_path + "distribution_L_" + str(key).zfill(2) + ".p", "wb"))
        
if __name__ == "__main__":
    # Navigate to the folder the distributions lie in before use
    # max_length = pickle_distributions() # Run each distribution to make RNN input shape and return length
    max_length = 17531.2
    collect_normalize(max_length) # Normalize length and sort files in number of edges 
