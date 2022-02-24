import csv
import pandas as pd
import os
import numpy as np

def split_data(data, test_ratio=0.20):
    """
    Split a dataset into a training and test set, where test_ratio refers to the % size of the test set(test_ratio * dataset_size).

    Args:
        data (pd.DataFrame): dataset
        test_ratio (float, optional): size of test set w.r.t training set, Defaults to 0.20 per convention.

    Returns:
        (pd.DataFrame, pd.DataFrame): (training_set, test_set)
    """    
    # permutate a an array based on the number of data elements eg. [1, 5, 9, ..., n-5, n, n-3, 12, ... , 2]
    generate_random_index = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    # permutated indices set for training set and test set with size test_ratio (0.2 def*)
    test_set_indices = generate_random_index[:test_set_size]
    training_set_indices = generate_random_index[test_set_size:]
    
    # return permutated dataframe of training set and test set
    return data.iloc[training_set_indices], data.iloc[test_set_indices] 

