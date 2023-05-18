'''
Helper loading functions from assignment 4

@Author: George Witt
(NOTE: Some code is taken from assignment 4 for MNIST data loading, 
originally written by Dayal Singh, these functions include...

'one_hot', 'standardize', and 'load_data' for interfacing with the class
datastructres).
'''

import numpy as np
import pickle as pl

def _one_hot(x, k, dtype = np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def standardize(x):
  """Standardization per sample across feature dimension."""
  axes = tuple(range(1, len(x.shape)))
  mean = np.mean(x, axis=axes, keepdims=True)
  std_dev = np.std(x, axis=axes, keepdims=True)
  return (x - mean) / std_dev

def load_data(dataset):
    """
    loads mnist dataset from google drive
    """

    path = f'datasets/{dataset}/{dataset}.dump'    
    in_file = open(path, 'rb')
    (x_train, y_train), (x_test, y_test) = pl.load(in_file)

    #flatten x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    #get info about the numner of training and testing exammples
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    #standardize input
    x_train, x_test = standardize(x_train), standardize(x_test)

    return (x_train, y_train), (x_test, y_test), (num_train, num_test)