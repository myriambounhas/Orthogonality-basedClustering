"""Create synthetic dataset = a CSV file"""

import numpy as np

# FOR LOADING DATA
from pandas import read_csv
import arff
from sklearn.utils import shuffle


def create_a_csv_row(f, dimension, _range=1):
    arr = [randint(0, _range) for _ in range(dimension)]
    row = ",".join(map(str, arr))
    label = f(arr)
    row += f",{label}"
    return row, arr
 
 
def load_supervised_dataset(filename):
    dataset = read_csv(filename, header=None)
    data = dataset.values
    X = data[:, :-1]
    y = data[:, -1]
    dimension = data.shape[1] - 1
    initial_size = data.shape[0]
    return data, X, y, dimension, initial_size

def load_unsupervised_dataset(filename):
    dataset = read_csv(filename, header=None)
    data = dataset.values
    X = data[:, :]
    dimension = data.shape[1] 
    initial_size = data.shape[0]
    return data, X, dimension, initial_size
 

def load_arffdataset(filename):
   # Load ARFF file

   # Load the ARFF file
    with open(filename, 'r') as f:
        dataset = arff.load(f)

    # Extract attributes and data
    attributes = dataset['attributes']  # Metadata about attributes
    raw_data = dataset['data']  # The actual data

    # Convert the data to a NumPy array
    data = np.array(raw_data, dtype=object)
    
    X = data[:, :-1]
    y = data[:, -1]
    dimension = data.shape[1] - 1
    initial_size = data.shape[0]
    return data, X, y, dimension, initial_size


