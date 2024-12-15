import numpy as np
import matplotlib.pyplot as plt
import h5py

file = h5py.File("data/ANN.h5", 'r')

# Assuming the dataset 'model_weights' contains a 1D array of weights
dataset_name = 'model_weights'
weights_dataset = file[dataset_name]
weights_values = weights_dataset[:]

print("Values in the 'model_weights' dataset:")
print(weights_values)

file.close()
