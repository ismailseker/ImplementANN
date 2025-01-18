import numpy as np
import pandas as pd



def initialize_parameters_and_layers_NN(x_train,y_train):
    parameters = {"weight1": np.random.rand(3, x_train.shape[0])* 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.rand(x_train.shape[0],3)* 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters
