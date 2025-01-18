import numpy as np
import pandas as pd
import math

def sigmoid(x):
        return 1/(1 + math.exp(-x))

def initialize_parameters_and_layers_NN(x_train,y_train):
    parameters = {"weight1": np.random.rand(3, x_train.shape[0])* 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.rand(x_train.shape[0],3)* 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters
def forward_propogation_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"],x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    cache = {"Z1" : Z1,
             "A1" : A1,
             "Z2" : Z2,
             "A2" : A2}
    return A2,cache
