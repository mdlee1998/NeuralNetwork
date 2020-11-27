import numpy as np


class Layer:

    def __init__(self, n_inputs, n_nodes):

        #Initializes weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))

    def forwardProp(self, input):
        self.output = np.dot(input, self.weights) + self.biases
