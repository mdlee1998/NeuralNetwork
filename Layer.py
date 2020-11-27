import numpy as np


class Layer:

    def __init__(self, n_inputs, n_nodes):

        #Initializes weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))

    def forwardProp(self, input):
        self.output = np.dot(input, self.weights) + self.biases
        self.input = input

    def backProp(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)
