import numpy as np


class Layer:

    def __init__(self, n_inputs, n_nodes,
                 weight_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):

        #Initializes weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forwardProp(self, input):
        self.output = np.dot(input, self.weights) + self.biases
        self.input = input

    def backProp(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                             self.biases


        self.dinputs = np.dot(dvalues, self.weights.T)
