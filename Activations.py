import numpy as np

class ReLU:

    def forwardProp(self, input):
        self.output = np.maximum(0, input)
        self.input = input


    def backProp(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.input <= 0] = 0

class Softmax:

    def forwardProp(self, input):

        #Gets unnormalized probabilities
        exp_val = np.exp(input - np.max(input, axis=1, keepdims=True))

        #Normalize probabilites for each sample
        self.output = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.input = input


    def backProp(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
