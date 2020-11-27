import numpy as np
from Activations import Softmax

class Cost:

    def calc(self, output, true_y):

        sample_costs = self.forwardProp(output, true_y)

        return np.mean(sample_costs)


class CategoricalCrossentropy(Cost):

    def forwardProp(self, pred, true_y):

            samples = len(pred)

            #Clip data to prevent division by zero, to both sides to prevent
            #altering of the mean
            pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)

            if len(true_y.shape) == 1:
                confidence = pred_clipped[
                    range(samples),
                    true_y
                ]
            elif len(true_y.shape) == 2:
                confidence = np.sum(
                    pred_clipped * true_y,
                    axis=1
                )

            return -np.log(confidence)

    def backProp(self, dvalues, true_y):

        samples = len(dvalues)


        labels = len(dvalues[0])

        if len(true_y.shape) == 1:
            true_y = np.eye(labels)[true_y]

        self.dinputs = -true_y / dvalues
        self.dinputs = self.dinputs / samples



class Softmax_CategoricalCrossentropy(Cost):

    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

    def forwardProp(self, input, true_y):

        self.activation.forwardProp(input)

        self.output = self.activation.output

        return self.loss.calc(self.output, true_y)

    def backProp(self, dvalues, true_y):

        samples = len(dvalues)

        if len(true_y.shape) == 2:
            true_y = np.argmax(true_y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), true_y] -= 1
        self.dinputs = self.dinputs / samples
