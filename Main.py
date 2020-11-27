from Layer import Layer
from Activations import Softmax, ReLU
from Cost import Softmax_CategoricalCrossentropy as Soft_Ce
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

def main():
    nnfs.init()
    X, y = spiral_data(samples = 100, classes = 3)

    dense1 = Layer(2, 3)
    activation1 = ReLU()

    dense2 = Layer(3,3)
    activation2 = Softmax()

    cost_avc = Soft_Ce()

    dense1.forwardProp(X)
    activation1.forwardProp(dense1.output)

    dense2.forwardProp(activation1.output)
    activation2.forwardProp(dense2.output)


    cost = cost_avc.forwardProp(dense2.output, y)

    print(cost_avc.output[:5])

    print("Cost: ", cost)

    predictions = np.argmax(cost_avc.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    print("Acc: ", accuracy)

    cost_avc.backProp(cost_avc.output, y)
    dense2.backProp(cost_avc.dinputs)
    activation1.backProp(dense2.dinputs)
    dense1.backProp(activation1.dinputs)

    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)




if __name__ == "__main__":
    main()
