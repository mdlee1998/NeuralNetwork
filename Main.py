import Layer
import Activations
import nnfs
from nnfs.datasets import spiral_data

def main():
    nnfs.init()
    X, y = spiral_data(samples = 100, classes = 3)

    dense1 = Layer.Layer(2, 3)
    activation1 = Activations.ReLU()

    dense1.forwardProp(X)
    activation1.forwardProp(dense1.output)

    print(activation1.output[:5])



if __name__ == "__main__":
    main()
