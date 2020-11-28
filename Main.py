from Layer import Layer
from Activations import Softmax, ReLU
from Cost import Softmax_CategoricalCrossentropy as Soft_Ce
from Optimizer import Adam
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

def main():
    nnfs.init()
    X, y = spiral_data(samples = 1000, classes = 3)

    dense1 = Layer(2, 512, weight_regularizer_l2 = 5e-4,
                          bias_regularizer_l2 = 5e-4)
    activation1 = ReLU()

    dense2 = Layer(512,3)

    cost_act = Soft_Ce()

    optimizer = Adam(learning_rate = 0.02, decay = 5e-7)

    for epoch in range(10001):

        dense1.forwardProp(X)
        activation1.forwardProp(dense1.output)

        dense2.forwardProp(activation1.output)

        data_cost = cost_act.forwardProp(dense2.output, y)

        regularization_cost = \
            cost_act.cost.regularization_cost(dense1) + \
            cost_act.cost.regularization_cost(dense2)

        cost = data_cost + regularization_cost

        predictions = np.argmax(cost_act.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)


        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'cost: {cost:.3f}, (' +
                  f'data_cost: {data_cost:.3f}, ' +
                  f'reg_cost: {regularization_cost:.3f}), ' +
                  f'lr: {optimizer.curr_learning_rate}')


        cost_act.backProp(cost_act.output, y)
        dense2.backProp(cost_act.dinputs)
        activation1.backProp(dense2.dinputs)
        dense1.backProp(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    X_test, y_test = spiral_data(samples = 100, classes = 3)
    dense1.forwardProp(X_test)
    activation1.forwardProp(dense1.output)
    dense2.forwardProp(activation1.output)

    cost = cost_act.forwardProp(dense2.output, y_test)

    predictions = np.argmax(cost_act.output, axis=1)
    if len(y.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f'validation, acc: {accuracy:.3f}, cost: {cost:.3f}')


if __name__ == "__main__":
    main()
