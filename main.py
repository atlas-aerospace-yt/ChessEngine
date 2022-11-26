"""
main file to test the libraries while under development

TODO: Overflow handling - more testing
"""

import matplotlib.pyplot as plt

from model.training_methods import BackProp
from model.network import NeuralNetwork

from vector import Vector, activation

if __name__ == "__main__":

    example_one = Vector([0,0,0,0])
    example_two = Vector([1,0,1,0])
    example_three = Vector([0,1,1,0])

    training_method = BackProp(activation.sigmoid_prime)

    EPOCH = 2000

    for item in [0.5, 1.0, 1.5]:

        cost = []

        network = NeuralNetwork((4, 1, 4, 10)
                                , activation.sigmoid, training_method)
        network.learn_rate = item

        for i in range(EPOCH):
            network.train_network(example_one, Vector([0]))
            network.train_network(example_two, Vector([1]))
            network.train_network(example_three, Vector([0]))

            predicted_output = []
            predicted_output.append(network.forward_propagation(example_one))
            predicted_output.append(network.forward_propagation(example_two))
            predicted_output.append(network.forward_propagation(example_three))

            actual_output = [Vector(0), Vector(1), Vector(0)]

            cost.append(network.cost_function(actual_output, predicted_output))

            print(i)

        print(network.forward_propagation(Vector([0,1,1,1])))
        print(network.forward_propagation(Vector([1,0,0,0])))

        plt.plot([i for i in range(EPOCH)], cost)

    plt.show()
