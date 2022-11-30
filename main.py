"""
main file to test the libraries while under development

TODO: Overflow handling - more testing
"""

import matplotlib.pyplot as plt

from model.training_methods import BackProp
from model.network import NeuralNetwork

from vector import Vector, activation

if __name__ == "__main__":

    example_inputs = [Vector([0,0,0,0]),
                    Vector([1,0,1,0]),
                    Vector([0,1,1,0])]

    example_outputs = [Vector(0), Vector(1), Vector(0)]

    training_method = BackProp(activation.sigmoid_prime)
    model = NeuralNetwork((4, 1, 5, 5), activation.sigmoid, training_method)

    cost = model.train_network(example_inputs, example_outputs, 10000)

    plt.plot(cost)
    plt.show()
