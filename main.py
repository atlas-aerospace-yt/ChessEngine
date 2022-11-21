"""
main file to test the libraries while under development

TODO -> Overflow handling - more testing
"""

from model.network import NeuralNetwork
from vector import activation
from vector import Vector

if __name__ == "__main__":

    example_one = Vector([0,0,0,0])
    example_two = Vector([1,0,1,0])
    example_three = Vector([0,1,1,0])

    network = NeuralNetwork((4, 1, 3, 3), activation.sigmoid, activation.sigmoid_prime)

    for i in range(0, 10000):
        network.backward_propagation(example_one, Vector([0]))
        network.backward_propagation(example_two, Vector([1]))
        network.backward_propagation(example_three, Vector([0]))

        predicted_output = []
        predicted_output.append(network.forward_propagation(example_one))
        predicted_output.append(network.forward_propagation(example_two))
        predicted_output.append(network.forward_propagation(example_three))

        actual_output = [Vector(0), Vector(1), Vector(0)]

        print(network.cost_function(actual_output, predicted_output))

    print(network.forward_propagation(Vector([1,0,0,0])))
