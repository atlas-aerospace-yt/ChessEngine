"""
main file to test the libraries while under development

TODO -> Overflow handling - more testing
"""

from model.network import NeuralNetwork
from vector import activation
from vector import Vector

if __name__ == "__main__":

    example_one = Vector([0,0])
    example_two = Vector([1,0])
    example_three = Vector([0,1])

    network = NeuralNetwork((2, 1, 2, 2), activation.sigmoid, activation.sigmoid_prime)

    print(network.forward_propagation(example_two))

    for i in range(0, 1000):
        network.backward_propagation(example_one, Vector([0]))
        network.backward_propagation(example_two, Vector([1]))
        network.backward_propagation(example_three, Vector([0]))

    print(network.forward_propagation(example_two))
    print(network.forward_propagation(example_three))

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    if float(a * b) != 70.0:
        print("Error: multiplication failed!")
