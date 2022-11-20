"""
main file to test the libraries while under development
"""

from model.network import NeuralNetwork
from vector import activation
from vector import Vector

if __name__ == "__main__":

    example_one = Vector([0,0])
    example_two = Vector([1,0])
    example_three = Vector([0,1])

    network = NeuralNetwork((2, 1, 10, 10), activation.sigmoid, activation.sigmoid_prime)

    print(network.forward_propagation(Vector([1,1])))

    for i in range(0, 1000):
        guess = network.backward_propagation(example_one, Vector([0]))
        guess = network.backward_propagation(example_two, Vector([1]))
        guess = network.backward_propagation(example_three, Vector([0]))

    print(network.forward_propagation(Vector([1,0])))

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    if float(a * b) != 70.0:
        print("Error: multiplication failed!")
