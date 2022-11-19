"""
main file to test the libraries while under development
"""

from network.network import NeuralNetwork
from vector import activation
from vector import Vector

if __name__ == "__main__":

    example_one = Vector([0,0,0,0,0,0,0,0,0,0])
    example_two = Vector([1,0,0,0,1,0,0,0,0,0])
    example_three = Vector([0,0,0,0,1,0,0,0,0,0])

    network = NeuralNetwork(10, 1, 10, 10, activation.sigmoid)

    print(network.forward_propagation(example_one))
    print(network.forward_propagation(example_two))
    print(network.forward_propagation(example_three))

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    if float(a * b) != 70.0:
        print("Error: multiplication failed!")
