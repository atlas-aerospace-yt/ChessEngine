"""
main file to test the libraries while under development
"""

from model.network import NeuralNetwork
from vector import activation
from vector import Vector

if __name__ == "__main__":

    example_one = Vector([0,0])
    example_two = Vector([1,0])
    example_three = Vector([0,0])

    network = NeuralNetwork(2, 1, 2, 2, activation.sigmoid, activation.sigmoid_prime)

    network.cost_function_derivative(example_one, Vector([0]))

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    if float(a * b) != 70.0:
        print("Error: multiplication failed!")
