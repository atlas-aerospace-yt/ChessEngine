"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Create a neural network
"""

from vector import random

# from vector import activation

from vector.vector import Vector


# class Network():
#
#    def __init__(self):
#
#        self.__weights = []
#        self.__biases = []
#
#    def set_weights(self, new_weights):
#        """
#        sets the private object weights to the new weights
#
#        Args:
#            new_weights (vector): defines the new weights
#        """
#
#        self.__weights = new_weights
#
#    def set_biases(self, new_biases):
#        """
#        sets the private objext biases to the new biases
#
#        Args:
#            new_biases (vector): defines the new biases
#        """
#
#        self.__biases = new_biases


if __name__ == "__main__":
    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    print(random.random_vector(4))

    print(a * b)
