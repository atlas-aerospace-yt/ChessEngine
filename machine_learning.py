"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Create a neural network
"""

from vector import random

# import vector.activation as activation

from vector.vector import Vector


# class Network():
#
#    def __init__(self):
#
#        self.__weights = self.set_weights()
#        self.__biases = self.set_biases()
#
#    def set_weights(self):
#
#        return random.random_vector(10)
#
#    def set_biases(self):
#
#        return random.random_vector(10)


if __name__ == "__main__":
    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    print(random.random_vector(4))

    print(a * b)
