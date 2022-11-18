"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Create a neural network
"""

from vector import random

# from vector import activation

from vector import Vector


class Layer():
    """
    This class holds an list of vectors which is intended to be used as a layer
    """

    def __init__(self, input_amt, output_amt):
        """
        private object weights is a list of vectors
        private object biases is a vector
        """
        self.__weights = [random.random_vector(input_amt) for _ in range(output_amt)]
        self.__biases = random.random_vector(output_amt)

    def __getitem__(self, index):
        """
        uses the built in python function of list[index]

        Args:
            index (int): the index for the item within list (0 -> len(layer))
        Return:
            tuple: weight and bias in that index
        """
        return (self.__weights[index], self.__biases[index])

    def __str__(self):
        """
        converts the layer into an easy to read string

        TODO -> add more info about layer to string

        return:
            string: the information about the layer
        """
        display_weights = ""

        for value in self.__weights:
            display_weights += f"{str(value)}\n"

        return f"weights: {display_weights}\n\
biases: {self.__biases}\n"

    @property
    def weights(self):
        """
        gets the private weights object

        Return:
            list: a list of vector
        """
        return self.__weights

    @weights.setter
    def weights(self, new_weights):
        """
        sets the private object weights to the new weights

        Args:
            new_weights (vector): defines the new weights
        """
        self.__weights = new_weights

    @property
    def biases(self):
        """
        gets the private biases object

        Return:
            Vector: the biases of this layer
        """
        return self.__biases

    @biases.setter
    def biases(self, new_biases):
        """
        sets the private objext biases to the new biases

        Args:
            new_biases (vector): defines the new biases
        """
        self.__biases = new_biases

if __name__ == "__main__":

    layer = Layer(10, 1)

    print(layer)

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    print(random.random_vector(4))

    print(a * b)
