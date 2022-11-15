"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Re-write file_manager
TODO -> Create a neural network
"""

import vector as vec
from vector import Vector as v


class Learn():
    """
    This class holds the weights and biases as well as some operators
    for machine learning / neural networks.
    """

    def __init__(self):

        self.weights = self.load_weights()
        self.biases = self.load_biases()

    def load_weights(self):
        """
        Loads the weights for the network

        TODO -> load weights from a file

        Returns:
            Vector: the weights for the network
        """

        return vec.random_vector(10)

    def load_biases(self):
        """
        Loads the biases for the network

        TODO -> load biases from a file

        Returns:
            Vector: the biases for the network
        """

        return vec.random_vector(1)

    def forward_propagation(self, input_state):
        """
        Forward propagation predicts the output based on the input

        Args:
            input (vector): the input to recognise patterns within
        """

        output = input_state * self.weights + self.biases

        return vec.sigmoid(output)


learning = Learn()

if __name__ == "__main__":
    a = v([1, 2, 3, 4])
    b = v([5, 6, 7, 8])

    print(vec.random_vector(4))

    print(a * b)
