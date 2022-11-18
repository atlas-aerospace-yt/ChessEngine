"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Finish layer class
TODO -> Add error handling to getters and setters
"""

import sys

from vector import random
from vector import activation
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

        return:
            string: the information about the layer
        """
        display_weights = ""

        for value in self.__weights:
            display_weights += f"{str(value)}\n"

        return f"Weights: {display_weights}\n\
Biases: {self.__biases}\n\n\
Information:\n\
    size of weights: {sys.getsizeof(self.__weights)} bytes\n\
    size of biases: {sys.getsizeof(self.__biases)} bytes\n\
    number of input nodes: {len(self.__weights[0])}\n\
    number of output nodes: {len(self.__biases)}"

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

    def forward_propagation(self, layer_input):
        """
        performs forward propagation using the layer

        Args:
            layer_input (Vector): the vector which is the input into the layer
        Return:
            Vector: the result of the forward pass
        """
        prediction = Vector([layer_input * node for node in self.__weights])
        prediction += self.__biases
        prediction = activation.sigmoid(prediction)

        return prediction

    def backwards_propagation(self, layer_output, layer_input, cost_derivative):
        """
        performs the backwards propagation calculation

        Args:
            layer_output (Vector): the predicted output of the layer
            layer_input (Vector): the input that caused the prediction
            cost_derivative (Vector): the cost / output gradient
        Returns:
            Vector: bias gradient
            Vector: weight gradient
        """

        weight_gradient = layer_input * activation.sigmoid_prime(layer_output) * cost_derivative
        bias_gradient = activation.sigmoid_prime(layer_output) * cost_derivative

        return bias_gradient, weight_gradient

if __name__ == "__main__":

    layer = Layer(10, 1)

    print(layer.forward_propagation(random.random_vector(10)))

    a = Vector([1, 2, 3, 4])
    b = Vector([5, 6, 7, 8])

    print(a * b)
