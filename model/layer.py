"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Finish layer class
TODO -> Add error handling to getters and setters
"""

import sys

from vector import random
from vector import Vector


class Layer():
    """
    This class holds an list of vectors which is intended to be used as a layer

    Args:
        input_amt (int): the number of inputs to the layer
        output_amt (int): the number of outputs of the layer
        activation_function (function): the activation function of the layer

    Attributes:
        weights (private list): a list of vector
        biases (private vector): a vector to add to the layer output
        activation_function (private function): to perform the activation

    Methods:
        forward_propagation:
            performs forward propagation using the layer

            Args:
                layer_input (Vector): the vector which is the input into the layer
            Return:
                Vector: the result of the forward pass
    """

    def __init__(self, input_amt, output_amt, activation_function):
        """
        private object weights is a list of vectors
        private object biases is a vector
        """
        self.__weights = [random.random_vector(input_amt, lower=-1, upper=1) for _ in range(
output_amt)]
        self.__biases = random.random_vector(output_amt, lower=-1, upper=1)
        self.__activation_function = activation_function

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

        return f"\
Weights: {display_weights}\n\
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

    @property
    def activation_function(self):
        """
        gets the private objex - activation function

        Return:
            function: the function that is used as activation
        """

        return self.__activation_function

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
        activated_prediction = self.__activation_function(prediction)

        return [prediction, activated_prediction]
