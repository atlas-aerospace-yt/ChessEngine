"""
Neural network class which holds all the layers of the network.
"""

from network.layer import Layer

class NeuralNetwork:
    """
    Class of a neural network

    Attributes:
        network (list): Holds a list of layer objects from layer.py
        outputs (list): Holds vector which store the output of each layer
    """

    def __init__(self, num_of_input, num_of_output, num_of_layers, num_of_nodes, activation_func):

        first_layer = Layer(num_of_input, num_of_nodes, activation_func)
        last_layer = Layer(num_of_nodes, num_of_output, activation_func)

        self.__outputs = []
        self.network = [first_layer]

        for _ in range(0, num_of_layers - 2):
            self.network.append(Layer(num_of_nodes, num_of_nodes, activation_func))

        self.network.append(last_layer)

    def forward_propagation(self, input_vector):
        """
        Forward propagation performs the prediction of the neural network

        Args:
            input_vector (Vector): the input vector to predict
        """

        self.__outputs = [input_vector]

        for layer in self.network:

            output = layer.forward_propagation(self.__outputs[-1])
            self.__outputs.append(output)

        return self.__outputs[-1]
