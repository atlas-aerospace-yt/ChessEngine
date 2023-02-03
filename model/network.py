"""
Neural network class which holds all the layers of the network.

This file is intended to be used to import NeuralNetwork and then to
pass in a training methof from training_methods.py
"""

from model.layer import Layer
from model.file import FileManager

class NeuralNetwork:
    """
    Class of a neural network

    Args:
        tuple(num_of_input, num_of_output, num_of_layers, num_of_nodes)
        activation_func
        derivative_func

    Attributes:
        network (list): Holds a list of layer objects from layer.py
        training_method (class): holds the train_network function
            note: this class must be from training_methods.py

    Methods:
        train_network:
            Trains the network using one of the classes from "training_methods.py"

            Args:
                input_vector_list (list): the inputs to the network
                output_vector_list (list): the wanted outputs
                epoch(int): the number of iterations to train the network over

        forwards_propagation:
            Forward propagation performs the prediction of the neural network

            Args:
                input_vector (Vector): the input vector to predict
    """

    def __init__(self, network_stats, activation_func, training_method):

        self.file_manager = FileManager(network_stats)

        num_of_input = network_stats[0]
        num_of_output = network_stats[1]
        num_of_layers = network_stats[2]
        num_of_nodes = network_stats[3]

        # the first and last layers have a different number of inputs and outputs respectively
        first_layer = Layer(activation_func, input_amt=num_of_input, output_amt=num_of_nodes)
        last_layer = Layer(activation_func, input_amt=num_of_nodes, output_amt=num_of_output)

        self.network = [first_layer]
        self.training_method = training_method

        # iterates through the rest of the layers as they are the same
        for _ in range(0, num_of_layers - 2):
            self.network.append(Layer(activation_func, input_amt=num_of_input, output_amt=num_of_nodes))

        self.network.append(last_layer)

        self.file_manager.save_network(self.network)
        print(item for item in self.file_manager.load_network())

    def __str__(self):
        """
        Returns a formatted string of layers with information

        Returns:
            str: The string holding all the information of the netowrk
        """

        string = ""

        for index, layer in enumerate(self.network):
            string += f"Layer {index+1}:\n\n{layer}\n\n"

        return string

    def train_network(self, input_vector_list, output_vector_list, epoch=100):
        """
        Trains the network using one of the classes from "training_methods.py"

        Args:
            input_vector_list (list): the inputs to the network
            output_vector_list (list): the wanted outputs
            epoch(int): the number of iterations to train the network over
        """

        return self.training_method.train_network(self,
                                                  input_vector_list, output_vector_list, epoch)

    def forward_propagation(self, input_vector):
        """
        Forward propagation performs the prediction of the neural network

        Args:
            input_vector (Vector): the input vector to predict

        This uses the equation:
            z = sigmoid(w * x + b)

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s
        """

        self.training_method.outputs = [input_vector]
        self.training_method.deactivated_outputs = [input_vector]

        # iterates through each layer and performs the forward propagation calculation
        for layer in self.network:
            output = layer.forward_propagation(self.training_method.outputs[-1])
            self.training_method.deactivated_outputs.append(output[0])
            self.training_method.outputs.append(output[1])

        return output[1]
