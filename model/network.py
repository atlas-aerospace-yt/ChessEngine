"""
Neural network class which holds all the layers of the network.
"""

from model.layer import Layer
from vector import Vector

class NeuralNetwork:
    """
    Class of a neural network

    Args:
        tuple(num_of_input, num_of_output, num_of_layers, num_of_nodes)
        activation_func
        derivative_func

    Attributes:
        network (list): Holds a list of layer objects from layer.py
        outputs (list): Holds vector which store the output of each layer
        derivatives (list): Holds the derivatives of the cost function
        derivative_func (function): The function needed to perform backwards propagation

    Methods:
        cost_function:
            The cost function for the network. Uses the equation:
            cost = 1 / 2n Sum((y - y_hat) ^ 2)

            Args:
                inputs (list): a list of all the vector inputs
                outputs (list): a list of all the output vector
            Return:
                float: the sum of the cost function

        cost_function_derivative:
            All of the derivatives with the respect to the cost function are calculated here
            and given to the private object - derivatives.

            Each derivative can then be indexed simply via self.__derivatives[layer][node]

            Args:
                input_vector (Vector): the input to the network
                output (Vector): the expected output of the network

        forwards_propagation:
            Forward propagation performs the prediction of the neural network

            Args:
                input_vector (Vector): the input vector to predict
    """

    def __init__(self, network_stats, activation_func, derivative_func):

        num_of_input = network_stats[0]
        num_of_output = network_stats[1]
        num_of_layers = network_stats[2]
        num_of_nodes = network_stats[3]


        first_layer = Layer(num_of_input, num_of_nodes, activation_func)
        last_layer = Layer(num_of_nodes, num_of_output, activation_func)

        self.__derivative_func = derivative_func
        self.__outputs = []
        self.__deactivated_outputs = []
        self.__derivatives = []
        self.__network = [first_layer]
        self.__learn_rate = 0.1

        for _ in range(0, num_of_layers - 2):
            self.__network.append(Layer(num_of_nodes, num_of_nodes, activation_func))

        self.__network.append(last_layer)

    def __str__(self):
        """
        Returns a formatted string of layers with information

        Returns:
            str: The string holding all the information of the netowrk
        """
        string = ""

        for index, layer in enumerate(self.__network):
            string += f"Layer {index+1}:\n\n{layer}\n\n"

        return string

    @property
    def outputs(self):
        """
        Gets the private variable outputs

        Returns:
            list: a list of vectors of outputs for each layer
        """
        return self.__outputs

    @property
    def network(self):
        """
        Gets the private variable network

        Returns:
            list: a list of layers
        """
        return self.__network

    @property
    def learn_rate(self):
        """
        Returns the current learn rate
        """
        return self.__learn_rate

    @learn_rate.setter
    def learn_rate(self, new_learn_rate):
        """
        Sets the learn rate to the new rate
        """
        self.__learn_rate = new_learn_rate

    def cost_function(self, output, predicted_output):
        """
        The cost function for the network. Uses the equation:
        cost = 1 / 2n Sum((y - y_hat) ^ 2)

        Args:
            inputs (list): a list of all the vector inputs
            outputs (list): a list of all the output vector
        Return:
            float: the sum of the cost function
        """
        if len(output) != len(predicted_output):
            raise Exception(f"Error: Input examples does not equal output examples. \
{(output)}, {len(predicted_output)}!")

        total = 0
        for index, item in enumerate(output):
            total += sum(predicted_output[index] - item) ** 2

        return total

    def __cost_function_derivative(self, predicted_vector, output_vector):
        """
        All of the derivatives with the respect to the cost function are calculated here
        and given to the private object - derivatives.

        Each derivative can then be indexed simply via self.__derivatives[layer][node]

        Args:
            predicted_vector (Vector): the actual output of the network
            output (Vector): the expected output of the network
        """

        self.__derivatives = [list((predicted_vector - output_vector) * 2)]

        for layer in range(len(self.__network)-1):
            delc_dela = []
            for weight in self.__network[layer].weights:
                total = 0
                for j in range(len(self.__derivatives[0])):
                    temporary_answer = weight[j]
                    temporary_answer *= self.__derivative_func(self.__outputs[layer][j])
                    total += temporary_answer * self.__derivatives[0][j]
                delc_dela.append(total)
            self.__derivatives = [delc_dela, *self.__derivatives]

    def update_weights(self):
        """
        Updates the networks weights
        """
        weight_grad = []
        for layer, network_layer in enumerate(self.__network):
            delc_delwi = []
            for i in range(len(network_layer.weights[0])):
                delc_delwj = []
                for j in range(len(self.__derivatives[layer])):
                    temporary_answer = self.__outputs[layer][i]
                    temporary_answer *= self.__derivative_func(
                        self.__deactivated_outputs[layer+1][j])
                    delc_delwj.append(temporary_answer * self.__derivatives[layer][j])
                delc_delwi.append(delc_delwj)
            weight_grad.append(delc_delwi)

        for layer, layer_grad in enumerate(weight_grad):
            for node, node_grad in enumerate(list(zip(*layer_grad))):
                self.__network[layer].weights[node] -= Vector(node_grad) * self.__learn_rate

    def update_biases(self):
        """
        Updates the networks biases
        """
        bias_grad = []
        for layer in range(len(self.__network)):
            delc_delwi = []
            for j in range(len(self.__derivatives[layer])):
                temporary_answer = self.__derivative_func(self.__deactivated_outputs[layer+1][j])
                delc_delwi.append(temporary_answer * self.__derivatives[layer][j])
            bias_grad.append(delc_delwi)

        for layer, gradient in enumerate(bias_grad):
            self.__network[layer].biases -= Vector(gradient) * self.learn_rate

    def backward_propagation(self, input_vector, output_vector):
        """
        Calls the update weights and biases functions

        Args:
            input_vector (Vector): the input to the network
            output_vector (Vector): the wanted output
        """
        predicted_vector = self.forward_propagation(input_vector)
        self.__cost_function_derivative(predicted_vector, output_vector)

        self.update_weights()
        self.update_biases()

    def forward_propagation(self, input_vector):
        """
        Forward propagation performs the prediction of the neural network

        Args:
            input_vector (Vector): the input vector to predict
        """

        self.__outputs = [input_vector]
        self.__deactivated_outputs = [input_vector]

        for layer in self.__network:

            output = layer.forward_propagation(self.__outputs[-1])
            self.__deactivated_outputs.append(output[0])
            self.__outputs.append(output[1])

        return output[1]
