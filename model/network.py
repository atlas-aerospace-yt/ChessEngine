"""
Neural network class which holds all the layers of the network.
"""

from model.layer import Layer

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
        self.__derivatives = []
        self.__network = [first_layer]

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

    def cost_function(self, inputs, outputs):
        """
        The cost function for the network. Uses the equation:
        cost = 1 / 2n Sum((y - y_hat) ^ 2)

        Args:
            inputs (list): a list of all the vector inputs
            outputs (list): a list of all the output vector
        Return:
            float: the sum of the cost function
        """
        if len(inputs) != len(outputs):
            raise Exception(f"Error: Input examples does not equal output examples. \
{(inputs)}, {len(outputs)}!")

        num_of_examples = len(inputs)

        multiplier = 1 / 2 * num_of_examples

        print(multiplier)

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

        layer = len(self.__derivatives) - 1

        for layer in range(len(self.__network)-1):
            delc_dela = []
            for i in range(len(self.__network[layer].weights[0])):
                total = 0
                for j in range(len(self.__derivatives[0])):
                    temporary_answer = self.__network[layer].weights[i][j]
                    temporary_answer *= self.__derivative_func(self.__outputs[layer][j])
                    total += temporary_answer * self.__derivatives[0][j]
                delc_dela.append(total)
            self.__derivatives = [delc_dela, *self.__derivatives]

    def backward_propagation(self, input_vector, output_vector):
        """
        Trains the neural network.

        Args:
            input_vector (list): list of example inputs
            output_vector (list): list of the corresponding outputs
        """
        predicted_vector = self.forward_propagation(input_vector)

        self.__cost_function_derivative(predicted_vector, output_vector)

        weight_grad = []
        for layer, network_layer in enumerate(self.__network):
            delc_delwi = []
            for i in range(len(network_layer.weights[0])):
                delc_delwj = []
                for j in range(len(self.__derivatives[layer])):
                    temporary_answer = self.__outputs[layer][i]
                    temporary_answer *= self.__derivative_func(self.__outputs[layer][j])
                    delc_delwj.append(temporary_answer * self.__derivatives[layer][j])
                delc_delwi.append(delc_delwj)
            weight_grad.append(delc_delwi)

    def forward_propagation(self, input_vector):
        """
        Forward propagation performs the prediction of the neural network

        Args:
            input_vector (Vector): the input vector to predict
        """

        self.__outputs = [input_vector]

        for layer in self.__network:

            output = layer.forward_propagation(self.__outputs[-1])
            self.__outputs.append(output)

        return self.__outputs[-1]
