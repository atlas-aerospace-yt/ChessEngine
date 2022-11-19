"""
Neural network class which holds all the layers of the network.
"""

from model.layer import Layer

class NeuralNetwork:
    """
    Class of a neural network

    Attributes:
        network (list): Holds a list of layer objects from layer.py
        outputs (list): Holds vector which store the output of each layer

    Methods:
        cost_function:
            Args:
                inputs (list): a list of all the vector inputs
                outputs (list): a list of all the output vector
            Returns:
                list: a list of Vector objects
        forwards_propagation:
            Args:
                input_vector (Vector): the input to be processed
            Returns:
                Vector: the predicted output
    """

    def __init__(self, num_of_input, num_of_output, num_of_layers, num_of_nodes,
                activation_func, derivative_func):

        first_layer = Layer(num_of_input, num_of_nodes, activation_func)
        last_layer = Layer(num_of_nodes, num_of_output, activation_func)

        self.__derivative_func = derivative_func
        self.__outputs = []
        self.__derivatives = []
        self.__network = [first_layer]

        for _ in range(0, num_of_layers - 2):
            self.__network.append(Layer(num_of_nodes, num_of_nodes, activation_func))

        self.__network.append(last_layer)

    @property
    def outputs(self):
        """
        Gets the private variable outputs

        Returns:
            list: a list of vectors of outputs for each layer
        """
        return self.__outputs

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

    def cost_function_derivative(self, input_vector, output):
        """
        All of the derivatives with the respect to the cost function are calculated here
        and given to the private object - derivatives.

        Each derivative can then be indexed simply via self.__derivatives[layer][node]

        Args:
            input_vector (Vector): the input to the network
            output (Vector): the expected output of the network
        """
        predicted = self.forward_propagation(input_vector)

        self.__derivatives = [list((predicted - output) * 2)]

        layer = len(self.__derivatives) - 1

        for layer in range(len(self.__network)-1):
            delc_dela = []
            for i in range(len(self.__network[-layer])):
                total = 0
                for j in range(len(self.__derivatives[0])):
                    total += self.__network[layer].weights[i][j] * self.__derivative_func(self.__outputs[-layer][j]) * self.__derivatives[0][j]
                delc_dela.append(total)
            self.__derivatives = [delc_dela, *self.__derivatives]

        print(self.__derivatives)

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
            print(output)
        return self.__outputs[-1]
