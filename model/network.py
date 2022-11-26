"""
Neural network class which holds all the layers of the network.

TODO: This class needs to be able to inherit different training methods
      such as backwards propagation or random evolution.
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

            Each derivative can then be indexed simply via self.derivatives[layer][node]

            Args:
                input_vector (Vector): the input to the network
                output (Vector): the expected output of the network

        forwards_propagation:
            Forward propagation performs the prediction of the neural network

            Args:
                input_vector (Vector): the input vector to predict
    """

    def __init__(self, network_stats, activation_func, derivative_func, training_method):

        num_of_input = network_stats[0]
        num_of_output = network_stats[1]
        num_of_layers = network_stats[2]
        num_of_nodes = network_stats[3]


        first_layer = Layer(num_of_input, num_of_nodes, activation_func)
        last_layer = Layer(num_of_nodes, num_of_output, activation_func)

        self.derivative_func = derivative_func
        self.outputs = []
        self.deactivated_outputs = []
        self.derivatives = []
        self.network = [first_layer]
        self.learn_rate = 0.1
        self.training_method = training_method

        for _ in range(0, num_of_layers - 2):
            self.network.append(Layer(num_of_nodes, num_of_nodes, activation_func))

        self.network.append(last_layer)

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

    def cost_function_derivative(self, predicted_vector, output_vector):
        """
        All of the derivatives with the respect to the cost function are calculated here
        and given to the private object - derivatives.

        Each derivative can then be indexed simply via self.derivatives[layer][node]

        Args:
            predicted_vector (Vector): the actual output of the network
            output (Vector): the expected output of the network
        """

        self.derivatives = [list((predicted_vector - output_vector) * 2)]

        for layer in range(len(self.network)-1):
            delc_dela = []
            for weight in self.network[layer].weights:
                total = 0
                for j in range(len(self.derivatives[0])):
                    temporary_answer = weight[j]
                    temporary_answer *= self.derivative_func(self.outputs[layer][j])
                    total += temporary_answer * self.derivatives[0][j]
                delc_dela.append(total)
            self.derivatives = [delc_dela, *self.derivatives]

    def update_weights(self):
        """
        Updates the networks weights

        This is done using the equation:
            delC_0 / delw_ij^L = delz_j^L / delw_jk^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s
        """

        weight_grad = []

        # iterates through each layer in the network
        # then through each node and then goes through
        # each weight that feeds the node and calculates the
        # gradient of the weights against the cost function
        # j is the node and i is the weight
        for layer, network_layer in enumerate(self.network):
            delc_delwi = []
            for i in range(len(network_layer.weights[0])):
                delc_delwj = []
                for j in range(len(self.derivatives[layer])):
                    # each derivative is in a seperate line to keep the line size small
                    temporary_answer = self.outputs[layer][i]
                    temporary_answer *= self.derivative_func(
                        self.deactivated_outputs[layer+1][j])
                    delc_delwj.append(temporary_answer * self.derivatives[layer][j])
                delc_delwi.append(delc_delwj)
            weight_grad.append(delc_delwi)

        # with the gradient calculated in a multi dimensional array,
        # we iterate through each layer and node then update the weights
        for layer, layer_grad in enumerate(weight_grad):
            for node, node_grad in enumerate(list(zip(*layer_grad))):
                self.network[layer].weights[node] -= Vector(node_grad) * self.learn_rate

    def update_biases(self):
        """
        Updates the networks biases

        This is done using the equation:
            delC_0 / delb_j^L = delz_j^L / delb_j^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

        note: delz_j^L / delb_j^L = 1

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s
        """

        # iterates through each layer in the network
        # then through each node to calculate the gradient with
        # respect to the cost function
        # j is the node
        bias_grad = []
        for layer in range(len(self.network)):
            delc_delwi = []
            for j in range(len(self.derivatives[layer])):
                # each derivative is in a seperate line to keep the line size small
                temporary_answer = self.derivative_func(self.deactivated_outputs[layer+1][j])
                delc_delwi.append(temporary_answer * self.derivatives[layer][j])
            bias_grad.append(delc_delwi)

        # with the gradient calculated in a multi dimensional array,
        # we iterate through each layer to update the biases
        for layer, gradient in enumerate(bias_grad):
            self.network[layer].biases -= Vector(gradient) * self.learn_rate

    def backward_propagation(self, input_vector, output_vector):
        """
        Calls the update weights and biases functions

        TODO: return a list of costs
        TODO: loop within the function (EPOCH)
        TODO: allow the user to pass in the data seperately or grouped

        Args:
            input_vector (Vector): the input to the network
            output_vector (Vector): the wanted output
        """

        # gets the prediction from the network and calculates the cost derivative
        #predicted_vector = self.forward_propagation(input_vector)
        #self.cost_function_derivative(predicted_vector, output_vector)

        # updates the networks properties
        #self.update_weights()
        #self.update_biases()

        self.training_method.train_network(self, input_vector, output_vector)

    def forward_propagation(self, input_vector):
        """
        Forward propagation performs the prediction of the neural network

        Args:
            input_vector (Vector): the input vector to predict

        This uses the equation:
            z = sigmoid(w * x + b)

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s
        """

        self.outputs = [input_vector]
        self.deactivated_outputs = [input_vector]

        # iterates through each layer and performs the forward propagation calculation
        for layer in self.network:
            output = layer.forward_propagation(self.outputs[-1])
            self.deactivated_outputs.append(output[0])
            self.outputs.append(output[1])

        return output[1]
