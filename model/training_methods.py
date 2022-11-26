"""
Backwards propagation is a training method for neural networks
and this file holds some of those function

TODO: move any variables specific to BackProp here
"""

from vector import Vector

class BackProp():
    """
    This class is intended to be aggregated by the main
    NeuralNetwork class in network.py

    TODO: Test this file structure

    Attributes:
        outputs (list): the output of each layer
        deactivated_outputs (list): the output of each layer before being activated
        derivatives (list): the cost function derivative with respect to each layer input
        lear_rate (float): the multiplier of the gradient to modify the weights

    Methods:
        cost_function_derivative:
            All of the derivatives with the respect to the cost function are calculated here
            and given to the private object - derivatives.

            Each derivative can then be indexed simply via self.derivatives[layer][node]

            Args:
                network (Network): the network which is being trained
                predicted_vector (Vector): the actual output of the network
                output (Vector): the expected output of the network

        update_weights:
            Updates the networks weights

            This is done using the equation:
                delC_0 / delw_ij^L = delz_j^L / delw_jk^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

            https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s

            Args:
                network (Network): the network which is being trained

        update_biases:
            Updates the networks biases

            This is done using the equation:
                delC_0 / delb_j^L = delz_j^L / delb_j^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

            note: delz_j^L / delb_j^L = 1

            https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s

            Args:
                network (Network): the network which is being trained
    """

    def __init__(self, derivative_func):

        self.derivative_func = derivative_func

        self.outputs = []
        self.deactivated_outputs = []
        self.derivatives = []
        self.learn_rate = 0.1

    def cost_function_derivative(self, network, predicted_vector, output_vector):
        """
        All of the derivatives with the respect to the cost function are calculated here
        and given to the private object - derivatives.

        Each derivative can then be indexed simply via self.derivatives[layer][node]

        Args:
            network (Network): the network which is being trained
            predicted_vector (Vector): the actual output of the network
            output (Vector): the expected output of the network
        """

        self.derivatives = [list((predicted_vector - output_vector) * 2)]

        # iterates through each layer in the network
        # then through each node and then goes through
        # each input that feeds the node and calculates the
        # gradient of the weights against the input
        # j is the insput
        for layer in range(len(network.network)-1):
            delc_dela = []
            for weight in network.network[layer].weights:
                total = 0
                for j in range(len(self.derivatives[0])):
                    # each derivative is in a seperate line to keep the line size small
                    temporary_answer = weight[j]
                    temporary_answer *= self.derivative_func(self.outputs[layer][j])
                    total += temporary_answer * self.derivatives[0][j]
                delc_dela.append(total)
            self.derivatives = [delc_dela, *self.derivatives]

    def update_weights(self, network):
        """
        Updates the networks weights

        This is done using the equation:
            delC_0 / delw_ij^L = delz_j^L / delw_jk^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s

        Args:
            network (Network): the network which is being trained
        """

        weight_grad = []

        # iterates through each layer in the network
        # then through each node and then goes through
        # each weight that feeds the node and calculates the
        # gradient of the weights against the cost function
        # j is the node and i is the weight
        for layer, network_layer in enumerate(network.network):
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
                network.network[layer].weights[node] -= Vector(node_grad) * self.learn_rate

    def update_biases(self, network):
        """
        Updates the networks biases

        This is done using the equation:
            delC_0 / delb_j^L = delz_j^L / delb_j^L * dela_j^L / delz_j^L * delC_0 / dela_j^L

        note: delz_j^L / delb_j^L = 1

        https://www.youtube.com/watch?v=tIeHLnjs5U8&t=499s

        Args:
            network (Network): the network which is being trained
        """

        # iterates through each layer in the network
        # then through each node to calculate the gradient with
        # respect to the cost function
        # j is the node
        bias_grad = []
        for layer in range(len(network.network)):
            delc_delwi = []
            for j in range(len(self.derivatives[layer])):
                # each derivative is in a seperate line to keep the line size small
                temporary_answer = self.derivative_func(self.deactivated_outputs[layer+1][j])
                delc_delwi.append(temporary_answer * self.derivatives[layer][j])
            bias_grad.append(delc_delwi)

        # with the gradient calculated in a multi dimensional array,
        # we iterate through each layer to update the biases
        for layer, gradient in enumerate(bias_grad):
            network.network[layer].biases -= Vector(gradient) * self.learn_rate

    def train_network(self, network, input_vector, output_vector):
        """
        Calls the update weights and biases functions

        TODO: return a list of costs
        TODO: loop within the function (EPOCH)
        TODO: allow the user to pass in the data seperately or grouped

        Args:
            network (Network): the network which is being trained
            input_vector (Vector): the input to the network
            output_vector (Vector): the wanted output
        """

        # gets the prediction from the network and calculates the cost derivative
        predicted_vector = network.forward_propagation(input_vector)
        self.cost_function_derivative(network, predicted_vector, output_vector)

        # updates the networks properties
        self.update_weights(network)
        self.update_biases(network)
