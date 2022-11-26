"""
Backwards propagation is a training method for neural networks
and this file holds some of those function
"""

from vector import Vector

class BackProp():
    """
    This class is intended to be aggregated by the main
    NeuralNetwork class in network.py

    TODO: Work on the functions and structure
    TODO: Test this file structure
    """

    def cost_function_derivative(self, network, predicted_vector, output_vector):
        """
        All of the derivatives with the respect to the cost function are calculated here
        and given to the private object - derivatives.

        Each derivative can then be indexed simply via network.derivatives[layer][node]

        Args:
            network (Network): the network which is being trained
            predicted_vector (Vector): the actual output of the network
            output (Vector): the expected output of the network
        """

        network.derivatives = [list((predicted_vector - output_vector) * 2)]

        for layer in range(len(network.network)-1):
            delc_dela = []
            for weight in network.network[layer].weights:
                total = 0
                for j in range(len(network.derivatives[0])):
                    temporary_answer = weight[j]
                    temporary_answer *= network.derivative_func(network.outputs[layer][j])
                    total += temporary_answer * network.derivatives[0][j]
                delc_dela.append(total)
            network.derivatives = [delc_dela, *network.derivatives]

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
                for j in range(len(network.derivatives[layer])):
                    # each derivative is in a seperate line to keep the line size small
                    temporary_answer = network.outputs[layer][i]
                    temporary_answer *= network.derivative_func(
                        network.deactivated_outputs[layer+1][j])
                    delc_delwj.append(temporary_answer * network.derivatives[layer][j])
                delc_delwi.append(delc_delwj)
            weight_grad.append(delc_delwi)

        # with the gradient calculated in a multi dimensional array,
        # we iterate through each layer and node then update the weights
        for layer, layer_grad in enumerate(weight_grad):
            for node, node_grad in enumerate(list(zip(*layer_grad))):
                network.network[layer].weights[node] -= Vector(node_grad) * network.learn_rate

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
            for j in range(len(network.derivatives[layer])):
                # each derivative is in a seperate line to keep the line size small
                temporary_answer = network.derivative_func(network.deactivated_outputs[layer+1][j])
                delc_delwi.append(temporary_answer * network.derivatives[layer][j])
            bias_grad.append(delc_delwi)

        # with the gradient calculated in a multi dimensional array,
        # we iterate through each layer to update the biases
        for layer, gradient in enumerate(bias_grad):
            network.network[layer].biases -= Vector(gradient) * network.learn_rate

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
        network.cost_function_derivative(predicted_vector, output_vector)

        # updates the networks properties
        network.update_weights()
        network.update_biases()
