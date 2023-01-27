"""
A file handler for the model library in python
"""

import os

from vector import Vector
from model.layer import Layer

class FileManager():
    """
    TODO
    """

    def __init__(self, size, dir="./Data"):

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

        self.size = size
        self.weights_dir = f"{dir}/Weights.txt"
        self.biases_dir = f"{dir}/Biases.txt"

    def load_network(self):
        """
        Loads both weights and biases

        Returns:
            vector: the biases
            list: a list of vectors which are the weights
        """

        with open(self.weights_dir, "r") as weights:
            data = weights.readlines()
        if data != []: weights = self.load_weights(data)
        print(weights)

        with open(self.biases_dir, "r") as biases:
            data = biases.readlines()
        if data != []: biases = self.load_biases(data)

    def load_weights(self, weight):
        """
        Loads the weights from a text file
        Checks that the weights match the network dimensions
        """
        layer = []
        weights = []
        for line in weight:
            weights.append(Vector([float(item) for item in line.strip().split(", ")]))
            if not line.strip():
                layer.append(weights)
                weights = []

    def save_network(self, network):
        """
        Saves the weights and biases to the respective txt files.

        Args:
            network (list): a list of layers objects
        """

        with open(self.weights_dir, "w") as weights, open(self.biases_dir, "w") as biases:

            for layer in network:

                for weight_vector in layer.weights:
                    weights.write(str(weight_vector) + "\n")
                biases.write(str(layer.biases) + "\n")

                weights.write("\n")
                biases.write("\n")