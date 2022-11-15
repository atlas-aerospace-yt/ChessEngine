"""
Main machine learning files

This file will hold the neural network and the learning
algorithm

TODO -> Re-write file_manager
TODO -> Create a neural network
"""

import vector as vec
from vector import Vector as v


class Learn():
    """
    This class holds the weights and biases as well as some operators
    for machine learning / neural networks.
    """

    def __init__(self):
        self.weight_one, self.bias_one = vec.random_vector(
            8), vec.random_vector(8)


learning = Learn()

if __name__ == "__main__":
    a = v([1, 2, 3, 4])
    b = v([5, 6, 7, 8])

    print(vec.random_vector(4))

    print(a * b)
