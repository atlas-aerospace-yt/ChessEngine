"""
Random vector generator code.

Generally intended to be used to randomly select initial weights and biases of a network.
"""

import random

from vector import Vector


def random_vector(num, lower=-0.1, upper=0.1):
    """
    Generates a random array

    Args:
        num (int): length of the array
        lower (float): minimum value in the vector
        upper (float): maximum value in the vector
    Return:
        Vector object
    """
    return Vector([random.uniform(lower, upper) for i in range(num)])
