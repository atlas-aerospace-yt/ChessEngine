"""
Holds mathematical activation functions for vector objects.
"""

from vector import Vector

E = 2.718281828459
PI = 3.141592653589


def sigmoid(num):
    """
    Applies the sigmoid function to the input

    Args:
        num (int or float or Vector): The object to apply the sigmoid curve to
    Return:
        int or float or Vector: the result of the sigmoid funciton
    """
    if isinstance(num, Vector):
        return Vector([1 / (1 + E ** (-num[i])) for i in range(len(num))])
    if isinstance(num, (int, float)):
        return 1 / (1 + E ** (-num))

    return 0


def sigmoid_prime(num):
    """
    Applies the sigmoid prime function to the input

    Args:
        num (int or float or Vector): The object to apply the sigmoid prime curve to
    Return:
        int or float or Vector: the result of the sigmoid prime funciton
    """

    if isinstance(num, (int, float)):
        return sigmoid(num) * (1 - sigmoid(num))

    temp_list = []
    for value in num:
        result = sigmoid(value) * (1-sigmoid(value))
        temp_list.append(result)

    return Vector(temp_list)


def linear(num):
    """
    Linear actvation function y=kx

    Args:
        num (int or float or Vector): The object to apply the weight to
    Return:
        int or float or Vector: the result of the weight
    """
    if isinstance(Vector):
        return Vector(num * 0.5)

    return num * 0.5


def linear_prime(num):
    """
    Linear actvation function y=kx

    Args:
        num (int or float or Vector): The object to apply the derivative to
    Return:
        int or float or Vector: the result of the derivative
    """
    if isinstance(num, Vector):
        return Vector([float(0.5) for i in range(len(num))])

    return 0.5
